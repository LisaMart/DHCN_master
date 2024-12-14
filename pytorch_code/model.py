# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.sparse
from numba import jit
import heapq

# Function to move a tensor to GPU if available, otherwise return the tensor on CPU
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

# Function to move a tensor to CPU if it is on GPU
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

# Class representing Hyperbolic Convolution (HyperConv) operation
class HyperConv(Module):
    def __init__(self, layers,dataset,emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size  # Embedding size
        self.layers = layers  # Number of layers in the hyperbolic graph
        self.dataset = dataset  # Dataset name

    def forward(self, adjacency, embedding):
        item_embeddings = embedding # Initial item embeddings
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]

        # Apply multiple layers of graph convolutions
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings) # Sparse matrix multiplication
            final.append(item_embeddings)

        # Average the embeddings over all layers
        item_embeddings = np.sum(final, 0) / (self.layers+1)
        return item_embeddings

# Class representing Line Graph Convolution (LineConv) operation
class LineConv(Module):
    def __init__(self, layers,batch_size,emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size  # Embedding size
        self.batch_size = batch_size  # Batch size
        self.layers = layers  # Number of layers for LineConv

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)  # Zero vector for padding
        item_embedding = torch.cat([zeros, item_embedding], 0)  # Padding the item embeddings
        seq_h = []  # Sequence of session embeddings

        # Create session embeddings by selecting item embeddings for each session
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))

        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h])) # Transfer to CUDA
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)  # Aggregate session embeddings

        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()  # Matrix multiplication of D and A for LineConv

        # Apply multiple layers of LineConv
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)

        session_emb_lgcn = np.sum(session, 0) / (self.layers+1)
        return session_emb_lgcn

# Main model class for DHCN
class DHCN(Module):
    def __init__(self, adjacency, n_node,lr, layers,l2, beta,dataset,emb_size=100, batch_size=100):
        super(DHCN, self).__init__()
        self.emb_size = emb_size  # Embedding size
        self.batch_size = batch_size  # Batch size
        self.n_node = n_node  # Number of nodes in the dataset
        self.L2 = l2  # L2 regularization
        self.lr = lr  # Learning rate
        self.layers = layers  # Number of layers
        self.beta = beta  # SSL magnitude
        self.dataset = dataset  # Dataset name

        # Prepare the adjacency matrix for sparse matrix multiplication
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.adjacency = adjacency # Store adjacency matrix

        # Initialize the embedding layers
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_embedding = nn.Embedding(200, self.emb_size)  # Positional embedding
        self.HyperGraph = HyperConv(self.layers, dataset)  # HyperConv layer
        self.LineGraph = LineConv(self.layers, self.batch_size)  # LineConv layer

        # Linear layers for computing attention scores
        self.w_1 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))  # Parameterized linear layer
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()  # Loss function for classification
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # Optimizer (Adam)

        self.init_parameters()  # Initialize the model parameters

    # Initialize model parameters with a uniform distribution
    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # Function to generate session embeddings considering position
    def generate_sess_emb(self,item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len) # Aggregate session embeddings

        # Applying positional embeddings and linear transformations
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)  # Select the final session embedding
        return select

    # Function to generate session embeddings without positional embeddings
    def generate_sess_emb_npos(self,item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len) # Aggregate session embeddings
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1) # Select the final session embedding
        return select

    # Function to calculate the contrastive loss (SSL task)
    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)  # Calculate dot product between embeddings

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    # Forward pass: compute item embeddings and session embeddings
    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask):
        item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)
        if self.dataset == 'Tmall':
            sess_emb_hgnn = self.generate_sess_emb_npos(item_embeddings_hg, session_item, session_len, reversed_sess_item, mask)
        else:
            sess_emb_hgnn = self.generate_sess_emb(item_embeddings_hg, session_item, session_len, reversed_sess_item, mask)
        session_emb_lg = self.LineGraph(self.embedding.weight, D, A, session_item, session_len)
        con_loss = self.SSL(sess_emb_hgnn, session_emb_lg) # Contrastive loss for SSL
        return item_embeddings_hg, sess_emb_hgnn, self.beta*con_loss

# Function to find the top K largest values from a list of candidates
@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]  # Return the indices of the top K items
    return ids  # Return only the indices of the top K items

# Function to perform the forward pass for training and testing
def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, sess_emb_hgnn, con_loss = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask)
    scores = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1,0))  # Compute the scores between session and item embeddings
    return tar, scores, con_loss  # Return targets, scores, and contrastive loss


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)  # Initialize total loss to zero
    total_loss = 0.0 # Initialize total loss to zero
    slices = train_data.generate_batch(model.batch_size) # Generate batches of training data

    # Training loop
    for i in slices:
        model.zero_grad() # Zero the gradients of the model before the backward pass
        targets, scores, con_loss = forward(model, i, train_data) # Get the targets, scores, and contrastive loss from the forward pass

        # Compute the loss using the loss function and add the contrastive loss
        loss = model.loss_function(scores + 1e-8, targets)  # CrossEntropy loss
        loss = loss + con_loss  # Add the contrastive loss
        loss.backward()  # Backpropagate the loss
        model.optimizer.step()  # Update the model parameters
        total_loss += loss# Accumulate the total loss for the current batch
    print('\tLoss:\t%.3f' % total_loss)  # Print the total loss for the epoch

    # Initialize a dictionary to store metrics for different top-K values
    top_K = [5, 15, 20]  # Define the top-K values for evaluation
    metrics = {}  # Dictionary to store the metrics for Precision and MRR
    for K in top_K:
        metrics['precision%d' % K] = []  # Precision for each top-K
        metrics['mrr%d' % K] = []  # MRR for each top-K

    # Start the prediction phase and print the current time
    print('start predicting: ', datetime.datetime.now())

    model.eval() # Set the model to evaluation mode (disable dropout, etc.)
    slices = test_data.generate_batch(model.batch_size) # Generate batches of test data

    # Prediction loop
    for i in slices:
        tar, scores, con_loss = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()

        # Calculate metrics for each K in top_K
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                # Calculate MRR@K
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                # Calculate Precision@K
                correct_predictions = len(np.intersect1d(prediction, target))  # Количество правильных предсказаний
                precision = correct_predictions / K  # Precision@K
                metrics['precision%d' % K].append(precision)
    return metrics, total_loss