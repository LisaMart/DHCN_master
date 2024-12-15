# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix

# Function to create a sparse adjacency matrix from session data
def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []  # Prepare for CSR matrix format
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])  # Get unique items in each session
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)  # Adjust index to 0-based
            data.append(1)  # Assign value 1 to each item in session
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))  # Create sparse matrix

    return matrix

# Функция для добавления padding (выравнивание длины сессий)
def pad_data(data, max_len=50, pad_value=0):
    """
    Функция для добавления padding (выравнивание длины сессий).
    """
    processed_data = []
    for session in data:
        # Печать длины сессии для отладки
        print(f"Session length before padding: {len(session)}")

        if len(session) < max_len:
            # Добавляем padding, если сессия меньше max_len
            session = session + [pad_value] * (max_len - len(session))
        else:
            # Обрезаем, если сессия больше max_len
            session = session[:max_len]

        # Печать длины сессии после паддинга
        print(f"Session length after padding: {len(session)}")

        processed_data.append(session)

    return np.array(processed_data, dtype=object)

# Функция для создания маски
def create_mask(data, max_len=50):
    """
    Функция для создания маски: 1 для реальных элементов и 0 для padding.
    """
    mask = np.zeros((len(data), max_len), dtype=bool)
    for i, session in enumerate(data):
        mask[i, :len(session)] = 1  # Маска True для всех реальных элементов
    return mask

# Function to split the dataset into train and validation sets
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)  # Shuffle indices
    n_train = int(np.round(n_samples * (1. - valid_portion)))  # Calculate training set size
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]  # Validation set
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]  # Training set
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0])  # Raw session data
        # Create the adjacency matrix for sessions
        H_T = data_masks(self.raw, n_node)
        BH_T = H_T.T.multiply(1.0/H_T.sum(axis=1).reshape(1, -1))  # Normalize columns
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0/H.sum(axis=1).reshape(1, -1))  # Normalize rows
        DH = DH.T
        DHBH_T = np.dot(DH,BH_T)  # Compute final adjacency matrix

        self.adjacency = DHBH_T.tocoo()  # Store as sparse matrix (coordinate format)
        self.n_node = n_node  # Number of unique items
        self.targets = np.asarray(data[1])  # Targets (for supervised learning)
        self.length = len(self.raw)  # Number of sessions
        self.shuffle = shuffle  # Whether to shuffle data or not

    # Function to compute overlap matrix and degree matrix for sessions
    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))  # Initialize overlap matrix
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)  # Discard padding (0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)  # Calculate overlap
                ab_set = seq_a | seq_b  # Union of two sets
                matrix[i][j] = float(len(overlap))/float(len(ab_set))  # Similarity
                matrix[j][i] = matrix[i][j]  # Symmetric matrix
        matrix = matrix + np.diag([1.0]*len(sessions))  # Add self-similarity (diagonal)
        degree = np.sum(np.array(matrix), 1)  # Degree of nodes (sum of similarities)
        degree = np.diag(1.0/degree)  # Inverse degree matrix
        return matrix, degree

    # Function to generate batches for training
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)  # Shuffle the dataset
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)  # Number of batches
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)  # Split data into batches
        slices[-1] = np.arange(self.length-batch_size, self.length)  # Handle last batch
        return slices

    # Function to get a specific slice of data for a batch
    def get_slice(self, index):
        items, num_node = [], []  # Initialize lists to store session data
        inp = self.raw[index]  # Get the raw data slice
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))  # Get the number of non-zero elements (active items)
        max_n_node = np.max(num_node)  # Find the maximum session length
        session_len = []  # List to store session lengths
        reversed_sess_item = []  # List to store reversed session items
        mask = []  # List to store masks for padding
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]  # Get non-zero elements (active items)
            session_len.append([len(nonzero_elems)])  # Store session length
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])  # Pad session
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])  # Pad with mask
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])  # Reverse session items

        return self.targets[index]-1, session_len, items, reversed_sess_item, mask