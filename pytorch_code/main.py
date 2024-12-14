# -*- coding: utf-8 -*-

import argparse
import pickle
from util import Data, split_validation
from model import *
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/Tmall/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)

def main():
    """
    Main function to load the dataset, initialize the model, and start training.
    """

    # SERVER
    #train_data = pickle.load(open('/root/DHCN_master/datasets/' + opt.dataset + '/train.txt', 'rb'))
    #test_data = pickle.load(open('/root/DHCN_master/datasets/' + opt.dataset + '/test.txt', 'rb'))

    # LOCAL - Абсолютный путь
    #train_data = pickle.load(open('C:/Users/lisa/python_practice/DHCN_master/datasets/' + opt.dataset + '/train.txt', 'rb'))

    # LOCAL - Относительный путь
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

    # Determine the number of nodes based on the selected dataset
    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    else:
        n_node = 309

    # Initialize the Data class with the training and test data
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=True, n_node=n_node)

    # Initialize the DHCN model and move it to GPU (if available)
    model = trans_to_cuda(DHCN(adjacency=train_data.adjacency,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))

    # Top K values for evaluating metrics
    top_K = [5, 15, 20]

    # Initialize a dictionary to store the best results for each K
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    # Training loop for the specified number of epochs
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: %d/%d' % (epoch + 1, opt.epoch))

        # Get metrics and total loss for the current epoch
        metrics, total_loss = train_test(model, train_data, test_data)

        # Process the metrics for each K value
        for K in top_K:
            metrics['precision%d' % K] = np.mean(metrics['precision%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100

            # Update the best results for Precision@K
            if best_results['metric%d' % K][2] < metrics['precision%d' % K]:
                best_results['metric%d' % K][2] = metrics['precision%d' % K]
                best_results['epoch%d' % K][2] = epoch

            # Update the best results for MRR@K
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch

        # Print the metrics for the current epoch
        print(metrics)

        # Print the best results for each metric (Precision@K and MRR@K) for each K value
        for K in top_K:
            print('train_loss:\t%.4f\ttPrecision@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))

if __name__ == '__main__':
    main()