# -*- coding: utf-8 -*-

import argparse
import pickle
from util import Data, split_validation, pad_data, create_mask
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

def check_and_fix_data_integrity(data, n_node):
    """
    Функция для проверки и исправления целостности данных.
    Убедитесь, что все индексы положительные и находятся в пределах от 1 до n_node.
    Если индексы начинаются с 0, сдвигаем их на 1.
    """
    for i, session in enumerate(data[0]):  # Проверка только первой части данных (сессии)
        # Если индексы начинаются с 0 или отрицательные, сдвигаем на 1
        if any(i <= 0 for i in session):
            print(f"Invalid index found in session {i+1}: {session}. Shifting indices.")
            data[0][i] = [x + 1 if x <= 0 else x for x in session]  # Сдвигаем все индексы на 1
    return data

# Проверка целостности данных
def check_data_integrity(data, n_node):
    """
    Функция для проверки целостности данных.
    Убедитесь, что все индексы положительные и находятся в пределах от 1 до n_node.
    """
    for session in data[0]:  # Проверка только первой части данных (сессии)
        if any(i <= 0 or i > n_node for i in session):  # Проверка на отрицательные и нулевые индексы
            print(f"Invalid index found in session: {session}")
            return False
    return True

def load_and_process_data(train_data, test_data, max_len=50):
    """
    Функция для обработки данных: паддинг и создание масок.
    """
    # Применяем padding для сессий
    train_data_processed = pad_data(train_data[0], max_len=max_len)  # Паддинг для тренировочных данных
    test_data_processed = pad_data(test_data[0], max_len=max_len)    # Паддинг для тестовых данных

    # Создаем маски для данных (если нужно)
    train_mask = create_mask(train_data_processed, max_len=max_len)
    test_mask = create_mask(test_data_processed, max_len=max_len)

    return train_data_processed, test_data_processed, train_mask, test_mask

def main():
    """
    Main function to load the dataset, initialize the model, and start training.
    """

    # SERVER
    train_data = pickle.load(open('/root/DHCN_master/datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('/root/DHCN_master/datasets/' + opt.dataset + '/test.txt', 'rb'))

    # LOCAL - Относительный путь
    #train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    #test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

    # Определяем количество узлов на основе выбранного датасета
    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    else:
        n_node = 309
    # Проверка и исправление целостности данных
    train_data = check_and_fix_data_integrity(train_data, n_node)
    test_data = check_and_fix_data_integrity(test_data, n_node)

    # Загружаем и обрабатываем данные
    train_data_processed, test_data_processed, train_mask, test_mask = load_and_process_data(train_data, test_data)

    # Инициализация класса Data с обработанными данными
    train_data = Data(train_data_processed, shuffle=True, n_node=n_node)
    test_data = Data(test_data_processed, shuffle=True, n_node=n_node)

    # Инициализация модели DHCN и перенос ее на GPU (если доступно)
    model = trans_to_cuda(
        DHCN(adjacency=train_data.adjacency, n_node=n_node, lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,
             emb_size=opt.embSize, batch_size=opt.batchSize, dataset=opt.dataset))

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