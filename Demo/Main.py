import csv
import numpy as np
import glob
import matplotlib.pyplot as plt
from Model.Utils import *
from Model.Network_lstm import Network_training

if __name__ == '__main__':
    datas, prices = Load_CSV('Data/Starbucks Dataset.csv', isave=False, filename='data_sr')
    # datas, prices = Load_npz('Data/data_tmp.npz')
    datas = datas.astype('float32')
    prices = prices.astype('float32')
    # prices_2 = datas[0:len(prices), 3]
    # datas = np.delete(datas, 3, 1)
    # print(datas.shape)
    # prices = (datas[0:datas.shape[0], 1] + datas[0:datas.shape[0], 2]) / 2

    window_size = 1000
    window_len = (datas.shape[0] // window_size) * window_size
    for i in range(0, window_len, window_size):
        datas[i:i + window_size, 0] = MinMax(datas[i:i + window_size, 0])
        datas[i:i + window_size, 1] = MinMax(datas[i:i + window_size, 1])
        datas[i:i + window_size, 2] = MinMax(datas[i:i + window_size, 2])
    datas[window_len:datas.shape[0], 0] = MinMax(datas[window_len:datas.shape[0], 0])
    datas[window_len:datas.shape[0], 1] = MinMax(datas[window_len:datas.shape[0], 1])
    datas[window_len:datas.shape[0], 2] = MinMax(datas[window_len:datas.shape[0], 2])

    # datas[0:datas.shape[0], 0] = EMA(datas[0:datas.shape[0], 0], window_size)
    # datas[0:datas.shape[0], 1] = EMA(datas[0:datas.shape[0], 1], window_size)
    # datas[0:datas.shape[0], 2] = EMA(datas[0:datas.shape[0], 2], window_size)

    # datas[0:datas.shape[0], 0] = MinMax(datas[0:datas.shape[0], 0])
    # datas[0:datas.shape[0], 1] = MinMax(datas[0:datas.shape[0], 1])
    # datas[0:datas.shape[0], 2] = MinMax(datas[0:datas.shape[0], 2])

    # datas[0:datas.shape[0], 3] = (datas[0:datas.shape[0], 3] + 5) / (10)
    # datas[0:datas.shape[0], 4] = (datas[0:datas.shape[0], 4]) / (14)
    # datas[0:datas.shape[0], 5] = (datas[0:datas.shape[0], 5] - 50) / (350)
    # datas[0:datas.shape[0], 6] = (datas[0:datas.shape[0], 6] - 0) / (0.14)

    
    prices = MinMax(prices)
    # prices = EMA(prices)
    # prices_2 = (prices_2 - np.min(prices_2[0:len(prices) - 1])) / (np.max(prices_2[0:len(prices) - 1]) + - np.min(prices_2[0:len(prices) - 1]))
    # prices = (prices - np.mean(prices)) / np.std(prices)
    # prices = prices/3.5
    # print(datas[0:datas.shape[0], 7])
    # print(prices[len(prices) - 1:0:-1].shape)
    # print(prices[0:len(prices) - 1, 0])
    # prices[0:len(prices) - 1, 0] =  -(prices[1:len(prices),0] + prices[0:len(prices) - 1, 0])
    # print(prices[0:20, 0])

    # print(prices[2:0:-1, 0])
    # t = np.arange(0, len(prices) - 1)
    # plt.plot(t, prices[0:len(prices) - 1, 0])
    # plt.plot(t, prices_2[0:len(prices) -1])
    # plt.show()
    network = Network_training(datas, prices, 30, 0.01)
    network.run(False)
