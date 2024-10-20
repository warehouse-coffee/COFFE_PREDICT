import csv
import numpy as np
import glob
import pandas as pd


def Load_CSV(path, isave=False, filename='data'):
    df = pd.read_csv(path)
    array = np.append([], df[['Open', 'High', 'Low']].values)
    label = np.append([], df['Close'])

    array = array.reshape((-1, 3))
    label = label.reshape((-1, 1))
    if isave:
        exist = glob.glob('Data/' + filename + '*.npz')
        if len(exist) > 0:
            filename += ' (' + str(len(exist)) + ')'
        np.savez_compressed('Data/' + filename, x_train=array, y_train=label)
    return array, label
    # print(array[0])
    # print(label[0])


def Load_npz(path):
    file = np.load(path)
    # print(file.files)
    array = file['x_train']
    labels = file['y_train']
    return array, labels


def MinMax(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def MinMax_Negative(array):
    return 2 * (array - np.min(array)) / (np.max(array) - np.min(array)) - 1


def RSI(array, alpha) -> np.ndarray:
    '''
    Returns the relative strength index of x.

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}

    Returns:
    --------
    rsi: numpy array
         the exponentially weighted moving average
    '''
    # Coerce x to an array
    n = array.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n, n)) * (1 - alpha)


def numpy_ewma(data, window):
    returnArray = np.empty((data.shape[0]))
    returnArray.fill(np.nan)
    e = data[0]
    alpha = 2 / float(window + 1)
    for s in range(data.shape[0]):
        e = ((data[s] - e) * alpha) + e
        returnArray[s] = e
    return returnArray
