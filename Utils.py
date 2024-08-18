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
