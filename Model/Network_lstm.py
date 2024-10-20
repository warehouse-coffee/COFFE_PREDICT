import numpy as np
import matplotlib.pyplot as plt
from .Utils import *
import datetime
import time


def Relu(vector):
    return np.maximum(0, vector)


def Derrivative(vector):
    array = np.array([0 if x <= 0 else 1 for x in vector], dtype=np.float32)
    array = array.reshape((-1, 1))
    return array


def sigmoid(vector):
    a = 1 / (1 + np.exp(-vector))
    return a


def normalize(vector):
    a = (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return a


class Network_training:
    def __init__(self, datas, labels, epochs, learn_rate, log=True, now_unix=time.time(), now_date=datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d')):
        self.datas = datas.astype('float32')
        self.labels = labels.astype('float32')
        self.datas = self.datas[0:self.datas.shape[0] - 1]
        self.labels = self.labels[0:len(self.labels) - 1]
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.log = log

        # Cell state
        self.Long_mem = np.zeros((1, 1))
        self.short_mem = np.zeros((1, 1))

        # Forget layer
        self.WF = np.random.uniform(-0.5, 0.5, (1, self.datas.shape[1]))
        self.BF = np.zeros((1, 1))
        self.UF = np.random.uniform(-0.5, 0.5, (1, 1))

        # Input layer
        self.WI = np.random.uniform(-0.5, 0.5, (1, self.datas.shape[1]))
        self.BI = np.zeros((1, 1))
        self.UI = np.random.uniform(-0.5, 0.5, (1, 1))

        self.WG = np.random.uniform(-0.5, 0.5, (1, self.datas.shape[1]))
        self.BG = np.zeros((1, 1))
        self.UG = np.random.uniform(-0.5, 0.5, (1, 1))

        # Output layer
        self.WO = np.random.uniform(-0.5, 0.5, (1, self.datas.shape[1]))
        self.BO = np.zeros((1, 1))
        self.UO = np.random.uniform(-0.5, 0.5, (1, 1))

        self.now_unix = now_unix
        self.now_date = now_date

    def run(self, isSave=False, filename=None):
        m = self.datas.shape[0]
        accuracy = None
        Cost = None
        Costs = np.array([])
        Predicton = np.array([])

        for i in range(self.epochs):
            accuracy = 0
            Cost = 0
            for data, label in zip(self.datas, self.labels):
                data = data.reshape((-1, 1))
                # data = normalize(data)
                # label = np.round(label, 2)
                label = label.reshape((1, 1))

                # Forward prop
                Z_F = self.WF @ data + self.UF @ self.short_mem + self.BF
                A_F = sigmoid(Z_F)  # Forget

                temp_forget = self.Long_mem @ A_F

                Z_I = self.WI @ data + self.UI @ self.short_mem + self.BI
                A_I = sigmoid(Z_I)  # Input

                Z_G = self.WG @ data + self.UG @ self.short_mem + self.BG
                A_G = np.tanh(Z_G)  # Input

                temp_input = A_I @ A_G
                new_Long_mem = temp_forget + temp_input

                Z_O = self.WO @ data + self.UO @ self.short_mem + self.BO
                A_O = sigmoid(Z_O)  # Out

                new_short_mem = A_O @ np.tanh(new_Long_mem)

                # Cost, Loss and Accucracy
                Loss = np.power(new_short_mem - label, 2)
                Cost += Loss

                # print(new_short_mem, label)

                accuracy += 1 if new_short_mem[0][0] > 0 and label[0][0] > 0 or new_short_mem[0][0] < 0 and label[0][0] < 0 else 0

                # Back prop and Gradient Descent
                Delta = new_short_mem - label
                DZ_O = Delta @ sigmoid(Z_O) @ (1 - sigmoid(Z_O)) @ np.tanh(new_Long_mem)
                DW_O = DZ_O @ data.T
                DU_O = DZ_O @ self.short_mem
                self.WO += -self.learn_rate * DW_O
                self.UO += - self.learn_rate * DU_O
                self.BO += - self.learn_rate * DZ_O

                DZ_F = Delta @ A_O @ (1 - np.power(np.tanh(new_Long_mem), 2)) @ self.Long_mem @ sigmoid(Z_F) @ (1 - sigmoid(Z_F))
                DW_F = DZ_F @ data.T
                DU_F = DZ_F @ self.short_mem
                self.WF += -self.learn_rate * DW_F
                self.UF += - self.learn_rate * DU_F
                self.BF += - self.learn_rate * DZ_F

                DZ_I = Delta @ A_O @ (1 - np.power(np.tanh(new_Long_mem), 2)) @ A_G @ sigmoid(Z_I) @ (1 - sigmoid(Z_I))
                DW_I = DZ_I @ data.T
                DU_I = DZ_I @ self.short_mem
                self.WI += -self.learn_rate * DW_I
                self.UI += - self.learn_rate * DU_I
                self.BI += - self.learn_rate * DZ_I

                DZ_G = Delta @ A_O @ (1 - np.power(np.tanh(new_Long_mem), 2)) @ A_I @ (1 - np.power(np.tanh(Z_G), 2))
                DW_G = DZ_G @ data.T
                DU_G = DZ_G @ self.short_mem
                self.WG += -self.learn_rate * DW_G
                self.UG += - self.learn_rate * DU_G
                self.BG += - self.learn_rate * DZ_G

                self.Long_mem = new_Long_mem
                self.short_mem = new_short_mem

                if i == self.epochs - 1:
                    temp = new_short_mem
                    Predicton = np.append(Predicton, temp)

            Cost = Cost / (2 * m)
            Costs = np.append(Costs, Cost)
            accuracy = round(accuracy / m * 100, 2)
            if (self.log):
                print('Epoch:', i, 'Cost:', Cost[0][0], 'Accu:', accuracy)

            if i == self.epochs - 1:
                if isSave:
                    if filename == None:
                        filename = str(accuracy)
                    np.savez_compressed('Model/models/' + filename + '_LSTM',
                                        WF=self.WF, UF=self.UF, BF=self.BF,
                                        WI=self.WI, UI=self.UI, BI=self.BI,
                                        WG=self.WG, UG=self.UG, BG=self.BG,
                                        WO=self.WO, UO=self.UO, BO=self.BO,
                                        now_unix=self.now_unix, now_date=self.now_date,
                                        accuracy=accuracy)

                return [Predicton, accuracy]


class Network_running:
    def __init__(self):
        # Cell state
        self.Long_mem = np.zeros((1, 1))
        self.short_mem = np.zeros((1, 1))

        # Forget layer
        self.WF = np.random.uniform(-0.5, 0.5, (1, 49))
        self.BF = np.zeros((1, 1))
        self.UF = np.random.uniform(-0.5, 0.5, (1, 1))

        # Input layer
        self.WI = np.random.uniform(-0.5, 0.5, (1, 49))
        self.BI = np.zeros((1, 1))
        self.UI = np.random.uniform(-0.5, 0.5, (1, 1))

        self.WG = np.random.uniform(-0.5, 0.5, (1, 49))
        self.BG = np.zeros((1, 1))
        self.UG = np.random.uniform(-0.5, 0.5, (1, 1))

        # Output layer
        self.WO = np.random.uniform(-0.5, 0.5, (1, 49))
        self.BO = np.zeros((1, 1))
        self.UO = np.random.uniform(-0.5, 0.5, (1, 1))

    def load_model(self, path):
        file = np.load(path)
        self.WF = file['WF']
        self.WI = file['WI']
        self.WG = file['WG']
        self.WO = file['WO']

        self.UF = file['UF']
        self.UI = file['UI']
        self.UG = file['UG']
        self.UO = file['UO']

        self.BF = file['BF']
        self.BI = file['BI']
        self.BG = file['BG']
        self.BO = file['BO']

        self.now_unix = int(file['now_unix'])
        self.now_date = str(file['now_date'])
        self.accuracy = float(file['accuracy'])

    def predict(self, data):
        data = data.reshape((-1, 1))

        # Forward prop
        Z_F = self.WF @ data + self.UF @ self.short_mem + self.BF
        A_F = sigmoid(Z_F)  # Forget

        temp_forget = self.Long_mem @ A_F

        Z_I = self.WI @ data + self.UI @ self.short_mem + self.BI
        A_I = sigmoid(Z_I)  # Input

        Z_G = self.WG @ data + self.UG @ self.short_mem + self.BG
        A_G = np.tanh(Z_G)  # Input

        temp_input = A_I @ A_G
        new_Long_mem = temp_forget + temp_input

        Z_O = self.WO @ data + self.UO @ self.short_mem + self.BO
        A_O = sigmoid(Z_O)  # Out

        new_short_mem = A_O @ np.tanh(new_Long_mem)
        Predicton = new_short_mem

        return Predicton

    def get_status(self):
        return {
            "now_unix": self.now_unix,
            "now_date": self.now_date,
            "accuracy": self.accuracy
        }
