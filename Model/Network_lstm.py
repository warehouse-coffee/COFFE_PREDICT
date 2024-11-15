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
        self.datas = self.datas[0:self.datas.shape[0]]
        self.labels = self.labels[0:len(self.labels)]
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.log = log

        # Cell state
        self.Long_mem = np.zeros((1, 1))
        self.short_mem = np.zeros((1, 1))

        # Forget layer
        self.WF = np.random.randn(1, self.datas.shape[1]) * np.sqrt(2 / (self.datas.shape[1] + 1))
        self.BF = np.zeros((1, 1))
        self.UF = np.random.uniform(-0.5, 0.5, (1, 1))

        # Input layer
        self.WI = np.random.randn(1, self.datas.shape[1]) * np.sqrt(2 / (self.datas.shape[1] + 1))
        self.BI = np.zeros((1, 1))
        self.UI = np.random.uniform(-0.5, 0.5, (1, 1))

        self.WG = np.random.randn(1, self.datas.shape[1]) * np.sqrt(2 / (self.datas.shape[1] + 1))
        self.BG = np.zeros((1, 1))
        self.UG = np.random.uniform(-0.5, 0.5, (1, 1))

        # Output layer
        self.WO = np.random.uniform(-0.5, 0.5, (1, self.datas.shape[1]))
        self.BO = np.zeros((1, 1))
        self.UO = np.random.uniform(-0.5, 0.5, (1, 1))

        self.now_unix = now_unix
        self.now_date = now_date

        self.clip_value = 5
        self.batch_size = 16

    def normalize_batch(self, batch_data):
        # Normalize each feature within the batch to have zero mean and unit variance
        mean = np.mean(batch_data, axis=0, keepdims=True)
        std = np.std(batch_data, axis=0, keepdims=True) + 1e-8  # Add epsilon to prevent division by zero
        normalized_data = (batch_data - mean) / std
        return normalized_data
    
    def forward(self, data):
        # data = data.reshape((-1, 1))
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
        return new_short_mem

    def run(self, today_data, isSave=False, filename=None):
        m = self.datas.shape[0]
        today_data = np.reshape(today_data, (-1, 1))

        num_batches = m // self.batch_size
        left_over = m % self.batch_size
        if left_over > 0:
            num_batches += 1

        accuracy = None
        Cost = None
        Costs = np.array([])
        Predicton = np.array([])

        for i in range(self.epochs):
            accuracy = 0
            Cost = 0
            for batch in range(num_batches):
                start_idx = batch * self.batch_size
                end_idx = start_idx + self.batch_size
                if batch == num_batches - 1 and left_over > 0:
                    end_idx = start_idx + left_over
                batch_data = self.datas[start_idx:end_idx]
                batch_labels = self.labels[start_idx:end_idx]

                # Initialize gradients to zero for batch
                dWF, dBF, dUF = np.zeros_like(self.WF), np.zeros_like(self.BF), np.zeros_like(self.UF)
                dWI, dBI, dUI = np.zeros_like(self.WI), np.zeros_like(self.BI), np.zeros_like(self.UI)
                dWG, dBG, dUG = np.zeros_like(self.WG), np.zeros_like(self.BG), np.zeros_like(self.UG)
                dWO, dBO, dUO = np.zeros_like(self.WO), np.zeros_like(self.BO), np.zeros_like(self.UO)

                batch_data = self.normalize_batch(batch_data)

                for data, label in zip(batch_data, batch_labels):
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
                    dWO += DZ_O @ data.T
                    dUO += DZ_O @ self.short_mem
                    dBO += DZ_O

                    DZ_F = Delta @ A_O @ (1 - np.power(np.tanh(new_Long_mem), 2)) @ self.Long_mem @ sigmoid(Z_F) @ (1 - sigmoid(Z_F))
                    dWF += DZ_F @ data.T
                    dUF += DZ_F @ self.short_mem
                    dBF += DZ_F

                    DZ_I = Delta @ A_O @ (1 - np.power(np.tanh(new_Long_mem), 2)) @ A_G @ sigmoid(Z_I) @ (1 - sigmoid(Z_I))
                    dWI += DZ_I @ data.T
                    dUI += DZ_I @ self.short_mem
                    dBI += DZ_I

                    DZ_G = Delta @ A_O @ (1 - np.power(np.tanh(new_Long_mem), 2)) @ A_I @ (1 - np.power(np.tanh(Z_G), 2))
                    dWG += DZ_G @ data.T
                    dUG += DZ_G @ self.short_mem
                    dBG += DZ_G

                    self.Long_mem = new_Long_mem
                    self.short_mem = new_short_mem

                    if i == self.epochs - 1:
                        temp = new_short_mem
                        Predicton = np.append(Predicton, temp)

                # CLIPPING
                dWO = np.clip(dWO, -self.clip_value, self.clip_value)
                dUO = np.clip(dUO, -self.clip_value, self.clip_value)
                dBO = np.clip(dBO, -self.clip_value, self.clip_value)

                dWF = np.clip(dWF, -self.clip_value, self.clip_value)
                dUF = np.clip(dUF, -self.clip_value, self.clip_value)
                dBF = np.clip(dBF, -self.clip_value, self.clip_value)

                dWI = np.clip(dWI, -self.clip_value, self.clip_value)
                dUI = np.clip(dUI, -self.clip_value, self.clip_value)
                dBI = np.clip(dBI, -self.clip_value, self.clip_value)

                dWG = np.clip(dWG, -self.clip_value, self.clip_value)
                dUG = np.clip(dUG, -self.clip_value, self.clip_value)
                dBG = np.clip(dBG, -self.clip_value, self.clip_value)

                # UPDATING WEIGHTS
                self.WO += - self.learn_rate * dWO / self.batch_size
                self.UO += - self.learn_rate * dUO / self.batch_size
                self.BO += - self.learn_rate * dBO / self.batch_size

                self.WF += - self.learn_rate * dWF / self.batch_size
                self.UF += - self.learn_rate * dUF / self.batch_size
                self.BF += - self.learn_rate * dBF / self.batch_size

                self.WI += - self.learn_rate * dWI / self.batch_size
                self.UI += - self.learn_rate * dUI / self.batch_size
                self.BI += - self.learn_rate * dBI / self.batch_size

                self.WG += - self.learn_rate * dWG / self.batch_size
                self.UG += - self.learn_rate * dUG / self.batch_size
                self.BG += - self.learn_rate * dBG / self.batch_size

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
                                        Long_mem=self.Long_mem, short_mem=self.short_mem,
                                        mean = np.mean(self.datas, axis=0, keepdims=True),
                                        std = np.std(self.datas, axis=0, keepdims=True),
                                        accuracy=accuracy)
                tommorow_predict = self.forward(today_data)
                Predicton = np.append(Predicton, tommorow_predict)
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

        self.Long_mem = file['Long_mem']
        self.short_mem = file['short_mem']

        self.mean = np.array([file['mean']])
        self.std = np.array([file['std']])

        self.now_unix = int(file['now_unix'])
        self.now_date = str(file['now_date'])
        self.accuracy = float(file['accuracy'])

    def predict(self, data):
        data = (data - self.mean) / self.std
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
