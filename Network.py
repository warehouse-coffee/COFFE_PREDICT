import numpy as np
import matplotlib.pyplot as plt


def Relu(vector):
    return np.maximum(0, vector)


def Derrivative(vector):
    array = np.array([0 if x <= 0 else 1 for x in vector], dtype=np.float32)
    array = array.reshape((-1, 1))
    return array


class Network_training:
    def __init__(self, datas, labels, epochs, learn_rate, lambd=0.01):
        self.datas = datas.astype('float32')
        self.labels = labels.astype('float32')
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.lambd = lambd

        # 2 hidden layers
        self.first_weight = np.random.uniform(-0.5, 0.5, (20, 7))
        self.first_bias = np.zeros((20, 1))
        self.sec_weight = np.random.uniform(-0.5, 0.5, (40, 20))
        self.sec_bias = np.zeros((40, 1))

        # Output layer
        self.result_weight = np.random.uniform(-0.5, 0.5, (1, 40))
        self.result_bias = np.zeros((1, 1))

    def run(self, isSave=False, filename=None):
        m = self.datas.shape[0]
        res = None
        accuracy = None
        Cost = None
        Costs = np.array([])
        Predicton = np.array([])

        for i in range(self.epochs):
            accuracy = 0
            Cost = 0
            for data, label in zip(self.datas, self.labels):
                data = data.reshape((7, 1))
                label = np.round(label, 2)
                label = label.reshape((1, 1))

                # Forward prop
                pre_First = self.first_weight @ data + self.first_bias
                first = Relu(pre_First)

                pre_sec = self.sec_weight @ first + self.sec_bias
                sec = Relu(pre_sec)

                res = self.result_weight @ sec + self.result_bias
                res = np.round(res, 2)

                if i == self.epochs - 1:
                    Predicton = np.append(Predicton, res)

                # Cost, Loss and Accucracy
                Loss = np.power(res - label, 2)
                Cost += Loss

                # print(res, label)

                accuracy += int(res[0][0] == label[0][0])

                # Back prop and Gradient Descent
                DZ_3 = (res - label) / m  # DZ_3 == DB_3
                DW_3 = DZ_3 @ sec.T
                self.result_weight += -self.learn_rate * DW_3
                self.result_bias += -self.learn_rate * DZ_3

                DZ_2 = self.result_weight.T @ DZ_3 * Derrivative(pre_sec)  # DZ_2 == DB_2
                DW_2 = DZ_2 @ first.T
                self.sec_weight += -self.learn_rate * DW_2
                self.sec_bias += -self.learn_rate * DZ_2

                DZ_1 = self.sec_weight.T @ DZ_2 * Derrivative(pre_First)  # DZ_1 == DB_1
                DW_1 = DZ_1 @ data.T
                self.first_weight += -self.learn_rate * DW_1
                self.first_bias += -self.learn_rate * DZ_1

            Cost = Cost / (2 * m)
            Costs = np.append(Costs, Cost)
            accuracy = round(accuracy / m * 100, 2)
            print('Epoch:', i, 'Cost:', Cost[0][0], 'Accu:', accuracy)

          
            if i == self.epochs - 1:
                if isSave:
                    if filename == None:
                        filename = str(accuracy)
                    np.savez_compressed('Model/' + filename + '_NEURAL', first_weight=self.first_weight, first_bias=self.first_bias,
                                        sec_weight=self.sec_weight, sec_bias=self.sec_bias,
                                        result_weight=self.result_weight, result_bias=self.result_bias)
                
                t = np.arange(0, len(Costs))
                plt.plot(t, Costs)
                plt.xlabel("Itterations")
                plt.ylabel("Cost")
                plt.show()

                t = np.arange(0, len(self.labels))
                plt.plot(t, self.labels)
                plt.plot(t, Predicton)
                plt.show()
                print(Predicton)
