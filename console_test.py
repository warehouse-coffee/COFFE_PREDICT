import json
import matplotlib.pyplot as plt
from Model.Utils import *
import math
import numpy as np
from Model.Network_lstm import Network_training, Network_running


f = open('Data/' + 'train' + '.json', 'r')
data = json.load(f)
f.close()

f = open('API/links.json', 'r')
Links = json.load(f)
f.close()
links_name = list(Links.keys())

trainObj = {}
trainData = np.array([])
DataToday = np.array([])
length_min = 0
name = None
train_now_date = data['date_now']
train_now_unix = data['unix_time_now']
WindowsTime = 14

for key in links_name:
    trainObj[key] = [i['value'] for i in data[key]]
    date_key = key + '_date'
    trainObj[date_key] = [i['date'] for i in data[key]]
    if length_min == 0:
        length_min = len(trainObj[key])
        name = key
    elif len(trainObj[key]) < 100:
        links_name.remove(key)
    elif len(trainObj[key]) < length_min:
        length_min = len(trainObj[key])
        name = key

for key in links_name:
    if len(trainData) == 0:
        index = len(trainObj[key]) - length_min
        scaler = MinMax(trainObj[key][index:])
        scaler = numpy_ewma(scaler, WindowsTime)
        DataToday = np.array(scaler[-1])
        scaler = scaler[:-1]
        scaler = scaler.reshape((-1, 1))
        trainData = np.array(scaler)
    else:
        index = len(trainObj[key]) - length_min
        scaler = MinMax(trainObj[key][index:])
        scaler = numpy_ewma(scaler, WindowsTime)
        DataToday = np.append(DataToday, scaler[-1])
        scaler = scaler[:-1]
        scaler = scaler.reshape((-1, 1))
        trainData = np.concatenate((trainData, scaler), axis=1)

print(trainData.shape)

print(length_min, name)

# print(trainData.keys())
index = len(trainObj['Coffee']) - length_min
scaler = trainObj['Coffee'][index:]
label_1 = np.array(scaler[1:])
label_2 = np.array(scaler[:-1])
label = label_1 - label_2
label = numpy_ewma(label, WindowsTime)
print(label.shape)

nwtwork = Network_training(trainData, label, 100, 0.01, log=True, now_date=train_now_date, now_unix=train_now_unix)
res = nwtwork.run(True, 'model')
while(int(res[1]) < 70):
    nwtwork = Network_training(trainData, label, 100, 0.01, log=True, now_date=train_now_date, now_unix=train_now_unix)
    res = nwtwork.run(True, 'model')

# pred = res[0]
# print(pred.shape)
# print("Acc:", res[1])
# # pred = MinMax_Negative(pred)
# # pred = numpy_ewma(pred, 7)
# # label = MinMax_Negative(label)
# plt.plot(pred)
# plt.plot(label)
# plt.show()

res = np.array([])
nwtwork = Network_running()
nwtwork.load_model('Model/models/model_LSTM.npz')
for Data in trainData:
    res = np.append(res, nwtwork.predict(Data))
res = np.append(res, nwtwork.predict(DataToday))
plt.plot(res)
plt.plot(label)
plt.show()