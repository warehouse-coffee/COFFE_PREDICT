import json
import time
import datetime
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
    trainObj[key] = [i['real_price'] for i in data[key]]
    date_key = key + '_date'
    trainObj[date_key] = [i['date'] for i in data[key]]
    unix_ms_key = key + '_unix_ms'
    trainObj[unix_ms_key] = [i['unix_date_ms'] for i in data[key]]
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
date_obj_coffee = trainObj['Coffee_date'][index:]
unix_obj_coffee = trainObj['Coffee_unix_ms'][index:]
label_1 = np.array(scaler[1:])
label_2 = np.array(scaler[:-1])
label = label_1 - label_2
# label = MinMax_Negative(label)
label = numpy_ewma(label, WindowsTime)
# print(label.shape)
# print(trainData.shape)
print(DataToday.shape)
# print(len(scaler))

# nwtwork = Network_training(trainData, label, 100, 0.01, log=True, now_date=train_now_date, now_unix=train_now_unix)
# res = nwtwork.run(today_data=DataToday, isSave=True, filename='model')
# while (int(res[1]) < 70):
#     nwtwork = Network_training(trainData, label, 100, 0.01, log=True, now_date=train_now_date, now_unix=train_now_unix)
#     res = nwtwork.run(today_data=DataToday, isSave=True, filename='model')

# pred = res[0]
# print(label.shape)
# print(pred.shape)
# print("Acc:", res[1])
# # pred = MinMax_Negative(pred)
# # pred = numpy_ewma(pred, 7)
# # label = MinMax_Negative(label)
# plt.plot(pred)
# plt.plot(label)
# plt.plot(np.zeros(len(label)))
# plt.show()

# obj = {
#     "date_now": time.time(),
#     "unix_time_now": datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d'),
# }
# res_data = []
# for i in range(len(pred)):
#     if i == len(pred) - 1:
#         message = "Predict value for " + datetime.datetime.fromtimestamp(int(unix_obj_coffee[i] / 1000 + 24 * 60 * 60)).strftime('%Y-%m-%d')
#         res_data.append({
#             "index": i,
#             "AI_predict": pred[i],
#             "Real_price_difference_rate": 0,
#             "Date": date_obj_coffee[i],
#             "unix_date_ms": unix_obj_coffee[i],
#             "message": message
#         })
#     else:
#         res_data.append({
#             "index": i,
#             "AI_predict": pred[i],
#             "Real_price_difference_rate": label[i],
#             "Date": date_obj_coffee[i],
#             "unix_date_ms": unix_obj_coffee[i],
#             "message": "normal value"
#         })

# obj['data'] = res_data
# f = open('Data/' + 'result' + '.json', 'w')
# json.dump(obj, f)
# f.close()



f = open('Data/' + 'result' + '.json', 'r')
data = json.load(f)
f.close()
values = []
labels = []
for i in range(len(data['data'])):
    values.append(data['data'][i]['AI_predict'])
    labels.append(data['data'][i]['Real_price_difference_rate'])

plt.plot(values)
plt.plot(labels[:-1])
plt.plot(np.zeros(len(labels)))
plt.show()