from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import json
import datetime
import time
import os
import traceback
import schedule
import numpy as np
# from ..Network_lstm import Network_training, Network_running
from API.crawl import CrawClass, London_US_CRAWL
from Model.Utils import *
from Model.Network_lstm import Network_training, Network_running

f = open('API/links.json', 'r')
Links = json.load(f)
f.close()
# print(len(Links))

options = webdriver.EdgeOptions()
options.add_argument('headless')
# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36')
options.binary_location = 'c:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
service = Service(executable_path='API/web_driver/msedgedriver.exe')
url = 'https://tradingeconomics.com/commodities'

trainObj = {}
DataTrain = np.array([])
DataToday = np.array([])
length_min = 0
name = None
label = np.array([])
train_now_date = 0
train_now_unix = 0
windowTime = 14
date_obj_coffee = np.array([])
unix_obj_coffee = np.array([])

def Load_Data():
    global DataTrain
    global length_min
    global name
    global trainObj
    global DataToday
    global train_now_date
    global train_now_unix

    links_name = list(Links.keys())
    train_now_date = 0
    train_now_unix = 0
    trainObj = {}
    DataTrain = np.array([])
    DataToday = np.array([])
    length_min = 0
    name = None

    f = open('Data/' + 'train' + '.json', 'r')
    data = json.load(f)
    f.close()
    train_now_date = data['date_now']
    train_now_unix = data['unix_time_now']

    for key in links_name:
        trainObj[key] = [i['value'] for i in data[key]]
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
        if len(DataTrain) == 0:
            index = len(trainObj[key]) - length_min
            scaler = MinMax(trainObj[key][index:])
            scaler = numpy_ewma(scaler, windowTime)
            DataToday = np.array(scaler[-1])
            scaler = scaler[:-1]
            scaler = scaler.reshape((-1, 1))
            DataTrain = np.array(scaler)
        else:
            index = len(trainObj[key]) - length_min
            scaler = MinMax(trainObj[key][index:])
            scaler = numpy_ewma(scaler, windowTime)
            DataToday = np.append(DataToday, scaler[-1])
            scaler = scaler[:-1]
            scaler = scaler.reshape((-1, 1))
            DataTrain = np.concatenate((DataTrain, scaler), axis=1)


def SetLabel():
    global label
    global trainObj
    global date_obj_coffee
    global unix_obj_coffee
    label = np.array([])

    index = len(trainObj['Coffee']) - length_min
    scaler = trainObj['Coffee'][index:]
    date_obj_coffee = trainObj['Coffee_date'][index:]
    unix_obj_coffee = trainObj['Coffee_unix_ms'][index:]
    label_1 = np.array(scaler[1:])
    label_2 = np.array(scaler[:-1])
    label = label_1 - label_2
    label = numpy_ewma(label, windowTime)


def Datapoint(index, value, time_s):
    return {
        "index": index,
        "date": datetime.datetime.fromtimestamp(time_s).strftime('%Y-%m-%d'),
        "unix_date_ms": time_s * 1000,
        "value": value
    }


def update():
    count = 0
    current = int(time.time())
    now = datetime.datetime.fromtimestamp(current).strftime('%Y-%m-%d')
    trainData = {
        "unix_time_now": current,
        "date_now": now
    }

    try:
        f = open('Data/' + 'train' + '.json', 'r')
        data = json.load(f)
        f.close()
        print('current date:', now, '; train date:', data["date_now"])
        if (now != data["date_now"]):
            raise Exception("Update the file")
        else:
            print('updated amount - none: ', count)
            return []
    except:
        name_list = []
        browser = webdriver.Edge(options=options, service=service)
        browser.get(url)
        rows = browser.find_elements(By.CSS_SELECTOR, 'tr[data-symbol]')
        print(len(rows))
        for row in rows:
            try:
                td = row.find_elements(By.CSS_SELECTOR, 'td')
                value = float(td[1].get_attribute('innerText'))
                name_split = td[0].find_element(By.CSS_SELECTOR, 'b').get_attribute('innerHTML').strip().split(' ')
                name = name_split[0]
                if (len(name_split) > 1):
                    name = name_split[0] + name_split[1].upper()[0] + name_split[1][1:]
                # READ AND SET VALUE
                f = open('API/Data_api/' + name + '.json', 'r')
                data = json.load(f)
                f.close()
                latest_obj = data['data'][len(data['data']) - 1]
                data['data'].append(Datapoint(latest_obj['index'] + 1, value, current))
                data["unix_time_now"] = current
                data["date_now"] = now
                # SET TRAIN DATA
                trainData[name] = data['data']
                # WRITE
                f = open('API/Data_api/' + name + '.json', 'w')
                json.dump(data, f)
                f.close()
                name_list.append(name)
                count += 1

            except:
                # traceback.print_exc()
                continue
        print('updated amount: ', count)
        f = open('Data/' + 'train' + '.json', 'w')
        json.dump(trainData, f)
        f.close()
        return name_list


def init():
    now = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d')
    current = time.time()
    trainData = {
        "unix_time_now": int(current),
        "date_now": now
    }
    for key in Links:
        # print(key)
        try:
            f = open('API/Data_api/' + key + '.json', 'r')
            data = json.load(f)
            f.close()
            if (now != data["date_now"]):
                os.remove('API/Data_api/' + key + '.json')
            else:
                # SET TRAIN DATA
                trainData[key] = data['data']
                continue
        except:
            pass

        try:
            crawl = CrawClass(url=Links.get(key)[0])
            data = {}
            data["unix_time_now"] = int(time.time())
            data["date_now"] = datetime.datetime.fromtimestamp(data["unix_time_now"]).strftime('%Y-%m-%d')
            data["data"] = crawl.Crawl()
            # SET TRAIN DATA
            trainData[key] = data['data']
            f = open('API/Data_api/' + key + '.json', 'w')
            json.dump(data, f)
            f.close()
        except:
            traceback.print_exc()
            continue
    current = time.time() - current
    f = open('Data/' + 'train' + '.json', 'w')
    json.dump(trainData, f)
    f.close()
    return "Time_taken:" + str(current)


def Running():
    nwtwork = Network_running()
    nwtwork.load_model('Model/models/model_LSTM.npz')
    print(nwtwork.get_status())
    res = nwtwork.predict(DataToday)
    return res


def Training():
    global DataToday
    nwtwork = Network_training(DataTrain, label, 200, 0.01, log=True, now_date=train_now_date, now_unix=train_now_unix)
    res = nwtwork.run(today_data=DataToday, isSave=True, filename='model')
    return res


def Training_FUll():
    global date_obj_coffee
    global unix_obj_coffee

    Load_Data()
    SetLabel()
    training_data = Training()
    accu = int(training_data[1])
    pred = training_data[0]
    while accu < 70:
        training_data = Training()
        accu = int(training_data[1])
        pred = training_data[0]

    obj = {
        "date_now": time.time(),
        "unix_time_now": datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d'),
        "accuracy": accu
    }
    res_data = []
    for i in range(len(pred)):
        if i == len(pred) - 1:
            message = "Predict value for " + datetime.datetime.fromtimestamp(int(unix_obj_coffee[i] / 1000 + 24 * 60 * 60)).strftime('%Y-%m-%d')
            res_data.append({
                "index": i,
                "AI_predict": pred[i],
                "Real_price_difference_rate": 0,
                "Date": date_obj_coffee[i],
                "unix_date_ms": unix_obj_coffee[i],
                "message": message
            })
        else:
            res_data.append({
                "index": i,
                "AI_predict": pred[i],
                "Real_price_difference_rate": label[i],
                "Date": date_obj_coffee[i],
                "unix_date_ms": unix_obj_coffee[i],
                "message": "normal value"
            })

    obj['data'] = res_data
    f = open('Data/' + 'result' + '.json', 'w')
    json.dump(obj, f)
    f.close()


def Main_func():
    print('Start Updating')
    update()
    print('Start Training')
    Training_FUll()
    print('Finish Training')


print(init())
# schedule.every().day.at("06:00").do(update)
# schedule.every().day.at("06:20").do(Training_FUll)
schedule.every().day.at("06:00").do(Main_func)

while True:
    schedule.run_pending()
    time.sleep(1)
