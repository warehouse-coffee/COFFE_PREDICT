from typing import Union
from fastapi import FastAPI
from API.crawl import CrawClass, London_US_CRAWL
import json
import datetime
import time
import os
import traceback
from Model.Utils import *
from Model.Network_lstm import Network_running, Network_training
import subprocess
import psutil


process = None  # Biến toàn cục để lưu tiến trình
script_path = "console.py"

app = FastAPI()
f = open('API/links.json', 'r')
Links = json.load(f)
f.close()

trainObj = {}
DataTrain = np.array([])
DataToday = np.array([])
length_min = 0
name = None
label = np.array([])
windowTime = 14
train_now_date = 0
train_now_unix = 0


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

    print(DataTrain.shape)


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


def Training():
    global DataTrain
    global label
    global train_now_date
    global train_now_unix
    global DataToday
    nwtwork = Network_training(DataTrain, label, 200, 0.01, log=False, now_date=train_now_date, now_unix=train_now_unix)
    res = nwtwork.run(today_data=DataToday, isSave=True, filename='model')
    return res


@app.get("/links")
def all_Links():
    return Links


@app.get("/crawl_one")
def crawl_one(product_name: Union[str, None] = None):
    if not Links.get(product_name):
        return "Don't have data"
    data = {}
    try:
        f = open('API/Data_api/' + product_name + '.json', 'r')
        data = json.load(f)
        f.close()
        now = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d')
        if (now != data["date_now"]):
            os.remove('API/Data_api/' + product_name + '.json')
            raise Exception("Fetch new Data")
    except:
        crawl = CrawClass(url=Links.get(product_name)[0])
        data["unix_time_now"] = int(time.time())
        data["date_now"] = datetime.datetime.fromtimestamp(data["unix_time_now"]).strftime('%Y-%m-%d')
        data["data"] = crawl.Crawl()
        f = open('API/Data_api/' + product_name + '.json', 'w')
        json.dump(data, f)
        f.close()
    return data


@app.get("/crawl_all")
def crawl_all():
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
    return {"Time_taken:": current}


@app.get("/get_names")
def get_names():
    names = list(Links.keys())
    return names


@app.get("/train_status")
def train_status():
    f = open('Data/' + 'result' + '.json', 'r')
    data = json.load(f)
    f.close()

    accuracy = data['accuracy']
    date = data['date_now']
    return {
        "accuracy": accuracy,
        "date": date
    }


@app.get("/predict_tommorow")
def predict_tommorow():
    f = open('Data/' + 'result' + '.json', 'r')
    data = json.load(f)
    f.close()

    latest = data['data'][-1]
    return latest


@app.get("/training")
def do_training():
    global date_obj_coffee
    global unix_obj_coffee
    time_taken = time.time()
    Load_Data()
    SetLabel()
    training_data = Training()
    accu = int(training_data[1])
    pred = training_data[0]
    print("start training")
    while accu < 70:
        training_data = Training()
        accu = int(training_data[1])
        pred = training_data[0]

    time_taken = time.time() - time_taken

    # SAVE THE RESULT
    obj = {
        "date_now": datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d'),
        "unix_time_now": time.time(),
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

    return {"Accuracy": accu, "message": "Done Training", "Time_taken": time_taken, "date": train_now_date, "unix": train_now_unix}


@app.get("/predict_graph")
def predict_graph():
    f = open('Data/' + 'result' + '.json', 'r')
    data = json.load(f)
    f.close()

    all_data = data['data']
    return all_data


@app.get("/London_US")
def London_US():
    now = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d')
    current = time.time()
    Data = {
        "unix_time_now": int(current),
        "date_now": now
    }
    try:
        f = open('API/Data_api/London_US.json', 'r')
        data = json.load(f)
        f.close()
        if (now != data["date_now"]):
            os.remove('API/Data_api/London_US.json')
            raise Exception("Fetch new Data")
        else:
            # SET TRAIN DATA
            Data = data['data']
            return Data
    except:
        crawl = London_US_CRAWL()
        Data["data"] = crawl.Crawl()
        f = open('API/Data_api/London_US.json', 'w')
        json.dump(Data, f)
        f.close()
        return Data


@app.get("/run_service")
def run_script():
    global process
    if process is None:
        process = subprocess.Popen(['py', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"message": "Script started", "pid": process.pid}


@app.get('/restart_service')
def restart_script():
    global process
    process = subprocess.Popen(['py', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"message": "Script started", "pid": process.pid}


@app.get('/stop_service')
def stop_script():
    global process
    if process is None:
        return {"message": "No script running", "code": 404}

    proc_id = process.pid
    # Kiểm tra xem tiến trình có đang chạy
    if psutil.pid_exists(proc_id):
        process.terminate()  # Dừng tiến trình
        process.wait()  # Chờ tiến trình hoàn tất
        process = None
        return {"message": "Script stopped -> Process is None", "pid": proc_id}
    else:
        process = None
        return {"message": "Process not running -> Process is None", "pid": proc_id}


@app.get('/status_service')
def check_status():
    global process
    if process is None:
        return {"message": "No script running", "code": 404}

    # Kiểm tra xem tiến trình có đang chạy không
    if psutil.pid_exists(process.pid):
        return {"status": "running", "pid": process.pid}
    else:
        return {"status": "finished", "pid": process.pid}
