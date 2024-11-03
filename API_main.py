from typing import Union
from fastapi import FastAPI
from API.crawl import CrawClass, London_US_CRAWL
import json
import datetime
import time
import os
import traceback
from Model.Utils import *
from Model.Network_lstm import Network_running
import subprocess
import psutil


process = None  # Biến toàn cục để lưu tiến trình
script_path = "console.py"

app = FastAPI()
f = open('API/links.json', 'r')
Links = json.load(f)
f.close()


@app.get("/link")
def all_Links():
    return Links


@app.get("/crawl")
def crawl(q: Union[str, None] = None):
    if not Links.get(q):
        return "Don't have data"
    data = {}
    try:
        f = open('API/Data_api/' + q + '.json', 'r')
        data = json.load(f)
        f.close()
        now = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d')
        if (now != data["date_now"]):
            os.remove('API/Data_api/' + q + '.json')
            raise Exception("Fetch new Data")
    except:
        crawl = CrawClass(url=Links.get(q)[0])
        data["unix_time_now"] = int(time.time())
        data["date_now"] = datetime.datetime.fromtimestamp(data["unix_time_now"]).strftime('%Y-%m-%d')
        data["data"] = crawl.Crawl()
        f = open('API/Data_api/' + q + '.json', 'w')
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
    try:
        nwtwork = Network_running()
        nwtwork.load_model('Model/models/model_LSTM.npz')
        return nwtwork.get_status()
    except:
        return {
            "error": "cook"
        }


@app.get("/predict")
def predict():
    links_name = list(Links.keys())
    trainObj = {}
    DataToday = np.array([])
    f = open('Data/' + 'train' + '.json', 'r')
    data = json.load(f)
    f.close()
    length_min = 0

    for key in links_name:
        trainObj[key] = [i['value'] for i in data[key]]
        if length_min == 0:
            length_min = len(trainObj[key])
        elif len(trainObj[key]) < 100:
            links_name.remove(key)
        elif len(trainObj[key]) < length_min:
            length_min = len(trainObj[key])

    for key in links_name:
        if DataToday.size == 0:
            index = len(trainObj[key]) - length_min
            scaler = MinMax(trainObj[key][index:])
            scaler = numpy_ewma(scaler, 7)
            DataToday = np.array(scaler[-1])
        else:
            index = len(trainObj[key]) - length_min
            scaler = MinMax(trainObj[key][index:])
            scaler = numpy_ewma(scaler, 7)
            DataToday = np.append(DataToday, scaler[-1])

    nwtwork = Network_running()
    nwtwork.load_model('Model/models/model_LSTM.npz')
    res = nwtwork.predict(DataToday)
    predict_now_unix = nwtwork.now_unix + 24 * 60 * 60
    predict_now_date = datetime.datetime.fromtimestamp(predict_now_unix).strftime('%Y-%m-%d')
    return {
        "index": index,
        "date": predict_now_date,
        "unix_date_ms": predict_now_unix * 1000,
        "value": res[0][0],
        "accuracy": nwtwork.accuracy
    }


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
