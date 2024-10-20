from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import datetime
import time


class CrawClass:
    def __init__(self, url) -> None:
        self.options = webdriver.EdgeOptions()
        self.options.add_argument('headless')
        # headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        self.options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36')
        self.options.binary_location = 'c:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
        self.service = Service(executable_path='API/web_driver/msedgedriver.exe')
        self.url = url

        # VARIABLES
        self.labels_y_html = None
        self.labels_y_max: float = 0
        self.labels_y_min: float = 0
        self.pathList = None
        self.path_list_length: int = 0
        self.svg = None
        self.svg_w: int = 0
        self.svg_h: int = 0
        self.dataPoints = []

    def Datapoint(self, index, x, y):
        return {
            "index": index,
            "date": self.converDate(index, self.path_list_length)[0],
            "unix_date_ms": self.converDate(index, self.path_list_length)[1] * 1000,
            "value": self.convertValue(y, self.svg_h, self.labels_y_max, self.labels_y_min)
        }

    def converDate(self,index, max_len):
        date_unix = int(time.time())
        date_unix = date_unix - (date_unix % 86400)
        scale = (max_len - index) * 363 / max_len
        date_unix = date_unix - int(scale) * 86400
        date = datetime.datetime.fromtimestamp(date_unix).strftime('%Y-%m-%d')
        return [date, date_unix]

    def convertValue(self, value, value_max, lb_max, lb_min):
        number = (1 - value / (value_max - 80)) * (lb_max - lb_min) + lb_min
        return number

    def Crawl(self):
        browser = webdriver.Edge(options=self.options, service=self.service)
        browser.get(self.url)
        wait = WebDriverWait(browser, 10)
        path = 'svg.highcharts-root .highcharts-series.highcharts-series-0.highcharts-line-series path'
        element = wait.until(lambda x: x.find_element(By.CSS_SELECTOR, path))

        # VARIABLES
        self.labels_y_html = browser.find_element(By.CSS_SELECTOR, '.highcharts-yaxis-labels').find_elements(By.CSS_SELECTOR, 'text')
        self.labels_y_max = float(self.labels_y_html[len(self.labels_y_html) - 1].get_attribute('innerHTML'))
        self.labels_y_min = float(self.labels_y_html[0].get_attribute('innerHTML'))
        self.pathList = element.get_attribute('d').split('L')
        self.path_list_length = len(self.pathList)
        self.svg = browser.find_element(By.CSS_SELECTOR, 'svg.highcharts-root')
        self.svg_w = int(self.svg.get_attribute('width'))
        self.svg_h = int(self.svg.get_attribute('height'))
        self.dataPoints = []
        for i in range(0, self.path_list_length):
            temp = self.pathList[i].split(' ')
            self.dataPoints.append(self.Datapoint(i, float(temp[1]), float(temp[2])))

        # print(dataPoints)
        # print(browser.title)
        browser.quit()

        return self.dataPoints
