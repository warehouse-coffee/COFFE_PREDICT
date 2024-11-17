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
            "real_price": self.convertValue(y, self.svg_h, self.labels_y_max, self.labels_y_min)
        }

    def converDate(self, index, max_len):
        date_unix = int(time.time())
        date_unix = date_unix - (date_unix % 86400)
        scale = (max_len - index - 1) * 363 / max_len
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


class London_US_CRAWL:
    def __init__(self) -> None:
        self.options = webdriver.EdgeOptions()
        self.options.add_argument('headless')
        # headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        self.options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36')
        self.options.binary_location = 'c:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
        self.service = Service(executable_path='API/web_driver/msedgedriver.exe')
        self.url = 'https://giacaphe.com/gia-ca-phe-noi-dia/'

    def DataPoint_ton(self, Tenure, Match_price, Change, Highest, Lowest, Mass, Open, The_day_before, Open_Contract):
        return {
            "Tenure": Tenure,
            "Match_price_USD_TON": Match_price,
            "Match_price_USD_LB": Match_price / 22.06,

            "Change_USD_TON": Change,
            "Change_USD_LB": Change / 22.06,

            "Highest_USD_TON": Highest,
            "Highest_USD_LB": Highest / 22.06,

            "Lowest_USD_TON": Lowest,
            "Lowest_USD_LB": Lowest / 22.06,

            "Mass_USD_TON": Mass,
            "Mass_USD_LB": Mass / 22.06,

            "Open_USD_TON": Open,
            "Open_USD_LB": Open / 22.06,

            "Day_before_USD_TON": The_day_before,
            "Day_before_USD_LB": The_day_before / 22.06,

            "Open_Contract_USD_TON": Open_Contract,
            "Open_Contract_USD_LB": Open_Contract / 22.06
        }

    def DataPoint_lb(self, Tenure, Match_price, Change, Highest, Lowest, Mass, Open, The_day_before, Open_Contract):
        return {
            "Tenure": Tenure,
            "Match_price_USD_TON": Match_price * 22.06,
            "Match_price_USD_LB": Match_price,

            "Change_USD_TON": Change * 22.06,
            "Change_USD_LB": Change,

            "Highest_USD_TON": Highest * 22.06,
            "Highest_USD_LB": Highest,

            "Lowest_USD_TON": Lowest * 22.06,
            "Lowest_USD_LB": Lowest,

            "Mass_USD_TON": Mass * 22.06,
            "Mass_USD_LB": Mass,

            "Open_USD_TON": Open * 22.06,
            "Open_USD_LB": Open,

            "Day_before_USD_TON": The_day_before * 22.06,
            "Day_before_USD_LB": The_day_before,

            "Open_Contract_USD_TON": Open_Contract * 22.06,
            "Open_Contract_USD_LB": Open_Contract
        }

    def Crawl(self):
        browser = webdriver.Edge(options=self.options, service=self.service)
        browser.get(self.url)
        wait = WebDriverWait(browser, 10)
        path = '#robusta-london > div.live-table > table > tbody'
        element = wait.until(lambda x: x.find_element(By.CSS_SELECTOR, path))
        dataPoints = {
            "London": [],
            "New York": []
        }

        table_1 = browser.find_element(By.CSS_SELECTOR, '#robusta-london > div.live-table > table > tbody')
        table_2 = browser.find_element(By.CSS_SELECTOR, '#arabica-newyork > div.live-table > table > tbody')

        table_1_rows = table_1.find_elements(By.CSS_SELECTOR, 'tr')
        table_2_rows = table_2.find_elements(By.CSS_SELECTOR, 'tr')

        for i in range(0, len(table_1_rows)):
            table_1_td = table_1_rows[i].find_elements(By.CSS_SELECTOR, 'td')
            Tenure = table_1_td[0].get_attribute('innerHTML')
            Match_price = int(table_1_td[1].get_attribute('innerHTML').replace(',', ''))
            Change = int(table_1_td[2].find_element(By.CSS_SELECTOR, 'span').get_attribute('innerHTML').replace(',', ''))
            Highest = int(table_1_td[3].find_element(By.CSS_SELECTOR, 'span').get_attribute('innerHTML').replace(',', ''))
            Lowest = int(table_1_td[4].find_element(By.CSS_SELECTOR, 'span').get_attribute('innerHTML').replace(',', ''))
            Mass = int(table_1_td[5].get_attribute('innerHTML').replace(',', ''))
            Open = int(table_1_td[6].get_attribute('innerHTML').replace(',', ''))
            Day_before = int(table_1_td[7].get_attribute('innerHTML').replace(',', ''))
            Open_Contract = int(table_1_td[8].get_attribute('innerHTML').replace(',', ''))
            dataPoint = self.DataPoint_ton(Tenure, Match_price, Change, Highest, Lowest, Mass, Open, Day_before, Open_Contract)
            dataPoints["London"].append(dataPoint)

        for i in range(0, len(table_2_rows)):
            table_2_td = table_2_rows[i].find_elements(By.CSS_SELECTOR, 'td')
            Tenure = table_2_td[0].get_attribute('innerHTML')
            Match_price = float(table_2_td[1].get_attribute('innerHTML'))
            Change = float(table_2_td[2].find_element(By.CSS_SELECTOR, 'span').get_attribute('innerHTML'))
            Highest = float(table_2_td[3].find_element(By.CSS_SELECTOR, 'span').get_attribute('innerHTML'))
            Lowest = float(table_2_td[4].find_element(By.CSS_SELECTOR, 'span').get_attribute('innerHTML'))
            Mass = int(table_2_td[5].get_attribute('innerHTML').replace(',', ''))
            Open = float(table_2_td[6].get_attribute('innerHTML'))
            Day_before = float(table_2_td[7].get_attribute('innerHTML'))
            Open_Contract = int(table_2_td[8].get_attribute('innerHTML').replace(',', ''))
            dataPoint = self.DataPoint_lb(Tenure, Match_price, Change, Highest, Lowest, Mass, Open, Day_before, Open_Contract)
            dataPoints["New York"].append(dataPoint)

        browser.quit()
        return dataPoints
