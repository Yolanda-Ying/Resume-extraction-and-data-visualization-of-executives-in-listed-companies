import requests
from bs4 import BeautifulSoup
import traceback
import re
import pandas as pd


def getHTMLText(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""


def getStockList(lst,stockName, stockURL):
    html = getHTMLText(stockURL)
    soup = BeautifulSoup(html, 'html.parser')
    a = soup.find_all('a')
    for i in a:
        try:
            href = i.attrs['href']
            name = i.get_text()

            stockName.append(re.findall(r'.*\)',name)[0])
            lst.append(re.findall(r'[S][HZ]\d{6}', href)[0])
        except:
            traceback.print_exc()
            continue


def getStockInfo(lst, stockName, stockURL, fpath):
    managerInfo = pd.DataFrame()
    for i in range(len(lst)):
        stock = lst[i]
        stockname = stockName[i]
        url = stockURL + stock + "/gaoguanjieshao/"
        html = getHTMLText(url)
        try:
            if html == "":
                continue
            soup = BeautifulSoup(html, 'html.parser')
            stockInfo = soup.find_all('td', attrs={'class': 'hiddenTD'})
            for info in stockInfo:
                managerinfo_onepiece = {"ManagerResume":info.get_text(),"StockNumber":stockname}
                print(managerinfo_onepiece)
                managerInfo = managerInfo.append(managerinfo_onepiece, ignore_index=True)
                print(managerInfo)
        except:
            traceback.print_exc()
            continue
    pd.DataFrame(managerInfo).to_csv(fpath)

def main():
    stock_list_url = 'https://hq.gucheng.com/gpdmylb.html'
    stock_info_url = 'http://hq.gucheng.com/'
    output_file = '/Users/yingyue/Desktop/ResumeEntityIdentification/data/RawData/stockinfo.csv'

    slist = []
    sname = []
    getStockList(slist,sname, stock_list_url)
    print(slist)
    print(sname)
    getStockInfo(slist,sname, stock_info_url, output_file)

if __name__ == '__main__':
    main()