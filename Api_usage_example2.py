import Catch_BigData as Stock_Api
import numpy as np
import pandas as pd
import time
Api = Stock_Api.Catch_Stocks_BigData()

def catch_data():
    global Api
    start = time.time()
    Updata_status = Api.Initial_Opening()#建立資料，首次運行可能需要6小時以上，每天收盤後運行一次大約4分鐘。
    if(Updata_status):
        print('更新成功')
    else:
        print('更新失敗')
    end = time.time()
    print(end-start)

def get_stock(stock_num='2330'):
    global Api
    return Api.Stocks(Number = stock_num)#Number 輸入股票代號
    #data.get(data_name=data_name,T=days)#data_name欲取得的參數名稱，T=天數
    #data.get_parm() #印出所有參數名稱
    '''
    print(d)
    print(np.array(d))
    print(data.get_datalen())
    '''
if __name__ == "__main__":
    catch_data()
    '''data_2337 = get_stock(stock_num='2337')
    data_2337.get()
    parm_2337 = data_2337.get_parm()
    for i in parm_2337:
        print(i)'''
        
'''
Open 
High  
Low   
Close 
UpDown    漲跌 $
Volume    成交量 股
Amount    成交金額 $
UpDownPC  漲跌幅度 %
SA        
DI
MA5
MA10
MA20
MA60
VOL_MA5
RSI5_index
RSI10_index
RSV
K
D
J
EMA12
EMA26
DIF
MACD
MACD9
OSC
'''


