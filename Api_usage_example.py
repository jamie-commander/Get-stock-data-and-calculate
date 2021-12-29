import Catch_BigData as Stock_Api
import Technical_analysis as Tech_Anal
# basic
import numpy as np
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from cycler import cycler
Calc = Tech_Anal.Indicator_Calc()
Api = Stock_Api.Catch_Stocks_BigData()
#Api = Stock_Api.Stocks()
'''
start = time.time()
Updata_status = Api.Initial_Opening()#建立資料，首次運行可能需要6小時以上，每天收盤後運行一次大約4分鐘。
if(Updata_status):
    print('更新成功')
else:
    print('更新失敗')
end = time.time()
print(end-start)
'''
#s_data = Api.Stocks(Number = '2337')#輸入想取得的個股代號，沒有ETF。
#s_data = Api.Stocks(Number = "2337")
df = Api.Stocks(Number = "2337")
#gg = Api.ff()
#gg.fa()
df = df.data
print(df[-10:])
#print(s_data)
'''
#print(s_2330['歷史成交資訊']['Open'])
df = pd.DataFrame(s_data['歷史成交資訊'])
df = df

#data = pd.read_csv('d:/python_work/202010/test2020.csv', index_col='Date')
#data.index = pd.DatetimeIndex(data.index)
df.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close','UpDown':'UpDown','Volume':'Volume','Amount':'Amount','Date':'Date'},inplace=True)
df.Date = pd.to_datetime(df.Date)
df.set_index('Date',inplace=True)
Calc.Initial(df)
MA_TEST = Calc.SMA(df,5)#[-30:]
df['MA5'] = MA_TEST
df['MA10'] = Calc.SMA(df,10)
df['MA20'] = Calc.SMA(df,20)
df['MA60'] = Calc.SMA(df,60)
df['VOL_MA5'] = Calc.SMA(df,5,PARM = 'Volume')
#df['MA4_MA5'] = Calc.SMA(df,4,PARM = 'MA5')
#df['SA_MA5'] = Calc.SMA(df,5,PARM = 'SA')
#df['RSI5_avg'] = Calc.RSI(df,5,exp = False)
df['RSI5_index'] = Calc.RSI(df,5)
df['RSI10_index'] = Calc.RSI(df,10)
#df['RSI6'] = Calc.RSI(df,6,index = False)
df['RSV'] = Calc.RSV(df,9)
df['K'] = Calc.SO(df,PARM = 'RSV')
df['D'] = Calc.SO(df,PARM = 'K')
df['J'] = Calc.J(df,PARM_1 = 'D',PARM_2 = 'K')
#df['MA5_RSI6'] = Calc.MA(df,5,PARM = 'RSI6')
df['EMA12'] = Calc.EMA(df,12,PARM = 'DI')
df['EMA26'] = Calc.EMA(df,26,PARM = 'DI')
df['DIF'] = Calc.DIF(df,PARM_1 = 'EMA12',PARM_2 = 'EMA26')
df['MACD'] = Calc.MACD(df,9,PARM = 'DIF')
df['MACD9'] = Calc.EMA(df,9,PARM = 'DIF')
df['OSC'] = Calc.OSC(df,PARM_1 = 'DIF',PARM_2 = 'MACD9')'''

'''
print('oooooooooooo')
print(MA_TEST)
print("---")
print(MA_TEST.ndim)
print("---") # 分隔線
print(MA_TEST.shape)
print("---") # 分隔線
print(MA_TEST.dtype)
print("---")
print('oooooooooooo')
'''
'''
pd.set_option("display.max_rows",1000000000)#設定顯示行數
pd.set_option("display.max_columns",1000000000)#設定顯示列數
print(df[:30])
print(df[-30:])
print("---")
print(df.ndim)
print("---") # 分隔線
print(df.shape)
print("---") # 分隔線
print(df.dtypes)
print("---")
#print(df['Open'])
'''
#--------------畫圖test----------------
kwargs = dict(
	type='candle', #图表类型，可选值包含：’ohlc’, ‘candle’, ‘line’, ‘renko’, ‘pnf’
	mav=(),#均線
        xrotation =0,#x軸刻度旋轉
	volume=True, 
	title = "2337",#s_data['證券基本資料']['證券代號'],#標題
        ylabel ='Price',#縱軸標籤
	ylabel_lower='Volume',#成交量y軸標籤
	figratio=(15, 10), #控制圖表大小
	figscale=2, #设置图像的缩小或放大,1.5就是放大50%，最大不会超过电脑屏幕大小
        #tight_layout=True,緊密布局
        #fill_between=df['Close'].values,#填色
        #fill_between=dict(y1=df['Close'].values,alpha=0.5,color='g'),#填色
        #fill_between=dict(y1=df['Close'].values,y2=df['Open'].values,alpha=0.5,color='g'),#填色
        #fill_between=dict(y1=100,y2=500,alpha=0.5,color='g'),#填色
        )

mc = mpf.make_marketcolors(
	up='red', 
	down='green', 
	edge='i', 
	wick='i', 
	volume='in', 
	inherit=True)

ss = mpf.make_mpf_style(
	gridaxis='both', 
	gridstyle='-.', 
	y_on_right=False,#False左邊True右邊 
	marketcolors=mc)


mpf.available_styles()
['binance',
 'blueskies',
 'brasil',
 'charles',
 'checkers',
 'classic',
 'default',
 'mike',
 'nightclouds',
 'sas',
 'starsandstripes',
 'yahoo']

#style k線圖樣式 mplfinance 提供須多內置樣式


t_list = [np.nan,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,np.nan,-1,-1,-1,-1,np.nan,-1,-1,-1,np.nan,-1,-1,-1,-1,np.nan]
add_plot = [
    #mpf.make_addplot(df[['High','Low']][-30:], linestyle='dashdot'),#, linestyle='dashdot'虛線
    #mpf.make_addplot(t_list[-30:], scatter=True, markersize=200, marker='^', color='y'),
    #mpf.make_addplot(df['UpDown'][-30:], panel=2, color='g', secondary_y='auto'),.
    mpf.make_addplot(df[['VOL_MA5']][-30:], panel=1, linestyle='solid',color = 'y', secondary_y=False),
    mpf.make_addplot(df[['MA5','MA10','MA20','MA60']][-30:], linestyle='solid', secondary_y='auto'),#secondary_y=True #solid、dashed、dash-dot、dotted
    mpf.make_addplot(df[['RSI5_index']][-30:], panel=4, linestyle='solid',color = 'r',ylabel = 'RSI', secondary_y=False),
    mpf.make_addplot(df[['RSI10_index']][-30:], panel=4, linestyle='solid',color = 'b', secondary_y=False),
    mpf.make_addplot(df[['J']][-30:], panel=3,type='bar',width=0.7,color = 'y',ylabel = 'KDJ', secondary_y=False),
    mpf.make_addplot(df[['K']][-30:], panel=3, linestyle='solid',color = 'r', secondary_y=False),
    mpf.make_addplot(df[['D']][-30:], panel=3, linestyle='solid',color = 'b', secondary_y=False),
    mpf.make_addplot(df['OSC'][-30:],type='bar', panel=2,width=0.7,color = 'y',ylabel = 'MACD', secondary_y=False),
    mpf.make_addplot(df[['DIF']][-30:], panel=2,color = 'r', linestyle='solid', secondary_y=False),
    mpf.make_addplot(df[['MACD9']][-30:], panel=2,color = 'b', linestyle='solid', secondary_y=False),
    ]
plt.rcParams['axes.prop_cycle'] = cycler(color=['dodgerblue', 'deeppink', 'navy', 'teal', 'maroon', 'darkorange', 'indigo'])#顏色
plt.rcParams['lines.linewidth'] = .5 #線寬
plt.rcParams['xtick.labelsize'] = 5
mpf.plot(df[-30:],**kwargs,style = ss,show_nontrading=False,savefig='test_plot.png',block = False, addplot=add_plot)#, addplot=add_plot
mpf.show()

'''
x = np.linspace(0.0, 2*np.pi)  # 50x1 array between 0 and 2*pi
y = np.cos(x)                  # cos(x)
plt.plot(x, y, 'r')   # red line without marker
plt.show()
'''
