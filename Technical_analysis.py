import numpy as np
import pandas as pd
import time
class Indicator_Calc:
    def __init__(self,data = []):
        True
    def Initial(self,data):
        if(self.type_check(data) == True):
            data['UpDownPC'] = self.UpDownPC(data)
            data['UpDown'] = self.UpDown(data)
            data['SA'] = self.SA(data)
            data['DI'] = self.DI(data)
        return True
    def SA(self,data):
        if(self.type_check(data) == False):
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        for i in range(0,data_len):
            if(i>=1):
                PC_value = abs(data['High'][i] - data['Low'][i])/data['Close'][i-1]*100
                result = np.append(result,PC_value)
            else:
                result = np.append(result,np.nan)
        return result
    def UpDownPC(self,data,PARM = 'Close'):
        if(self.type_check(data) == False):
            return False
        if((PARM in data.columns) == False):
            print('不存在的參數')
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        for i in range(0,data_len):
            if(i>=1):
                PC_value = (data[PARM][i] - data[PARM][i-1])/data[PARM][i-1]*100
                result = np.append(result,PC_value)
            else:
                result = np.append(result,np.nan)
        return result
    def UpDown(self,data,PARM = 'Close'):
        if(self.type_check(data) == False):
            return False
        if((PARM in data.columns) == False):
            print('不存在的參數')
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        for i in range(0,data_len):
            if(i>=1):
                value = (data[PARM][i] - data[PARM][i-1])
                result = np.append(result,value)
            else:
                result = np.append(result,np.nan)
        return result
    def RSV(self,data,T):
        if(self.type_check(data) == False):
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        for i in range(1,data_len + 1):
            if(i>=T and (pd.isnull(data['Close'][i-T:i]).any() == False)):
                if(data['High'][i-T:i].max() != data['Low'][i-T:i].min()):
                    RSV = (data['Close'][i-1] - data['Low'][i-T:i].min())/(data['High'][i-T:i].max() - data['Low'][i-T:i].min())*100
                else:
                    RSV = np.nan#此種情況幾乎不可能發生
                result = np.append(result,RSV)
            else:
                result = np.append(result,np.nan)
        return result
    def DI(self,data):
        if(self.type_check(data) == False):
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        for i in range(0,data_len):
            if(i>=1):
                DI_value = (data['High'][i] + data['Low'][i] + data['Close'][i]*2)/4
                result = np.append(result,DI_value)
            else:
                result = np.append(result,np.nan)
        return result
    def SMA(self,data,T,PARM = 'Close'):
        #print(data)
        #print(data['Open'][-20:-10])
        if(self.type_check(data) == False):
            return False
        if((PARM in data.columns) == False):
            print('不存在的參數')
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        #result.astype('float32')
        data_len = len(data)
        #print(data_len)
        for i in range(1,data_len + 1):
            #if(i == data_len):
            #    print(i-T)#
            #    print(data[PARM][i-T])#
            #    print(i)#
            #    print(data[PARM][i-1])#
            #    print(data[PARM][i-T:i-1])#[i:j] 取i到j-1
            #if(10>i):
            #    print(data[PARM][i-T:i])
            #    print(pd.isnull(data[PARM][i-T:i]).any())
            #print(data)
            if(i>=T and (pd.isnull(data[PARM][i-T:i]).any() == False)):
                result = np.append(result,data[PARM][i-T:i].mean())
            else:
                result = np.append(result,np.nan)
        return result
    def EMA(self,data,T,PARM = 'DI'):
        if(self.type_check(data) == False):
            return False
        if((PARM in data.columns) == False):
            print('不存在的參數')
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        Flag = False
        EMA = 0
        for i in range(1,data_len + 1):
            if(Flag == False):
                if(i>=T and (pd.isnull(data[PARM][i-T:i]).any() == False)):
                    Flag = True
                    EMA = data[PARM][i-T:i].mean()
                    result = np.append(result,EMA)
                else:
                    result = np.append(result,np.nan)
            else:
                EMA = ((EMA * (T-1)) + (data[PARM][i-1] * (2)))/(T+1)
                result = np.append(result,EMA)
        return result
    def RSI(self,data,T,exp = True):#指數平均RSI、簡單平均RSI
        if(self.type_check(data) == False):
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        '''for i in range(1,data_len + 1):#簡單平均
            if(i>=T and (pd.isnull(data['UpDown'][i-T:i]).any() == False)):
                Up = 0
                Down = 0
                for j in data['UpDown'][i-T:i]:
                    if(j>0):
                        Up = Up + j
                    else:
                        Down = Down + j
                if(Down!=0):
                    Up = Up/T
                    Down = abs(Down/T)
                    RS = Up/Down
                    RSI = (RS/(RS+1))*100
                else:
                    RSI = 100
                result = np.append(result,RSI)
            else:
                result = np.append(result,np.nan)'''
        Up = 0
        Down = 0
        Flag = False
        for i in range(1,data_len + 1):
            if(Flag == False):
                if(i>=T and (pd.isnull(data['UpDown'][i-T:i]).any() == False)):
                    Up = 0
                    Down = 0
                    Flag = exp
                    for j in data['UpDown'][i-T:i]:
                        if(j>0):
                            Up = Up + j
                        else:
                            Down = Down + j
                    if(Down!=0):
                        Up = Up/T
                        Down = abs(Down/T)
                        RS = Up/Down
                        RSI = (RS/(RS+1))*100
                    else:
                        RSI = 100
                    result = np.append(result,RSI)
                else:
                    result = np.append(result,np.nan)
            else:
                if(data['UpDown'][i-1]>0):
                    Up = Up + (abs(data['UpDown'][i-1]) - Up)/T
                    Down = Down + (0 - Down)/T
                else:
                    Up = Up + (0 - Up)/T
                    Down = Down + (abs(data['UpDown'][i-1]) - Down)/T
                RS = Up/Down
                RSI = (RS/(RS+1))*100
                result = np.append(result,RSI)
        return result
    def SO(self,data,PARM = 'RSV'):#隨機指標KD
        if(self.type_check(data) == False):
            return False
        if((PARM in data.columns) == False):
            print('不存在的參數')
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        SO = 0
        Flag = False
        for i in range(0,data_len):
            if(Flag == False):
                if((pd.isnull(data[PARM][i]) == False)):
                    SO = 0
                    Flag = True
                    SO = data[PARM][i]
                    result = np.append(result,SO)
                else:
                    result = np.append(result,np.nan)
            else:
                SO = SO*2/3 + data[PARM][i]*1/3
                result = np.append(result,SO)
        return result
    def J(self,data,PARM_1 = 'K',PARM_2 = 'D'):
        if(self.type_check(data) == False):
            return False
        if(((PARM_1 in data.columns) == False) or ((PARM_2 in data.columns) == False)):
            print('不存在的參數')
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        J = 0
        for i in range(0,data_len):
            if((pd.isnull(data[PARM_1][i]) == False) and (pd.isnull(data[PARM_2][i]) == False)):
                J = 3*data[PARM_2][i] - 2*data[PARM_1][i]
                result = np.append(result,J)
            else:
                result = np.append(result,np.nan)
        return result
    def DIF(self,data,PARM_1 = 'EMA12',PARM_2 = 'EMA26'):
        if(self.type_check(data) == False):
            return False
        if(((PARM_1 in data.columns) == False) or ((PARM_2 in data.columns) == False)):
            print('不存在的參數')
            return False
        result = []
        result = np.array(result, dtype=np.float64)
        data_len = len(data)
        result = data[PARM_1] - data[PARM_2]
        return result
    def MACD(self,data,T,PARM = 'DIF'):
        return self.EMA(data,T,PARM)
    def OSC(self,data,PARM_1 = 'DIF',PARM_2 = 'MACD9'):
        return self.DIF(data,PARM_1,PARM_2)
    def type_check(self,data):
        if(type(data)!=pd.core.frame.DataFrame):
            print('請使用pandas.core.frame.DataFrame型態，並且包含相同長度的''\'Open''\'、''\'High''\'、''\'Low''\'、''\'Close''\'、''\'Volume''\'、''\'Amount''\'。')
            return False
        elif(('Open' in data.columns and 'High' in data.columns and 'Low' in data.columns and 'Close' in data.columns and 'Volume' in data.columns and 'Amount' in data.columns) == False):
            print('請使用pandas.core.frame.DataFrame型態，並且包含相同長度的''\'Open''\'、''\'High''\'、''\'Low''\'、''\'Close''\'、''\'Volume''\'、''\'Amount''\'。')
            return False
        return True
