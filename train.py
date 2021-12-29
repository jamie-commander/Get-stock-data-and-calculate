# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 23:18:04 2021

@author: Jamie
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

import Catch_BigData as Stock_Api
import numpy as np
Api = Stock_Api.Catch_Stocks_BigData()
    
class StockDataset(Dataset):
    def __init__(self, stock_num, days=360, mode='train', target_only=False):
        self.mode = mode
        
        # Read data
        stock = Api.Stocks(Number = stock_num)
        datas_name = np.array(['Open', 'High', 'Low', 'UpDown', 'Volume', 'Amount', 
                      'UpDownPC', 'SA', 'DI', 'MA5', 'MA10', 'MA20', 'MA60', 'VOL_MA5',
                      'RSI5_index', 'RSI10_index', 'RSV', 'K', 'D', 'J', 'EMA12', 
                      'EMA26', 'DIF', 'MACD', 'MACD9', 'OSC', 'Close'])
        
        #TODO: choose features
        if not target_only:
            features = stock.get_parm()
        else:
            features = ['Close','VOL_MA5']
            #features = ['Close']
        #.get() 參數說明{data_name：特徵，例如：Open=開盤價、MA5=五日均價，格式：dataname = ['Open','MA5']，預設是['Close']。
        #               days：欲取得資料天數，例如：3 就是取得最新三天的資料，格式：days=3，預設是1。
        #               before：取得往前幾天的資料，例如： 2 就是往前兩天之前開始取資料，格式： before = 2，預設是0。
        #               pieces：將幾天的資料包成一筆，例如： 2 就是會有當天跟昨天的資料，格式： pieces = 2，預設是1。
        #   }
        input_datas = stock.get(data_name=features,days=days,before=1,pieces = 5)
        #要用前一天資料預測後一天Close因此before給1，
        
        input_datas = torch.FloatTensor(input_datas)
        
        #預測明天收盤價'Close'
        output_datas = stock.get(data_name=['Close'],days=days,before=0,pieces = 1)
        output_datas = torch.FloatTensor(output_datas)
        

        data = input_datas#datas[:, :-1]
        target = output_datas#datas[:, -1]
        #print(data)
        
        # Training data (train/Val sets)
        # Splitting training set into train & Val sets
        Val = int(days/20)
        train_data = data[:-1*Val]
        train_target = target[:-1*Val]#torch.unsqueeze(target[:-1*Val], dim=1)
        Val_data = data[-1*Val:]
        Val_target = target[-1*Val:]#torch.unsqueeze(target[-1*Val:], dim=1)
        
        # Splitting training data into train & Val sets
        if mode == 'train':
            self.data = train_data
            self.target = train_target
        elif mode == 'Val':
            self.data = Val_data
            self.target = Val_target

        #TODO: Normalization
        # z-score standardization
        self.data = (self.data - self.data.mean(dim=0, keepdim=True)) / self.data.std(dim=0, keepdim=True)
        
        # min-max
        #data = (data - torch.min(data, dim=0)[0]) / (torch.max(data, dim=0)[0] - torch.min(data, dim=0)[0])
        
        self.dim = self.data.shape[1]

        print('Finished reading the {} set of Stock Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        if self.mode in ['train', 'Val']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index], self.target[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
    def get_data(self):
        if self.mode in ['train', 'Val']:
            # For training
            return self.data, self.target
        else:
            # For testing (no target)
            return self.data, self.target
    
def prep_dataloader(stock_num, days, mode, batch_size, n_jobs=0, target_only=False):
    dataset = StockDataset(stock_num, days, mode=mode, target_only=target_only)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False,
                            num_workers=n_jobs, pin_memory=True)
    return dataloader

def set_seed(myseed=8):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )
        '''nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)'''
        '''
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
        '''
        self.criterion = nn.L1Loss()

    def forward(self, x):
        return self.net(x)
    
    def cal_loss(self, pred, y):
        return self.criterion(pred, y)

def train(train_set, Val_set, model, optimizer, num_epoch, early_stop):
    plt.ion()
    plt.show()
    
    count = 0
    min_loss = 1000.
    train_loss_list = []
    Val_loss_list = []
    epoch = 0
    model.train()
    for epoch in range(num_epoch):
        train_loss = 0
        for data, target in train_set:
            data = data.to(Valice)
            target = target.to(Valice)
            pred = model(data)
    
            loss = model.cal_loss(pred, target)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.cpu().numpy()
        train_loss /= (len(train_set))
        train_loss_list.append(train_loss)
        
        Val_loss = Val(Val_set, model)
        Val_loss_list.append(Val_loss)
        
        if  Val_loss < min_loss:
            min_loss = Val_loss
            count = 0
            
            print('Epoch %d : Loss=%.4f(saving model...)' % (epoch, Val_loss))
            torch.save(model.state_dict(), 'model.pth')
            
        else:
            count+=1
            
        if count > early_stop:
            break
        epoch += 1
    
        if epoch % 40 == 1:
            plt.cla()
            plt.plot(np.arange(1, len(target) + 1), target.data.cpu().numpy(), 'b-')
            plt.plot(np.arange(1, len(pred) + 1), pred.data.cpu().numpy(), 'r-')
            plt.pause(0.05)
       
    return train_loss_list, Val_loss_list

def Val(Val_set, model,show=False):
    model.eval()
    
    record_loss_list = []
    total_loss = 0
    for data, target in Val_set:
        data, target = data.to(Valice), target.to(Valice)
        with torch.no_grad():
            pred = model(data)
            loss = model.cal_loss(pred, target)
            record_loss_list.append(loss)
            total_loss += loss.detach().cpu()
            if(show):   
                plt.plot(np.arange(1, len(target) + 1), target.data.cpu().numpy(), 'b-')
                plt.plot(np.arange(1, len(pred) + 1), pred.data.cpu().numpy(), 'r-')
                plt.pause(0.05)
    total_loss /= len(Val_set)
    #print('total loss: %.2f%%' % total_loss)
          
    return total_loss

days = 120
num_epoch = 5000
early_stop = 500
stock_num = '2337'
batch_size = 120#int(days * 2 / 3)
target_only = True
loss_record = {}

#data, target, Val_data, Val_target = get_data(stock_num, days=days)
train_set = prep_dataloader(stock_num, days, 'train', batch_size, target_only=target_only)
Val_set = prep_dataloader(stock_num, days, 'Val', batch_size, target_only=target_only)
#print(train_set.dataset.dim)
#train_set2 = StockDataset(stock_num, days, mode='train', target_only=target_only).get_data()
#Val_set2 = StockDataset(stock_num, days, mode='Val', target_only=target_only).get_data()
print(len(train_set))

set_seed(0)
Valice = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(input_dim=train_set.dataset.dim).to(Valice)
#model = Net(input_dim=train_set2[0].shape[1]).to(Valice)
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

#train_no_use(data, target, Val_data, Val_target, model, optimizer, num_epoch, early_stop)
loss_record['train'], loss_record['Val'] = train(train_set, Val_set, model, optimizer, num_epoch, early_stop)

del model
model = Net(input_dim=Val_set.dataset.dim).to(Valice)
model.load_state_dict(torch.load('model.pth'))


Val_loss = Val(Val_set, model,show = True)
print('total loss: %.2f%%' % Val_loss)

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & Val loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = range(len(loss_record['Val']))
    plt.figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], 'r-', label='train')
    plt.plot(x_2, loss_record['Val'], 'b-', label='Val')
    plt.ylim(0.0, 10.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    #plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

plot_learning_curve(loss_record)