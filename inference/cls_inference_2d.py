import numpy as np
import pandas as pd
import sys
import gc
import os
from os.path import isfile
from copy import deepcopy

from torch import nn, optim
from torch.nn import functional as F
from torch.nn import LSTM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import *
from tqdm import tqdm

Model_Path = 'models'

# Data Preprocess
with open('./data/task2.txt', 'r') as f:
    words = f.readlines()
    f.close()

d2_data = []
for word in words:
    idx = int(float(word.split(';')[0]))
    if idx == 2: d2_data.append(','.join(word.split(';')[1:])[:-2])

with open('./data/task2-2d.csv', 'w') as f:
    f.write('pos;length\n')
    for word in d2_data:
        f.write(word+';')
        length = int(len(word.split(','))/2)
        f.write(str(length)+'\n')
    f.close()

del words, d2_data
gc.collect()

data = pd.read_csv('./data/task2-2d.csv', sep=';')

def SepX(df):
    return ','.join(df['pos'].split(',')[:df['length']])

def SepY(df):
    return ','.join(df['pos'].split(',')[df['length']:])

data['pos_x'] = data.apply(SepX, axis=1)
data['pos_y'] = data.apply(SepY, axis=1)

def normalize(x):
    data = np.array([float(i) for i in x.split(',')])
    mean = np.mean(data)
    std = np.std(data)
    data2 = (data - mean)/std
    return ','.join([str(i) for i in data2])

data['pos_x'] = data['pos_x'].apply(lambda x: normalize(x))
data['pos_y'] = data['pos_y'].apply(lambda x: normalize(x))

# Check Model File
MarkLength = [10,15,20,25,30,40,45,50,55,60,70,80,90,100,
              105,110,115,120,125,150,175,200,225,250,
              275,300,325,350,375,
              400,425,450,475,500,550,600,650,700,750,800,850,900,950]

flag = False
for fold in range(5):
    for mark in MarkLength:
        if not isfile('./{}/Fold{}/{}/bestmodel.pth'.format(Model_Path, fold, mark)):
            print('Model file is missing for length {} at fold {}'.format(mark, fold))
            flag = True

if flag: sys.exit(0)

# PyTorch Dataset
def fixlength(x):
    assert (x>=10)
    if x in MarkLength:
        return x
    MarkLengthTemp = deepcopy(MarkLength)
    MarkLengthTemp.append(x)
    MarkLengthTemp.sort()
    Mark = MarkLengthTemp.index(x)
    return MarkLengthTemp[Mark-1]

data['fix_length'] = data['length'].apply(lambda x: fixlength(x))

class AnDiDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()
        
    def __getitem__(self, index): 
        data_seq_x = torch.Tensor([float(i) for i in self.df['pos_x'].iloc[index].split(',')])
        data_seq_y = torch.Tensor([float(i) for i in self.df['pos_y'].iloc[index].split(',')])
        ori_length = self.df['length'].iloc[index]
        fix_length = self.df['fix_length'].iloc[index]
        
        if fix_length == ori_length:
            data_seq = torch.stack((data_seq_x, data_seq_y), dim = 0)
            return data_seq, fix_length, 1
        else:
            data_seq_list = []
            for i in [0, ori_length-fix_length]:
                seq_x = data_seq_x[i:i+fix_length]
                seq_y = data_seq_y[i:i+fix_length]
                seq = torch.stack((seq_x, seq_y), dim = 0)
                data_seq_list.append(seq)
            return data_seq_list, fix_length, 2
    
    def __len__(self):
        return len(self.df)

test_loader = DataLoader(AnDiDataset(data), batch_size=1, shuffle=False, num_workers=2)

# PyTorch Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#https://github.com/odie2630463/WaveNet/blob/master/model.py#L70
class Wave_LSTM_Layer(nn.Module):
    def __init__(self, filters, kernel_size, dilation_depth, input_dim, hidden_dim, layer_dim):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.dilations = [2**i for i in range(dilation_depth)]
        self.conv1d_tanh = nn.ModuleList([nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, 
                                                    padding=dilation, dilation=dilation) for dilation in self.dilations])
        self.conv1d_sigm = nn.ModuleList([nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, 
                                                    padding=dilation, dilation=dilation) for dilation in self.dilations]) 
        self.conv1d_0 = nn.Conv1d(in_channels=input_dim, out_channels=filters, 
                                  kernel_size=kernel_size, padding=1)
        self.conv1d_1 = nn.Conv1d(in_channels=filters, out_channels=filters, 
                                  kernel_size=1, padding=0)
        self.post = nn.Sequential(nn.BatchNorm1d(filters), nn.Dropout(0.1))
        self.lstm = LSTM(filters, hidden_dim, layer_dim, batch_first=True)
    
    def forward(self, x):
        # WaveNet Block
        x = self.conv1d_0(x)
        res_x = x
        
        for i in range(self.dilation_depth):
            tahn_out = torch.tanh(self.conv1d_tanh[i](x))
            sigm_out = torch.sigmoid(self.conv1d_sigm[i](x))
            x = tahn_out * sigm_out
            x = self.conv1d_1(x)
            res_x = res_x + x
            #x = res_x
        
        x = self.post(res_x)
        
        # LSTM Block
        x = x.permute(0,2,1)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, state = self.lstm(x, (h0.detach(), c0.detach()))
        #out = self.fc(out[:,-1,:])
        return out.permute(0,2,1)

class AnDiModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.wave_lstm_1 = Wave_LSTM_Layer(32, 3, 16, input_dim, hidden_dim, layer_dim)
        #self.wave_lstm_2 = Wave_LSTM_Layer(32, 3, 8, 16, 32, layer_dim)
        #self.wave_lstm_3 = Wave_LSTM_Layer(64, 3, 4, 32, hidden_dim, layer_dim)
        
        #for p in self.parameters():
        #    p.requires_grad=False
        
        self.fc = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        x = self.wave_lstm_1(x)
        #x = self.wave_lstm_2(x)
        #x = self.wave_lstm_3(x)
        self.feature_map = x.permute(0,2,1)[:,-1,:]
        return self.fc(x.permute(0,2,1)[:,-1,:])

model = AnDiModel(2, 64, 3, 5).to(device)

# Check PyTorch Version
try:
    model.load_state_dict(torch.load('./{}/Fold0/10/bestmodel.pth'.format(Model_Path)))
except:
    print('fail to load model file, please check the PyTorch version (1.6.0 is required).')

output_list_folds = []

for fold in range(5):

    output_list = []

    for seq_batch, seq_length, seq_mark in tqdm(test_loader):

        model.load_state_dict(torch.load('./{}/Fold{}/{}/bestmodel.pth'.format(Model_Path, fold, int(seq_length))));
        model.eval()

        with torch.no_grad():
            if int(seq_mark) == 1:
                seq_batch = seq_batch.to(device)
                output = model(seq_batch)
            elif int(seq_mark) == 2:
                output_sum = 0.
                for seq in seq_batch:
                    output = model(seq.to(device))
                    output_sum += output
                output = output_sum/len(seq_batch)
            output_list.append(output.detach().cpu().numpy())

    output_list = np.array(output_list)
    output_list_folds.append(deepcopy(output_list))

output_list = sum(output_list_folds)/5.

def softmax(arr):
    exp_arr = np.exp(arr)
    sum_exp = np.sum(exp_arr)
    return exp_arr/sum_exp

with open('./output/task2-2d.txt', 'w') as f:
    for i in output_list:
        prob = softmax(i[0])
        f.write('2')
        for j in prob:
            strj = str(format(j, '.6f'))
            f.write(';'+strj)
        f.write('\n')
    f.close()
