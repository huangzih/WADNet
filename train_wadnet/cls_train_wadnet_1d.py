import pandas as pd
import numpy as np
import os
import gc
import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--l', type=int)
arg('--f', type=int)
args = parser.parse_args()

l = args.l
valid_idx = int(args.f)
print('The validation fold is {}'.format(valid_idx))

model_path = './models/Fold{}/{}/'.format(valid_idx, l)
if not os.path.exists(model_path):
    os.makedirs(model_path)
filename = './pp_data/data-1d-{}-pp.csv'.format(l)
data = pd.read_csv(filename, sep=';')

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

data['fold'] = 0
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for idx, (train_idx, valid_idx) in enumerate(kf.split(data)):
    data['fold'].iloc[valid_idx] = idx

from torch import nn, optim
from torch.nn import functional as F
from torch.nn import LSTM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import *
from tqdm import tqdm

valid_idx = int(args.f)
train_df = data[data['fold']!=valid_idx].reset_index(drop=True)
valid_df = data[data['fold']==valid_idx].reset_index(drop=True)
print('There are {} samples in the training set.'.format(len(train_df)))
print('There are {} samples in the validation set.'.format(len(valid_df)))

class AnDiDataset(Dataset):
    def __init__(self, df, label=True):
        self.df = df.copy()
        self.label = label
        
    def __getitem__(self, index): 
        data_seq = torch.Tensor([float(i) for i in self.df['new_pos'].iloc[index].split(',')])
        if self.label:
            target = int(self.df['label'].iloc[index])
        else:
            target = 0
        return data_seq.unsqueeze(-1).permute(1,0), target
    
    def __len__(self):
        return len(self.df)

train_loader = DataLoader(AnDiDataset(train_df), batch_size=512, shuffle=True, num_workers=2)
valid_loader = DataLoader(AnDiDataset(valid_df), batch_size=512, shuffle=True, num_workers=2)

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
        return self.fc(x.permute(0,2,1)[:,-1,:])

criterion = nn.CrossEntropyLoss()
#metric = nn.L1Loss()
model = AnDiModel(1, 64, 3, 5).to(device)

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

def train_model(epoch, history=None):
    model.train() 
    t = tqdm(train_loader)
    
    for batch_idx, (seq_batch, label_batch) in enumerate(t):
        
        seq_batch = seq_batch.to(device)
        label_batch = label_batch.to(device)
        
        optimizer.zero_grad()
        output = model(seq_batch)
        loss = criterion(output.squeeze(), label_batch)
        t.set_description(f'train_loss (l={loss:.4f})')
        
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()    
        optimizer.step()
    
    torch.save(model.state_dict(), model_path+'epoch{}.pth'.format(epoch))

def evaluate(epoch, history=None): 
    model.eval() 
    valid_loss = 0.
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for batch_idx, (seq_batch, label_batch) in enumerate(valid_loader):
            all_targets.append(label_batch.numpy().copy())
            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device)

            output = model(seq_batch)
            loss = criterion(output.squeeze(), label_batch)
            #mae = metric(output.squeeze(), label_batch.float())
            valid_loss += loss.data
            all_predictions.append(np.argmax(torch.sigmoid(output.cpu()).numpy(), axis=-1))

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    valid_loss /= (batch_idx+1)
    valid_f1 = f1_score(all_targets, all_predictions, average='micro')

    if history is not None:
        history.loc[epoch, 'valid_loss'] = valid_loss.cpu().numpy()

    valid_status = 'Epoch: {}\tLR: {:.6f}\tValid Loss: {:.4f}\tValid F1: {:.4f}'.format(
        epoch, optimizer.state_dict()['param_groups'][0]['lr'], valid_loss, valid_f1)
    print(valid_status)
    with open(model_path+'log.txt', 'a+') as f:
        f.write(valid_status+'\n')
        f.close()
    
    return valid_loss, valid_f1

history_train = pd.DataFrame()
history_valid = pd.DataFrame()

n_epochs = 100
init_epoch = 0
max_lr_changes = 2
valid_losses = []
mae_metrics = []
lr_reset_epoch = init_epoch
patience = 2
lr_changes = 0
best_valid_loss = 1000.

for epoch in range(init_epoch, n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train_model(epoch, history_train)
    valid_loss, mae_metric = evaluate(epoch, history_valid)
    valid_losses.append(valid_loss)
    mae_metrics.append(mae_metric)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
    elif (patience and epoch - lr_reset_epoch > patience and
          min(valid_losses[-patience:]) > best_valid_loss):
        # "patience" epochs without improvement
        lr_changes +=1
        if lr_changes > max_lr_changes: # 早期停止
            break
        lr /= 5 # 学习率衰减
        print(f'lr updated to {lr}')
        lr_reset_epoch = epoch
        optimizer.param_groups[0]['lr'] = lr
