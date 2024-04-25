from collections import Counter
import torch
from torch import nn
from torch import optim
import math
# import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from torch.utils.data import random_split
from collections import Counter
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
# from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import random_split
import os
import h5py
SEED = 1210
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED) #设置几乎所有的随机种子 随机种子，可使得结果可复现
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

word_enbeding={"A":[1,0,0,0],"C":[0,1,0,0],"T":[0,0,1,0],"G":[0,0,0,1],"N":[0,0,0,0]}


# Gfish_dna = pd.read_csv('/home/u2208283040/tzx/LSTM/data_x/Gfish.csv')
# Hfish_dna = pd.read_csv('/home/u2208283040/tzx/LSTM/data_x/Hfish.csv')

# print(Hfish_dna.head())
# print(Gfish_dna.head())
# # # print("读数据结束")
def Kmers_funct(seq, size=500):
    return [seq[x:x+size] for x in range(len(seq) - size + 1)]
# data=[]
# lable=[]
# for i in tqdm(Gfish_dna['data']):
#     for item in Kmers_funct(i):
#         data.append([item])
#         lable.append(1)
# for i in tqdm(Hfish_dna['data']):
#     for item in Kmers_funct(i):
#         data.append([item])
#         lable.append(0)
# # print(data)
# # print(y)
# lable=np.array(lable)
# np.save('/home/u2208283040/tzx/LSTM/data_x/lable.npy',lable)   # 保存为.npy格式

# dt=h5py.special_dtype(vlen=str)
# data=np.array(data)
# with h5py.File('/home/u2208283040/tzx/LSTM/data_x/fish.hdf5','w') as f:
#     ds=f.create_dataset('data',data.shape,dtype=dt)
#     ds[:]=data

# #数据编码
data_x=[]
with h5py.File('/home/u2208283040/tzx/LSTM/data_x/fish.hdf5','r') as f:
    dset=f["data"]
    for i in tqdm(dset[:]):
        data_x_one=[]
        for i in i[0].decode():
            data_x_one.extend([word_enbeding[i]])
            # data_x_one.append(word_enbeding[i])
        data_x.extend([data_x_one])
        # data_x.append(data_x_one)
data_y=np.load('/home/u2208283040/tzx/LSTM/data_x/lable.npy')
data_y=data_y.tolist()

class mydataset(Dataset):
    def __init__(self): # 读取加载数据:torch.Size([423, 6, 4])
        self._x=torch.tensor(np.array(data_x).astype(float))
        self._y=torch.tensor(np.array(data_y).astype(float))
        self._len=len(data_y)
        
    def __getitem__(self,item):
        return self._x[item],self._y[item]
    
    def __len__(self):# 返回整个数据的长度 
            return self._len
        
data=mydataset()

# 划分 训练集 测试集 
train_data,test_data=random_split(data,[round(0.8*data._len),round(0.2*data._len)])#这个参数有的版本没有 generator=torch.Generator().manual_seed(0)
epochs = 1000
batch_size = 128
# 在模型测试中 这两个值：batch_size = 19 固定得 epochs = 随便设置
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

class CNN_LSTM(nn.Module):  # 注意Module首字母需要大写
    def __init__(self,):
        super().__init__()
        input_size = 3
        hidden_size = 32
        output_size = 32
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=2,stride=1)
        # input_size：输入lstm单元向量的长度 ，hidden_size输出lstm单元向量的长度。也是输入、输出隐藏层向量的长度
        self.lstm = nn.LSTM(input_size,output_size,num_layers=1)  # ,batch_first=True
#         self.linear_1 = nn.Linear(output_size,1)
#         self.ReLU = nn.ReLU()
        self.linear_2 = nn.Linear(128,2)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x,batch_size):
        x = x.type(torch.FloatTensor)
        x=x.to(device)
        x=x.unsqueeze(1)
        x =self.conv1(x) 
        x=x.squeeze(1)
        # 输入 lstm的矩阵形状是：[序列长度，batch_size,每个向量的维度] [序列长度,batch, 64]
        lstm_out,(h_n,c_n)= self.lstm(x, None)
        lstm_out=lstm_out.view(batch_size,-1)
        print(lstm_out.shape)
        lstm_out=self.linear_2(lstm_out)
        prediction=self.softmax(lstm_out)
        return prediction
# 这个函数是测试用来测试x_test y_test 数据 函数
def eval_test(model):  # 返回的是这10个 测试数据的平均loss
    test_epoch_loss = []
    with torch.no_grad():
        optimizer.zero_grad()
        for step, (test_x, test_y) in enumerate(test_loader):
            y_pre = model(test_x,batch_size )
            test_y = test_y.to(device)        
            test_loss = loss_function(y_pre, test_y.long())
            test_epoch_loss.append(test_loss.item())
    return np.mean(test_epoch_loss)

# 创建LSTM()类的对象，定义损失函数和优化器

model = CNN_LSTM().to(device)
loss_function = torch.nn.CrossEntropyLoss().to(device)# 损失函数的计算 交叉熵损失函数计算
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器实例
print(model)


sum_train_epoch_loss = []  # 存储每个epoch 下 训练train数据的loss
sum_test_epoch_loss = []  # 存储每个epoch 下 测试 test数据的loss
best_test_loss = 10000
for epoch in range(epochs):
    epoch_loss = []
    for step, (train_x, train_y) in enumerate(train_loader):
        y_pred = model(train_x,batch_size)
        # 训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值
        single_loss = loss_function(y_pred.cpu(),train_y.long())
        single_loss.backward()  # 调用backward()自动生成梯度
        optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络
        
        epoch_loss.append(single_loss.item())
        
    train_epoch_loss = np.mean(epoch_loss)
    test_epoch_loss = eval_test(model)  # 测试数据的平均loss
    
    if test_epoch_loss<best_test_loss:
        best_test_loss=test_epoch_loss
        print("best_test_loss",best_test_loss)
        best_model=model
    sum_train_epoch_loss.append(train_epoch_loss)
    sum_test_epoch_loss.append(test_epoch_loss)
    print("epoch:" + str(epoch) + "  train_epoch_loss： " + str(train_epoch_loss) + "  test_epoch_loss: " + str(test_epoch_loss))

torch.save(best_model, 'best_model.pth')
