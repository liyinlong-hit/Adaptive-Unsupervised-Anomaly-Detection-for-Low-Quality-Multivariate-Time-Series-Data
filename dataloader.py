import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle
import GetSMGData2 as SMD

class StockDataset(Dataset):
    def __init__(self, dataPath, window, is_test=False):
        # df1=pd.read_csv(dataPath)
        AllFeature, all_is_anomaly = SMD.GetDATA()
        df1 = pickle.load(open('machine-2-1_test.pkl', 'rb'))
        df1[:, -1] = all_is_anomaly
        # print("shuju weidu ",df1[0][2])
        min_max_scaler = preprocessing.MinMaxScaler()
        df0 = min_max_scaler.fit_transform(df1)
        seq_len=window
        amount_of_features = 38#有几列


        sequence_length = seq_len + 1#序列长度
        result = []
        LLabel=[]
        for index in range(len(all_is_anomaly) - sequence_length):#循环数据长度-sequence_length次
            result.append(df0[index: index + sequence_length])#第i行到i+sequence_length
            LLabel.append(all_is_anomaly[index: index + sequence_length])
        result = np.array(result)#得到样本，样本形式为6天*3特征
        LLabel= np.array(LLabel)
        row = round(0.9 * result.shape[0])#划分训练集测试集
        train = result[:int(row), :]
        x_train = train[:, :-1]
        y_train = result[:, -1][:, -1]
        # 这行代码从result数组中选择从第row行开始到最后的所有行（int(row):）和除了最后一列之外的所有列（:-1），
        # 即提取了测试集中的特征，并将它们赋值给变量x_test。
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][:, -1]
        #reshape成 6天*3特征
        X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
        if not is_test:
            self.data = X_train
            self.label = y_train
        else:
            self.data = X_test
            self.label = y_test
            
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self,idx): 
        return torch.from_numpy(self.data[idx]).to(torch.float32), torch.FloatTensor([self.label[idx]])
