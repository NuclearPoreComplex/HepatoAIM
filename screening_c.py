import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

from def_c_323 import Class_Bert_NN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集加载
from def_GNN import smiles_to_graph
from torch.utils.data import Dataset 
from torch_geometric.data import Batch, Data  # 添加Batch的导入
from torch_geometric.loader import DataLoader  # 确保使用PyG的DataLoader

def collate_fn(data_list):
    descriptors_list = []
    graphs_list = []


    for descriptors,graph  in data_list:
        descriptors_list.append(descriptors)
        graphs_list.append(graph)

    # 将描述符和标签转换为张量批次
    descriptors_batch = torch.stack(descriptors_list)
    # 将图数据合并为一个批次
    graphs_batch = Batch.from_data_list(graphs_list)


    return descriptors_batch, labels_batch, graphs_batch,graphs_batch_1,graphs_batch_2

class S_C_Data(Dataset):
    def __init__(self, filepath):
        # 导入数据
        self.df = pd.read_csv(filepath)
        df = self.df.iloc[:, 1:]
        
        # 标准化特征
        scaler = StandardScaler()
        #0是acitivate,-1 is smiles
        #print(df)
        df_x = scaler.fit_transform(df.iloc[:, 1:-1])  # 最大最小值
        df_x = pd.DataFrame(df_x)
        df_Standard = pd.concat([df.iloc[:, 0].to_frame(), df_x,], axis=1)

        #分子图处理
        smiles_to_graph
        # 使用 apply 方法处理 'smiles' 列
        df.rename(columns={'SMILES': 'Smiles'}, inplace=True)  # 小写→首字母大写
        df['graph'] = df.apply(smiles_to_graph,axis=1)

        # 将 dfs 列转换为 NumPy 数组
        graph_data = df['graph'].tolist()

        # 转换为 float32 类型的 NumPy 数组
        arr = df_Standard.values.astype(np.float32)

        # 转换为张量
        ts = torch.tensor(arr, dtype=torch.float32)
        # 划分特征和标签
        self.X = ts[:, :]  # 第 2 列及以后为输入特征,最后一列是smiles
        #self.y = torch.nn.functional.one_hot(self.y, num_classes=2)  # 将标签转换为 one-hot 编码
        self.graph = graph_data
        # 样本总数
        self.len = ts.shape[0]
        
        # 特征数
        self.feature_size = self.X.shape[1] 
        self.date_value = self.X.shape[0] 

    def __getitem__(self, index):
        graph_data = self.graph[index]
         # 返回单个样本的特征和标签
        return self.X[index],  graph_data

    def __len__(self):
        # 返回数据集的样本总数
        return self.len

def screen_for_c(tg, screening_pth):
    # 获取筛选后的预处理地址
    df = pd.read_csv(screening_pth)

    # 数据集加载
    data_of_desc = S_C_Data(screening_pth)  # tg为选择靶点
    descriptor_size = data_of_desc.feature_size

    # 划分筛选集
    screen_loader = DataLoader(data_of_desc, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # 实例化模型
    num_classes = 2  # 二分类模型[0,1]即无活性和有活性
    test_num = 0
    model = Class_Bert_NN(tg, descriptor_size, test_num,num_classes=num_classes,)
    # 加载
    model.load_state_dict(torch.load(f'./best_model/{tg}_classify_best_model.pth'))
    model = model.to(device)
    model.eval()

    put_out0 = []
    # 进行筛选
    with torch.no_grad():
        for batch_id,( descriptors, graph) in enumerate(screen_loader):
            # 将数据移动到与模型相同的设备上
            descriptors,graph= descriptors.to(device),graph.to(device)
            # 前向传播
            predictions,_ = model(descriptors, graph)
            put_out0.append(predictions)
    put_out = torch.cat(put_out0, dim=0)  # 合并所有批次的输出
    '''处理成概率'''
    #put_out = F.softmax(put_out, dim=1)  
    put_out = put_out.detach().cpu().numpy()  # 转化为 np 数组
    return put_out[:, 1]

# 筛选部分
def screen_c_tg_list(screening_data_pth, target_list):
    df = pd.read_csv(screening_data_pth)
    df_out = df.iloc[:, 0]
    for tg in target_list:
        print(f'正在筛选{tg}')
        screening_pth = f'./fingered_s_data/{tg}_s_c_fg.csv'
        out = screen_for_c(tg, screening_pth)  # [有活性的概率]
        # 使用 concat 拼接
        df_out = pd.concat([df_out.reset_index(drop=True), pd.Series(out, name=tg)], axis=1)
    return df_out
