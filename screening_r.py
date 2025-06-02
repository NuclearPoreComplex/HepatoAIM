import glob
import paddle
from paddle import nn
from paddle.static import InputSpec
from paddle.io import Dataset, DataLoader, random_split
import paddle.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from def_for_re import sreen_r_data,re_bert_NN

tg = 'CHEMBL4896'

def screen_for_r(tg , data,type00 ):
    # 数据集加载
    data_of_desc = sreen_r_data(data) #tg为选择靶点
    descriptor_size = data_of_desc.feature_size
    
    #划分筛选集
    screen_loader = DataLoader(data_of_desc, batch_size=64, shuffle=False)
    
    # 实例化模型
    num_classes = 1  # 回归问题
    model = re_bert_NN(tg,descriptor_size , num_classes=num_classes).to('gpu:0')
    #加载
    model.set_state_dict(paddle.load(f'./model/{tg}__{type00}_re_best_model.pdparams'))
    
    put_out0=[]
    #进行筛选
    for batch_id, descriptors in enumerate(screen_loader):
        # 前向传播
        predictions = model(descriptors)
        put_out0.append(predictions)
    put_out = paddle.concat(put_out0, axis=0)
    #处理成概率
    put_out=np.asarray(put_out.numpy())#转化为np数组
    
    return put_out

def screen_for_r_put_out(out,tg_value_type):
    for tg,types in tg_value_type.items():
        #对于靶点，提取out中score大于0.95的cat编号，形成新的list
        # 直接使用条件过滤
        high_score_df = out[out[tg] > 0.95]
        # 提取cat编号
        high_score_cat = high_score_df['Cat'].tolist()
        #获取预处理文件位置
        screening_pth = f'./fingered_s_data/{tg}_s_r_fg.csv'
        re_scr = pd.read_csv(screening_pth)
        # 使用isin方法提取特定Cat编号的行
        selected_rows = re_scr[re_scr['Cat'].isin(high_score_cat)]
        selected_rows_out = selected_rows.iloc[:, 0]
        for type00 in types:
            print(f'进行靶点{tg}的{type00}预测')
            p = screen_for_r(tg , selected_rows,type00 )#获取预测序列
            filepath =f"./fingered_r_data_dd/R_{tg}_{type00}.csv"
            p = p[:, 0]
            selected_rows_out = pd.concat([selected_rows_out.reset_index(drop=True), pd.Series(p, name=f'{tg}_{type00}')], axis=1)
        selected_rows_out.to_csv(f'result_r_{tg}.csv', index=False)#存档
    return '完成预测'
