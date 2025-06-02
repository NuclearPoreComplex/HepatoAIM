import glob
from torch.utils.data import DataLoader, random_split
import torch
from torch import nn
from def_for_cls import train_c_data, Class_bert_NN, train_c, get_true_labels, Grade_c

def train_cls(tg, train_epochs):
    tg_pth = "./fingered_c_data/*.csv"
    target_seq_pth = {
        'CHEMBL1811': 'P34995.fasta',
        'CHEMBL1974': 'P36888.fasta',
        'CHEMBL1985': 'P47871.fasta',
        'CHEMBL4896': 'Q96KB5.fasta'
    }
    tg_list = list(target_seq_pth.keys())
    tg_list.sort()
    
    tg_file = sorted(glob.glob(tg_pth))
    tg_file.sort()
    tg_dic = dict(zip(tg_list, tg_file))
    batch_size = 24
    
    # 数据集加载
    data_of_desc = train_c_data(tg_dic[tg])  # tg为选择靶点
    descriptor_size = data_of_desc.feature_size
    
    # 划分训练集与测试集
    size = 0.8
    train_size = int(len(data_of_desc) * size)  # 训练集的样本数量
    test_size = len(data_of_desc) - train_size  # 测试集的样本数量
    # 随机划分数据集
    train_dataset, test_dataset = random_split(data_of_desc, [train_size, test_size])
    
    # 使用 DataLoader 来迭代训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 定义模型
    num_classes = 2  # 二分类模型[0,1]即无活性和有活性
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Class_bert_NN(tg, descriptor_size, num_classes=num_classes).to(device)
    
    # 定义损失函数
    loss_fn = nn.BCEWithLogitsLoss()
    
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=torch.optim.Adam(model.parameters(), lr=0.001), step_size=1000, gamma=0.9)
    
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 开始训练
    train_epochs = train_epochs  # 您可以根据需要调整训练的轮数
    put_outs, model_data, losses = train_c(model, optimizer, loss_fn, train_loader, test_loader, train_epochs)
    
    # 获取测试集的值
    true_labels = get_true_labels(test_loader)
    
    # 实例化评价体系
    this_Grade_c = Grade_c(put_outs, true_labels, losses, tg, model_data)
    
    # 保存参数
    return this_Grade_c.save_all()
    
    # 绘制图像
    # this_Grade_c.get_fig()

    # return '完成训练'