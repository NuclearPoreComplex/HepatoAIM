# 模型主体
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Class_Bert_NN(nn.Module):
    def __init__(self, tg, descriptor_size, test_num, num_classes=1 ):
        date_value = test_num
        super(Class_Bert_NN, self).__init__()
        '''描述符处理部分'''
        layers = [
            nn.Linear(descriptor_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        ]
        # 针对复杂度进一步处理
        date_value += 0
        for _ in range(date_value):
            layers.extend([
                TransformerEncoderLayer(
                    d_model=128,
                    nhead=2,
                    dim_feedforward=256,
                    dropout=0.1
                ),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.3),
            ])

        self.desc_net = nn.Sequential(*layers)

        '''图神经网络（带残差）'''
        heads_num = [2,4,8,16,32,64,64,64,64][date_value]
        heads_num = 4
        self.conv1 = TransformerConv(15, 64, heads=heads_num)
        self.bn0 = nn.LayerNorm(64 * heads_num)
        self.res_fc = nn.Linear(15, 64 * heads_num)  # 残差连接适配层
        self.res_weight = nn.Parameter(torch.tensor(1.0))  # 可学习的残差权重
        self.res_bn = nn.LayerNorm(64 * heads_num)#残差分支的归一化层

        self.graph_proj = GATConv(64 * heads_num, 64)

        self.graph_norm = nn.LayerNorm(64)
        
        self.drop_edge_prob = 0.1
        self.mask_node_prob = 0.1

        
        #self.gate_fc = nn.Linear(64*2, 1)  # 恢复门控机制
        #self.sigmoid = nn.Sigmoid()
        '''门控融合'''
        # 多头注意力融合
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        # 拼接融合
        self.attn_norm = nn.LayerNorm(64*3)
        self.desc_feat_weight= nn.Parameter(torch.tensor(1.0))  # 可学习的拼接权重
        self.graph_feat_weight= nn.Parameter(torch.tensor(1.0))  # 可学习的拼接权重
        self.fused_feat_weight= nn.Parameter(torch.tensor(1.0))  # 可学习的拼接权重

        '''分类头'''
        self.classifier = nn.Sequential(
            nn.Linear(64*3, 256), 
            nn.LayerNorm(256),      
            nn.Linear(256, 256), 
            nn.LayerNorm(256),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )


        '''对比学习投影头'''
        self.projection_head = nn.Sequential(
            nn.Linear(64*3, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Linear(128, 64)
        )

        # 保存设备信息
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')