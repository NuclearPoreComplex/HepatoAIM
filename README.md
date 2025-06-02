# <span style="font-size:larger;">HepatoAIM——基于神经网络的肝癌药物重定位
</span>

## 目录

- [HepatoAIM——基于神经网络的肝癌药物重定位](#c4aihepatoaim基于神经网络的肝癌药物重定位)
  - [目录](#目录)
  - [项目背景](#项目背景)
  - [模型介绍](#模型介绍)
    - [模型优越性](#模型优越性)
    - [数据收集](#数据收集)
    - [主要模块](#主要模块)
    - [模型训练](#模型训练)
    - [模型评估](#模型评估)
  - [示例视频](#示例视频)
  - [环境配置](#环境配置)
  - [开始](#开始)
    - [数据导入](#数据导入)
    - [数据预处理](#数据预处理)
    - [导入训练模块](#导入训练模块)
    - [筛选函数](#筛选函数)
    - [运行时交互](#运行时交互)
  - [总结](#总结)

## 项目背景

**（1）全球的医疗需求**

肝癌是高发的恶性肿瘤之一，其发病率和死亡率在许多国家和地区居高不下。传统的治疗手段如手术、化疗和放疗等，往往效果有限，且伴随副作用与预后复发等情况。因此，寻找更有效、更安全的治疗手段成为迫切的需求。

**（2）定量构效关系与深度学习技术的结合**

最新的研究进展显示，深度学习技术正在与QSAR建模相结合，推动了所谓的“深度QSAR”领域的发展。深度学习模型能够处理大规模复杂数据集，并从中识别复杂模式，这对于药物发现和化合物设计具有重要意义。深度生成和强化学习方法在分子设计中的应用，以及深度学习模型在基于结构的虚拟筛选中的使用，都是近期的研究热点。

**（3）个性化医疗的趋势**

个性化医疗是当前医疗领域的热点，根据患者的具体情况（如基因型、表型、疾病阶段等）来定制治疗方案。在肝癌领域，通过分析特定的活性位点，可以更精准地设计针对不同患者亚群的药物，从而提高治疗效果和减少不良反应。

**（4）数据科学与AI的融合**

近年来，数据科学和人工智能技术的快速发展为药物发现带来了新的机遇。AI技术，如机器学习、深度学习，能够在处理大规模生物医学数据、优化药物设计过程、提高预测准确性等方面发挥关键作用。

**（5）学术与产业的结合**

团队在学习计算机辅助药物设计课程的过程中，不仅掌握了理论知识，还通过大学生创业相关实验将理论应用于实践。这种学术与产业的紧密结合，不仅能够促进知识转化，还能培养团队的创新能力和市场意识

## 模型介绍

### 模型优越性

**（1）基于DNN的DTI筛选模型**

基于深度神经网络（DNN）的药物-靶点作用（DTI）筛选模型相较于传统的机器学习模型，如基于随机森林的LRF-DTIs模型，具有更强的特征学习能力，以及在处理大规模和复杂数据集方面更为有效。DNN通过多层的非线性变换能够捕捉和学习数据中的复杂模式和关系，这在理解分子间的复杂相互作用方面具有显著优势。在新药筛选上，由于神经网络模型拥有更强的泛化能力，性能也更好。


**（2）靶点与小分子双特征提取**

相较于现有的DNN-DTI模型，如DEEPScreen、DEDTI等只学习了靶点对应的小分子特征，我们在模型训练时除了提取小分子特征还提取了靶蛋白特征，这增加了模型的复杂度，可以提供更全面的分子-靶标相互作用视图。这种综合方法可能有助于揭示分子与靶标之间更深层次的关系，从而提高预测的准确性和可靠性。

### 数据收集

**（1）数据准备**

| 数据               | 来源 / 处理方法                           |
| ------------------ | ----------------------------------------- |
| 相关靶点           | ChemBL / 文献                             |
| 活性分子查找       | ChemBL / PubChem                          |
| 分子结构及分子处理 | Padelpy / RDKit / CCDC Python API |
| 靶点序列 / 模型    | Uniprot / PDB                             |
| 蛋白处理           | Protein-bert                              |
| 数据清洗           | sklearn                                   |
| 模型构建           | PaddlePaddle                              |

**（2）靶点确定**
- 使用chembl数据库查询“Liver cancer cell”对应智人相关靶点，并通过查阅文献进行验证与筛选，在数据清洗后确定了如下四个靶点。

| 中文                    | English                               | CHEMBL ID  | 怎样抑制癌症（抑制or激活） |
| ----------------------- | ------------------------------------- | ---------- | -------------------------- |
| 胰高血糖素受体          | Glucagon receptor                     | CHEMBL1985 | 激活                       |
| PDZ结合激酶             | PDZ-binding kinase                    | CHEMBL4896 | 抑制                       |
| 酪氨酸蛋白激酶受体 FLT3 | Tyrosine-protein kinase receptor FLT3 | CHEMBL1974 | 抑制                       |
| 前列腺素类EP1受体       | Prostanoid EP1 receptor               | CHEMBL1811 | 抑制                       |



### 主要模块

**（1）分子处理模块**

经过Python的第三方库Padelpy进行12种分子指纹提取

通过低方差滤波得到分子描述符数据集

**（2）蛋白处理模块**

通过Protein-bert模型提取蛋白序列氨基酸的全局特征

通过卷积层、池化层、全连接层等对序列进行初步处理

**（3）融合模块**

将上述模块结合输出，通过不同的标签构筑分类模型和回归模型。

通过蛋白和小分子特征数据的结合，可以提高模型的预测准确性和鲁棒性。

### 模型训练

- 对于分类模型，计算了准确率、召回率、精确度、F2、Matthews 相关系数 （MCC） 和 ROC 曲线下面积 （AUC）同时绘出了模型分类性能散点图与混淆矩阵。

- 对于回归模型，计算了决定系数（R²）、均方误差（MSE）、均方根误差（RMSE）和平均绝对误差（MAE）。

- 对于每一轮训练的模型参数都将被保存，以便最终得到最优解。我们最终以准确率，F2，AUC均一化后加权打分最小为分类模型的最优模型，以MSE，R²均一化后加权打分最大为回归模型的最优模型

### 模型评估
- 对于分类模型，该模型包含2个隐藏层，每个隐藏层有 100个神经元，具有 ReLu 激活函数、Adam 优化器、0.001学习率和批量归一化。计算了准确率（accuracy）、召回率（recall）、精确度（precision）、F2函数（我们在精准率和召回率中更加侧重召回率，FB eta的一种特殊形式）、Matthews 相关系数 （MCC） 和 ROC 曲线下面积 （AUC）同时绘出了模型分类性能散点图与混淆矩阵。在机器学习中，混淆矩阵（Confusion Matrix）是一种用来评估分类模型性能的工具，通过对比实际标签和预测标签来展示模型在各个类别上的表现，True Positives (TP)为模型正确预测为正类的样本数量，False Positives (FP)为模型错误预测为正类的样本数量，True Negatives (TN)为模型正确预测为负类的样本数量，False Negatives (FN)为模型错误预测为负类的样本数量。对应混淆矩阵的四个值作散点图分析模型分类性能，分散程度越大表明效果越好。

- 对于回归模型，决定系数（R²）用于衡量模型的拟合优度，表示模型解释的变异性占总变异性的比例，R²值越接近1，表明模型的解释能力越强，预测值与实际值之间的差异越小。均方误差（MSE）是预测值与实际值之间差的平方的平均值，是衡量模型预测精度的一个指标，MSE值越小，表示模型的预测误差越小。均方根误差（RMSE）是MSE的算术平方根，与原始数据具有相同的单位，因此更易于解释。RMSE同样是衡量模型预测精度的指标，与MSE一样，RMSE值越小，模型的预测性能越好。平均绝对误差（MAE）是预测值与实际值之间差的绝对值的平均值。这些指标共同提供了模型性能的多维评估。

- 对于每一轮训练的模型参数都将被保存，以便最终得到最优解。我们最终以准确率，F2，AUC均一化后加权打分最小为分类模型的最优模型，以MSE，R²均一化后加权打分最大为回归模型的最优模型。


## 示例视频

[示例视频：HepatoAIM——基于神经网络的肝癌药物重定位](https://www.bilibili.com/video/BV14C23YKEBq/?spm_id_from=333.999.0.0)

## 环境配置


```python
!conda create -n HepatoAIM python=3.9 --yes
!conda activate HepatoAIM

#!pip install -r requirements.txt --yes
!python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ --timeout 100 --yes
!pip install padelpy
!pip install seaborn
!pip install biopython
!pip install scipy
!pip install -U scikit-learn
```


    运行具有“base (Python 3.9.13)”的单元格需要ipykernel包。


    运行以下命令，将 "ipykernel" 安装到 Python 环境中。


    命令: "conda install -n base ipykernel --update-deps --force-reinstall"


## 开始


```python
import os
import sys

# 将 work 文件夹添加到 sys.path
sys.path.append(os.path.join(os.getcwd(), 'work'))
# 检查当前工作目录
print(os.getcwd())
# 设置work目录为工作目录
os.chdir('/home/aistudio/work')
```

### 数据导入


```python
from pre_data import pre_date
```


```python
tg_pth = "./targets/*.csv"
tg_list = ['CHEMBL1811',
            'CHEMBL1974',
            'CHEMBL1985',
            'CHEMBL4896',]
fg_pth = "./fingerprints_xml/*.xml"
fg_list = ['AtomPairs2DCount',
            'AtomPairs2D',
            'EState',
            'CDKextended',
            'CDK','CDKgraphonly',
            'KlekotaRothCount','KlekotaRoth',
            'MACCS',
            'PubChem',
            'SubstructureCount',
            'Substructure']
```

### 数据预处理
- 为项目展示速度仅处理text，其余数据已放入对应文件夹；
- 描述符仅处理两个


```python
if input('是否进行预处理：（Y/N）') == 'Y':
    # 分子描述符处理函数,为项目展示速度仅处理text，其余数据已放入对应文件夹
    # 描述符仅处理两个
    print('正在对test.csv文件进行处理演示与处理过程')
    print('为了节约时间描述符仅处理两个')
    tg_list=['test']
    fg_list=fg_list[0:2]
    # 实例化预处理模型
    pre_data_instance = pre_date(tg_list, tg_pth, fg_list, fg_pth)

    if  input('是否进行分类模型数据预处理：（Y/N）') == 'Y':
        #争对分类模型指纹化
        pre_data_instance.figured(CRS = 'C')

    if  input('是否进行回归模型数据预处理：（Y/N）') == 'Y':
        #争对回归模型指纹化
        pre_data_instance.figured(CRS = 'R')

    if  input('是否进行筛选数据预处理：（Y/N）') == 'Y':
        if input('是否使用默认筛选数据库（FDA-approved-Drug-Library）：（Y/N）') == 'N':
            print('演示模型暂不支持自定义')
            #筛选数据地址
            screening_name='FDA-approved-Drug-Library-96-well'
            #争对筛选数据指纹化
            pre_data_instance.figured_S(screening_name)
        else:
            #筛选数据地址
            screening_name='FDA-approved-Drug-Library-96-well'
            #争对筛选数据指纹化
            pre_data_instance.figured_S(screening_name)
```

### 导入训练模块

- 蛋白靶点预测模块序调用如下方法**from get_tg_token import get_tg_bert**，

    但是由于平台环境不兼容tensorflow无法调用，调用预加载的bert结果，

    使用如下代码：**from protein_bert_pre import load_bert as get_tg_bert**

* 若需要调用新的靶点蛋白数据自行训练时需要将**def_for_re.py**和**def_for_cls.py**中的代码进行更换


```python
from train_cls import train_cls
from train_re import train_re
```


```python
if input('是否进行分类模型训练：（Y/N）') == 'Y':
    tg_list = ['CHEMBL1811',
            'CHEMBL1974',
            'CHEMBL1985',
            'CHEMBL4896',]
    train_epochs = int(input('train_epochs = （推荐100,若想快速结束可输入个位数）'))
    for tg in tg_list:
        train_cls(tg , train_epochs)#靶点训练

if input('是否进行回归模型训练：（Y/N）') == 'Y':
    tg_value_type={'CHEMBL1811':['IC50','Ki'],
              'CHEMBL1974':['IC50','Activity','Inhibition','Kd'],
               'CHEMBL1985':['IC50','Ki'],
               'CHEMBL4896':['IC50','Thermal melting change'] }
    train_epochs = int(input('train_epochs = （推荐100,若想快速结束可输入个位数）'))
    for tg,types in tg_value_type.items():
        for value_type in types:
            print(f'进行靶点{tg}的{value_type}')
            train_re(tg ,value_type , train_epochs)#靶点训练
```

### 筛选函数


```python
from screening_c import screen_c_tg_list
from screening_r import screen_for_r_put_out
import pandas as pd
import paddle
```


```python
if input('是否进行药物筛选：（Y/N）') == 'Y':
    print('已导入默认药物库')
    screening_data_pth = f'./screening/20240801-L1300-FDA-approved-Drug-Library-96-well.csv'
    target_list=[
            'CHEMBL1811',
            'CHEMBL1974',
            'CHEMBL1985',
            'CHEMBL4896']
    if input('是否进行过分类筛选：(Y/N)') == 'Y':
        out =  pd.read_csv('result_c.csv')
    else:
        #筛选部分C
        out = screen_c_tg_list(screening_data_pth,target_list)
        #储存
        out.to_csv('result_c.csv', index=False)#存档
    print('药物-靶点有作用的概率如下，已保存至对应文件result_c.csv中')
    print(out)

    tg_value_type={'CHEMBL1811':['IC50','Ki'],
                'CHEMBL1974':['IC50','Activity','Inhibition','Kd'],
                'CHEMBL1985':['IC50','Ki'],
                'CHEMBL4896':['IC50','Thermal melting change'] }
    p = screen_for_r_put_out(out,tg_value_type)
    print(p)

print('结束模型体验')
```


### 运行时交互
- 是否进行预处理：（Y/N） Y
- 正在对test.csv文件进行处理演示与处理过程
- 为了节约时间描述符仅处理两个
- 是否进行分类模型数据预处理：（Y/N） Y

## 总结
本项目利用PaddlePaddle搭建了基于DNN的DTI筛选模型，利用了深度学习在处理复杂生物医学数据上的优势,提高了肝癌靶向药物发现的效率和针对性。未来,该系统可为制药企业、医疗机构等提供有价值的辅助决策支持,在促进个性化治疗、加速新药研发等方面具有一定的前景。
