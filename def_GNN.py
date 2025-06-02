#将SMILES转换为分子图
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from rdkit import RDLogger
# 禁用 RDKit 的警告信息
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def smiles_to_graph(df):

    #print(df)
    #print(df['Smiles'])
    mol = Chem.MolFromSmiles(df['Smiles'])
    #print(mol)
    if mol is None:
        return None  # 过滤无效SMILES
    # 添加氢原子
    mol = Chem.AddHs(mol)
    '''
    # 生成三维坐标
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    '''
    AllChem.EmbedMolecule(mol)  # UFF 嵌入
    # 尝试使用 MMFF 进行优化
    try:
        opt_status = AllChem.MMFFOptimizeMolecule(mol)
        if opt_status != 0:  # 如果 MMFF 优化失败
            raise ValueError
    except ValueError:
        try:
            # 尝试使用 UFF 进行优化
            opt_status = AllChem.UFFOptimizeMolecule(mol)
            if opt_status != 0:  # 如果 UFF 优化失败
                raise ValueError
        except ValueError:
            # 三维构象优化失败，生成二维坐标
            AllChem.Compute2DCoords(mol)
    '''
    # 打印优化后构象信息
    try:
        final_conf = mol.GetConformer()
        print(f"优化后构象信息 - SMILES: {df['Smiles']}, ","原子数: {mol.GetNumAtoms()}, 坐标: {final_conf.GetPositions()}")
    except ValueError:
        print(f"优化后构象获取失败，跳过 SMILES: {df['Smiles']}")
        return None
    '''
    # 简单的电负性映射
    electronegativity_map = {
        'H': 2.20, 'He': None, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44,
        'F': 3.98, 'Ne': None, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58,
        'Cl': 3.16, 'Ar': None, 'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
        'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01,
        'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.00, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
        'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
        'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.6, 'Cs': 0.79, 'Ba': 0.89,
        'La': 1.1, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.2, 'Gd': 1.2,
        'Tb': 1.2, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27, 'Hf': 1.3,
        'Ta': 1.5, 'W': 2.36, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.2, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00,
        'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.0, 'At': 2.2, 'Rn': 2.2, 'Fr': 0.7, 'Ra': 0.9,
        'Ac': 1.1, 'Th': 1.3, 'Pa': 1.5, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.3, 'Cm': 1.3,
        'Bk': 1.3, 'Cf': 1.3, 'Es': 1.3, 'Fm': 1.3, 'Md': 1.3, 'No': 1.3, 'Lr': 1.3
    }
    # 节点特征（原子属性）
    node_features = []
    for atom in mol.GetAtoms():
        # 原子特征：原子类型、电荷、价态、杂化等
        element_symbol = atom.GetSymbol()
        electroneg = electronegativity_map.get(element_symbol, None)
        features = [
            atom.GetAtomicNum(),  # 原子序数
            atom.GetDegree(),  # 连接的键数
            atom.GetFormalCharge(),  # 形式电荷
            atom.GetHybridization(),  # 杂化类型
            atom.GetIsAromatic(),  # 是否芳香族
            electroneg,  # 电负性
            atom.IsInRing(),  # 是否在环内
            atom.GetTotalDegree(),  # 总成键数
            atom.GetChiralTag().real,  # 手性
            atom.GetNumImplicitHs(),  # 隐式氢数量
            atom.GetNumExplicitHs(),  # 显式氢数量
            atom.IsInRingSize(3),  # 是否在三元环中
            atom.IsInRingSize(4),  # 是否在四元环中
            atom.GetExplicitValence(),#显式价电子数。
            atom.GetImplicitValence()#隐式价电子数。
        ]
        # 如果分子有三维坐标，添加键的长度
        node_features.append(features)
    
    # 边特征（键属性）
    edge_indices = []
    edge_features = []
    def bond_len(i,j,mol):
        conf = mol.GetConformer()
        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        if len(pos_i) == 2:
            bond_length = ((pos_i.x - pos_j.x) ** 2 + (pos_i.y - pos_j.y) ** 2) ** 0.5
        else:
            bond_length = ((pos_i.x - pos_j.x) ** 2 + (pos_i.y - pos_j.y) ** 2 + (pos_i.z - pos_j.z) ** 2) ** 0.5
        return bond_length

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # 获取键两端的原子
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        # 尝试计算 Gasteiger 电荷差值
       
        # 使用电负性差值替代
        symbol_i = atom_i.GetSymbol()
        symbol_j = atom_j.GetSymbol()
        electroneg_i = electronegativity_map.get(symbol_i, 0)
        electroneg_j = electronegativity_map.get(symbol_j, 0)
        charge_diff = electroneg_i - electroneg_j

        # 边特征：键类型、是否共轭、是否环内键
        edge_features.append([
            bond.GetBondTypeAsDouble(),  # 键类型（单、双、三键）
            bond.GetIsConjugated(),       # 是否共轭
            bond.IsInRing(),               # 是否在环内
            bond.GetStereo(),              # 键的立体化学信息
            bond.GetIsAromatic(),#芳香性
            charge_diff,  # 键两端原子部分电荷差值
            bond_len(i,j,mol)
        ])


        # 无向图需添加双向边
        edge_indices.extend([[i, j], [j, i]])
    
    # 转换为PyTorch Tensor
    node_features = torch.tensor(node_features, dtype=torch.float)# 节点特征矩阵：[num_atoms, num_features]
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()# 边索引：[2, num_edges]
    edge_features = torch.tensor(edge_features, dtype=torch.float)# 边特征矩阵：[num_edges, num_edge_features]
    
    scaler_node = MinMaxScaler()
    scaler_edge = MinMaxScaler()
    
    node_features = torch.tensor(scaler_node.fit_transform(node_features), dtype=torch.float)
    edge_features = torch.tensor(scaler_edge.fit_transform(edge_features), dtype=torch.float)
    
    #y=torch.tensor(df['Activity'])
    return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_features) # 返回PyG的Data对象, y=y