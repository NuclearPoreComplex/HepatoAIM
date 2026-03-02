import pandas as pd
from rdkit import Chem
# 处理 SMILES 列
def process_smiles(smiles):
    if isinstance(smiles, str):  # 确保是字符串
        parts = smiles.split('.')
        if len(parts) > 1:
            # 找到最长的子字符串
            max_part = max(parts, key=len)
            return max_part
        else:
            return smiles
    else:
        return smiles  # 如果不是字符串，直接返回

# 应用函数到 df2 的 'SMILES' 列

def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=False)
        return smiles
    except:
        return smiles
import pandas as pd

def add_smiles(df1,df2):
    # 使用 merge 函数根据 id 列合并 df2 的 simle 列到 df1 中
    df2['SMILES'] = df2['SMILES'].apply(process_smiles)

    merged_df = pd.merge(df1, df2[['Cat', 'SMILES']], on='Cat', how='inner')
    # 对 Smiles 列进行规范处理
    merged_df['Cat'] = merged_df['SMILES'].apply(canonicalize_smiles)
    # 对于相同的 Molecule ChEMBL ID，只保留第一个 Smiles 式子
    merged_df = merged_df.drop_duplicates(subset='Cat', keep='first')
    return merged_df