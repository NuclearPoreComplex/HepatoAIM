import pandas as pd
from rdkit import Chem

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
    merged_df = pd.merge(df1, df2[['Molecule ChEMBL ID', 'Smiles']], on='Molecule ChEMBL ID', how='inner')
    # 对 Smiles 列进行规范处理
    merged_df['Smiles'] = merged_df['Smiles'].apply(canonicalize_smiles)
    # 对于相同的 Molecule ChEMBL ID，只保留第一个 Smiles 式子
    merged_df = merged_df.drop_duplicates(subset='Molecule ChEMBL ID', keep='first')
    return merged_df