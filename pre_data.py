import glob
from padelpy import padeldescriptor
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
class pre_date:
    
    def __init__(self,tg_list,tg_pth,fg_list,fg_pth):
        #初始化数据集路径
        self.tg_pth = tg_pth
        self.tg_list = tg_list
        self.tg_list.sort()
        self.fg_pth = fg_pth
        self.fg_list = fg_list
        #self.fg_list.sort()
        self.save_r_figered_pth = './fingered_r_data/'
        self.save_c_figered_pth = './fingered_c_data/'
        self.save_s_figered_pth = './fingered_s_data/'
        self.target_pth='./targets/'
        self.threshold = 0.5


    def sort_file(self):
        tg_file = sorted(glob.glob(self.tg_pth))
        tg_dic = dict(zip(self.tg_list, tg_file))
        fg_file = sorted(glob.glob(self.fg_pth))
        fg_dic = dict(zip(self.fg_list, fg_file))
        print(fg_dic)
        return tg_dic,fg_dic
        
    def mergeC(self,df):#分类模型的合并同一数值
        # 根据'Molecule ChEMBL ID'和'Smiles'列分组
        grouped = df.groupby(['Molecule ChEMBL ID','Smiles'])
        # 计算每组的均值，并创建新行的列表
        new_rows = []
        indexes_to_drop = []
        for name, group in grouped:
            if len(group) > 1:
                # 如果有多个相同的Molecule ChEMBL ID和Smiles，则计算均值
                mean_value = group['Standard Value'].mean()
                # 创建新行，注意使用正确的索引引用
                new_rows.append({'Molecule ChEMBL ID': name[0],
                                'Smiles': name[1],
                                'Standard Value': mean_value})
                # 添加所有重复行的索引到indexes_to_drop列表
                indexes_to_drop.extend(group.index)
        # 将新行列表转换为DataFrame
        new_df = pd.DataFrame(new_rows)
        # 在循环结束后，一次性删除所有需要删除的索引
        if indexes_to_drop:
            df = df.drop(indexes_to_drop, errors='ignore')  # 使用errors='ignore'来避免删除不存在的索引
        df = pd.concat([df, new_df], ignore_index=True)
        # 现在df包含了更新后的数据
        return df

    def mergeR(self,df):#回归模型的合并同一数值
        # 根据'Molecule ChEMBL ID'和'Smiles'列分组
        grouped = df.groupby(['Molecule ChEMBL ID','Smiles','Standard Type'])
        # 计算每组的均值，并创建新行的列表
        new_rows = []
        indexes_to_drop = []
        for name, group in grouped:
            if len(group) > 1:
                # 如果有多个相同的'Molecule ChEMBL ID','Smiles','Standard Type'，则计算均值
                mean_value = group['Standard Value'].mean()
                # 创建新行，注意使用正确的索引引用
                new_rows.append({'Molecule ChEMBL ID': name[0],
                                'Smiles': name[1],
                                'Standard Value': mean_value})
                # 添加所有重复行的索引到indexes_to_drop列表
                indexes_to_drop.extend(group.index)
        # 将新行列表转换为DataFrame
        new_df = pd.DataFrame(new_rows)
        # 在循环结束后，一次性删除所有需要删除的索引
        if indexes_to_drop:
            df = df.drop(indexes_to_drop, errors='ignore')  # 使用errors='ignore'来避免删除不存在的索引
        df = pd.concat([df, new_df], ignore_index=True)
        # 现在df包含了更新后的数据
        return df
    
    def remove_low_variance(self,input_data):
        # 创建VarianceThreshold对象
        selection = VarianceThreshold(self.threshold)
        # 拟合数据并尝试选择特征
        try:
            mask = selection.fit_transform(input_data)
            # 使用布尔索引来选择被保留的特征
            selected_columns = input_data.columns[selection.get_support(indices=True)]
            return input_data[selected_columns]
        except ValueError as e:
            # 如果没有特征满足方差阈值，则捕获异常
            # 返回0
            return pd.DataFrame()

    def log_in_data(self,target):#导入时处理
        data_name=self.target_pth+target+'.csv'
        df = pd.read_csv(data_name, sep=';')
        df = pd.concat([df['Molecule ChEMBL ID'], df['Smiles'],df['Molecular Weight'], df['Standard Type'],
                     df['Standard Relation'], df['Standard Value'],
                     df['Standard Units'], df['Assay Type']], axis=1)
        #清除空白
        df_filtered = df[(df['Standard Type'].notna() & df['Standard Type'] != '') &
                 (df['Standard Relation'].notna() & df['Standard Relation'] != '')]
        #只保留Standard Relation为“=”的行
        df = df[df['Standard Relation'] == "'='"]
        # 只保留B型
        df = df[df['Assay Type'] == 'B']
        # 使用Pandas的布尔索引和向量化操作来找到单位为 'ug.mL-1' 的行，并执行转换
        df.loc[df['Standard Units'] == 'ug.mL-1', 'Standard Value'] = \
        df[df['Standard Units'] == 'ug.mL-1']['Standard Value'] / \
          (df['Molecular Weight'] * 1e-6)  # 将ug/mL转换为nmol/L# 转换为纳摩尔每升
        df['Standard Units'] = df['Standard Units'].replace('ug.mL-1', 'nM')
        return df
        
    def pre_figure_class(self,target):#分类预处理
        df=self.log_in_data(target)
        # 只保留这几位
        df = df[(df['Standard Type'] == 'IC50') |
                (df['Standard Type'] == 'AC50') |
                (df['Standard Type'] == 'Kd50') |
                (df['Standard Type'] == 'Ki50') |
                (df['Standard Type'] == 'GI50')]
        #合并同类
        df = pd.concat([df['Molecule ChEMBL ID'],
                        df['Smiles'],
                        df['Standard Value']],
                        axis=1)
        df=self.mergeC(df)
        #分类标记
        #获取中位数
        middle = df['Standard Value'].median()
        df['Activity'] = df.apply(lambda row: 'Active' if row['Standard Value'] < middle else 'Inactive', axis=1)
        df = pd.concat([df['Molecule ChEMBL ID'], df['Smiles'], df['Activity']], axis=1)
        df = df.reset_index(drop=True)#更新索引
        return df
        
    def pre_figure_regression(self,target):#回归预处理
        df=self.log_in_data(target)
        #合并同类
        df = pd.concat([df['Molecule ChEMBL ID'],
                        df['Smiles'],
                        df['Standard Type'],
                        df['Standard Value']],
                        axis=1)
        df=self.mergeR(df)
        #分类标记
        # 根据'Standard Type'列对DataFrame进行分组
        grouped = df.groupby('Standard Type')
        # 初始化一个空的字典来存储每种类型的DataFrame
        type_dfs = {}
          
        # 遍历分组，筛选出大小大于或等于100的组，并创建新的DataFrame(样本量大于100）
        for type_name, group in grouped:
            if len(group) >= 100:
                type_dfs[type_name] = group
        concatenated_df = pd.concat(type_dfs.values(), ignore_index=True)

        return concatenated_df
    
            
    def fged(self,save_pth,tg_data_use,fg,fg_dic,CRS,target):
        # 导出数据
        tg_data_use.to_csv(f'./figering_data/molecule_{target}.smi', sep='\t', index=False, header=False)

        # 生成描述符
        fingerprint_descriptortypes = fg_dic[fg]
        fingerprint_output_file = ''.join(['./figering_data/','_',fg,'.csv']) #组合一个输出文件名get_fingers/tg_fp.csv
        print(fingerprint_descriptortypes,fingerprint_output_file)
              
        padeldescriptor(mol_dir=f'./figering_data/molecule_{target}.smi',
                    #指定分子输入文件的路径，这里假设是一个包含SMILES字符串的文件。
                    d_file=fingerprint_output_file, #'fingerprint_output_file.csv'
                    #descriptortypes='SubstructureFingerprint.xml',
                    descriptortypes= fingerprint_descriptortypes,
                    #指定指纹类型的XML配置文件路径
                    detectaromaticity=True,
                    #开启芳香性检测
                    standardizenitro=True,
                    #标准化异构体。
                    standardizetautomers=True,
                    #标准化异构体。
                    threads=2,
                    #设置使用2个线程进行计算。
                    removesalt=True,
                    #移除分子中的盐。
                    log=True,
                    #记录处理状态到日志文件。
                    fingerprints=True)#开启指纹计算
                
        # 导入刚生成的数据
        df2 = pd.read_csv(fingerprint_output_file)
        df2.fillna(0, inplace=True)
        # 只选择第二列之后
        df3 = df2.iloc[:, 1:] 
        if CRS == 'C' or CRS == 'R': #筛选有其他处理手段
            # 低方差滤波
            df3  = self.remove_low_variance(df3)
            print(CRS)
        df3 = pd.concat([df2.iloc[:, 0:1], df3], axis=1)
        return df3

    def get_fg_reg(self,target,tg_data,CRS):
        save_pth = self.save_r_figered_pth
        df_out = pd.concat([tg_data['Molecule ChEMBL ID'],  tg_data['Standard Type'],tg_data['Standard Value']], axis=1)
        tg_dic,fg_dic=self.sort_file()
        for fg in self.fg_list:
            tg_data_use = pd.concat( [tg_data['Smiles'],tg_data['Molecule ChEMBL ID']], axis=1 )
            df3 = self.fged(save_pth,tg_data_use,fg,fg_dic,CRS,target)
            df_out = pd.concat([df_out, df3], axis=1, ignore_index=False)
            print(f'靶点：{target} 完成 {fg} 指纹转化。')
        print(f'保存地址：{save_pth}{target}_r_fg.csv')
        df_out.to_csv(f'{save_pth}{target}_r_fg.csv', index=False)
        return df_out

    def get_fg_cls(self,target,tg_data,CRS):
        save_pth = self.save_c_figered_pth
        df_out = pd.concat([tg_data['Molecule ChEMBL ID'],  tg_data['Activity']], axis=1)
        tg_dic,fg_dic=self.sort_file()
        tg_data_use = pd.concat( [tg_data['Smiles'],tg_data['Molecule ChEMBL ID']], axis=1 )
        i = 0
        for fg in self.fg_list:
            i+=1
            df3 = self.fged(save_pth,tg_data_use,fg,fg_dic,CRS,target)
            df_out = pd.concat([df_out, df3], axis=1, ignore_index=False)
            print(f'靶点：{target} 完成 {fg} 指纹转化。')
            print(f'保存地址：{save_pth}{target}_c_{fg}.csv')
        df_out.to_csv(f'{save_pth}{target}_c_fg.csv', index=False)

    def pre_screening(self,CRS,screening_name):
        screening_pth=f'screening/{screening_name}.csv'
        df = pd.read_csv(screening_pth)
        save_pth = self.save_s_figered_pth
        df_out = pd.concat([df['ID']], axis=1)
        tg_dic,fg_dic=self.sort_file()
        for fg in self.fg_list:
            tg_data_use = pd.concat( [df['SMILES'],df['ID']], axis=1 )
            target = 'NULL'
            batch_size = 1000  # 每批处理1000条数据
            print(len(tg_data_use))
            df3 = pd.DataFrame()
            for i in range(0, len(tg_data_use), batch_size):
                imin = min(i + batch_size, len(tg_data_use))
                batch = tg_data_use.iloc[i:imin]
                batch = self.fged(save_pth,batch,fg,fg_dic,CRS,target)
                df3 = pd.concat([df3, batch], axis=0, ignore_index=True) 
            # 使用 merge 方法按照 ID 列将 df3 与 df_out 进行合并
            df3 = df3.rename(columns={'Name': 'ID'})
            df_out = df_out.merge(df3, on='ID', how='left')
            if len(df.columns) == 1 and 'ID' in df.columns:
                df_out = df_out.drop(df.index)
            df_out.to_csv(f'00000000.csv', index=False)
            #df_out = df3
            print(f'针对筛选：完成 {fg} 指纹转化。')
            print(f'保存地址：{save_pth}_s_{fg}.csv')
        # 只选择第二列之后
        return df_out

    def crs_get(self,target,df,df00):
        # 选择相同列
        df_out_new = df.iloc[:, 1:]
        cr_data_list=[]
        cr_list=['r','c']
        cr_list=['c']
        for cr in cr_list:
            #提取靶点列表
            if cr == 'r':
                p=3
            else:
                p=2
            target_data_read = pd.read_csv(f'./fingered_{cr}_data/{target}_{cr}s_fg.csv').iloc[:, p:-1] 
            # 根据 ID 合并两个 DataFrame
            merged_df = pd.merge(df, df00[['ID', 'SMILES']], on='ID', how='left')
            #print(f'{cr}读取完成')
            #获取target_data_read的列名列表
            columns_targets = target_data_read.columns.tolist()
            #oo0=df.columns.tolist()
            #print(oo0)
            #print(columns_targets)

            #从s中提取出与target_data_read列名相同的列
            df_out_new_new = df_out_new[columns_targets]
            #print(f'提取靶点为{columns_targets}')
            #print(df_out_new_new)
            
            # 使用concat函数横向合并两个DataFrame
            df_combined = pd.concat([merged_df.iloc[:, 0:1], df_out_new_new,merged_df.iloc[:, -1:]], axis=1)
            

            df_combined.to_csv(f'{self.save_s_figered_pth}{target}_s_{cr}_fg.csv', index=False)#存档
            print(f'靶点{target}的{cr}完成')
        
    def re_data_dd(self,tg):
        p = pd.read_csv(f'{self.save_r_figered_pth}{tg}_r_fg.csv')
        if p.empty:
            print('未找到文件')
        else:
            # 按 'Standard Type' 列对 DataFrame 进行分组
            grouped = p.groupby('Standard Type')
            # 遍历分组
            for standard_type, group in grouped:
                # 创建 CSV 文件名，根据 'Standard Type' 的值命名
                save=f'fingered_r_data_dd/R_{tg}_{standard_type}.csv'
                # 将分组的 DataFrame 保存为 CSV 文件
                group.to_csv(save, index=False)# 不保存行索引'''
            print('完成所有转化')
        
    def figured(self,CRS = 'C'):#指纹化
        if CRS == 'R' :#回归
            for target in self.tg_list:
                tg_data = self.pre_figure_regression(target)#预处理
                df0=self.get_fg_reg(target,tg_data,CRS)#指纹化
                self.re_data_dd(target)#拆分
        else:#默认，分类
            for target in self.tg_list:
                tg_data = self.pre_figure_class(target)#预处理
                self.get_fg_cls(target,tg_data,CRS)#指纹化

    def figured_S(self,screening_name,pre_or_re):#指纹化
        #数据集处理
        CRS = 'S'
        df=self.pre_screening(CRS,screening_name)#预处理+指纹化
        if len(df.columns) == 1 and 'ID' in df.columns:
            df = df.drop(df.index)
        df.to_csv(f'screening/{screening_name}_all_fed.csv', index=False)#存档

        print('已完成所有转换，即将进行分模型适配数据。')

        df00 = pd.read_csv(f'screening/{screening_name}.csv') 

        df = pd.read_csv(f'screening/{screening_name}_all_fed.csv')
        for target in self.tg_list:
            self.crs_get(target,df,df00)
            #print( dfr,dfc)