import sys

def combine_gro_files(file1, file2, output_file):
    # 读取第一个文件
    with open(file1, 'r') as f1:
        lines1 = f1.readlines()

    # 读取第二个文件
    with open(file2, 'r') as f2:
        lines2 = f2.readlines()

    # 获取第一个文件的标题和原子数
    title1 = lines1[0].strip()
    num_atoms1 = int(lines1[1].strip())

    # 获取第二个文件的标题和原子数
    title2 = lines2[0].strip()
    num_atoms2 = int(lines2[1].strip())

    # 合并原子信息
    combined_atoms = lines1[2:-1] + lines2[2:-1]

    # 获取第一个文件的盒子向量
    box_vector = lines1[-1].strip()

    # 写入合并后的文件
    with open(output_file, 'w') as fout:
        # 写入标题
        fout.write(title1 + '\n')
        # 写入总原子数
        fout.write(str(num_atoms1 + num_atoms2) + '\n')
        # 写入原子信息
        for line in combined_atoms:
            fout.write(line)
        # 写入盒子向量
        fout.write(box_vector + '\n')

def modify_topol_top(topol_file, ligand_filename):
    # 读取topol.top文件
    with open(topol_file, 'r') as f:
        lines = f.readlines()

    # 查找插入位置（在"[ moleculetype ]"之前）
    insert_itp = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('[ moleculetype ]'):
            insert_itp = i-1
            break

    if insert_itp != -1:
        # 添加注释行和#include指令
        lines.insert(insert_itp, '; Include ligand topology\n')
        lines.insert(insert_itp + 1, f'#include "{ligand_filename}.itp"\n')
        # 找到插入位置后，添加新的分子条目

    # 查找插入位置（在"[ molecules ]"之后）
    insert_pos = -1
    # 从最后一行开始往前查找
    for i in range(len(lines)-1, -1, -1):
        if lines[i].strip().startswith('[ molecules ]'):
            insert_pos = i + 3  # 在[molecules]部分之后插入
            break

    # 找到插入位置后，添加新的分子条目
    if insert_pos != -1:
        # 添加新分子条目
        new_entry = f'{ligand_filename}          1\n'
        lines.insert(insert_pos, new_entry)

    # 写回修改后的内容
    with open('top1.top', 'w') as f:
        f.writelines(lines)
#热浴前规范位置
def modify_topol_top2(topol_file, ligand_filename):
    '''热浴前配体位置规范'''
    # 读取topol.top文件
    with open(topol_file, 'r') as f:
        lines = f.readlines()

    # 查找插入位置（在"[ moleculetype ]"之前）
    insert_itp = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('[ moleculetype ]'):
            insert_itp = i-1
            break

    if insert_itp != -1:
        # 添加注释行和#include指令
        lines.insert(insert_itp, '; Ligand position restraints\n')
        lines.insert(insert_itp + 1, f'#ifdef POSRES\n')
        lines.insert(insert_itp + 2, f'#include "ligand.itp"\n')
        lines.insert(insert_itp + 3, f'#endif\n')
        # 找到插入位置后，添加新的分子条目

    # 写回修改后的内容
    with open('top2.top', 'w') as f:
        f.writelines(lines)


def gmx_part1_workflow(filename):
    '''初始获得分子拓扑'''
    # 调用 combine_gro_files 函数
    combine_gro_files('processed.gro', f'{filename}.gro', 'combined.gro')
    
    # 调用 modify_topol_top 函数修改拓扑文件
    modify_topol_top('topol.top', filename)

def gmx_part2_workflow(filename):
    # 调用 modify_topol_top2 函数修改拓扑文件
    modify_topol_top2('top1.top', filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请提供文件名作为参数")
        sys.exit(1)
    filename = sys.argv[1]
    if sys.argv[1] == 'part1':
        gmx_part1_workflow(sys.argv[2])
    elif sys.argv[1] == 'part2':
        gmx_part2_workflow(sys.argv[2])
