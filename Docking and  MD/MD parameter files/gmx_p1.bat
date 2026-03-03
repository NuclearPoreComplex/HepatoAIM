@echo off
chcp 65001

echo 获取小分子文件名（不包含扩展名）
for %%f in (*_*_out_ligand_*.gro) do (
    set filename=%%~nf
)
echo 文件名: %filename%

echo 调用Python脚本中的gmx_part1_workflow函数
python topol.py part1 %filename%

echo 调用GROMACS命令进行盒子调整
gmx editconf -f combined.gro -o newbox.gro -c -d 1.2 -bt cubic

echo 调用GROMACS命令进行溶剂化
gmx solvate -cp newbox.gro -cs spc216.gro -o solv.gro -p top1.top

echo 调用GROMACS命令生成能量最小化输入文件
gmx grompp -f em.mdp -c solv.gro -p top1.top -o ions.tpr -maxwarn 1

echo 调用GROMACS命令添加离子
(
echo 15
) | gmx genion -s ions.tpr -o solv_ions.gro -p top1.top -pname SOD -nname CLA -neutral

echo 调用GROMACS命令再次准备能量最小化
gmx grompp -f em.mdp -c solv_ions.gro -p top1.top -o em.tpr

echo 调用GROMACS命令执行能量最小化
gmx mdrun -v -deffnm em

echo 为小分子生成限制拓扑文件
(
echo 0
)|gmx genrestr -f %filename%.gro -o ligand.itp -fc 1000 1000 1000

echo 调用Python脚本中的gmx_part2_workflow函数
python topol.py part2 %filename%

echo 创建索引文件
gmx make_ndx -f em.gro -o index.ndx < make_ndx.txt

echo 准备NVT平衡
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p top2.top -o nvt.tpr -n index.ndx

echo 所有步骤执行完毕！
pause