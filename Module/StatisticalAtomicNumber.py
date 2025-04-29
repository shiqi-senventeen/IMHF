# -*-coding:utf-8-*-

import csv
from rdkit import Chem

filename='../dataset/train.csv'
# 初始化最大原子数为0
max_atoms = 0

# 读取CSV文件
with open(filename, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 尝试将smiles字符串转换为Mol对象
        mol = Chem.MolFromSmiles(row['smiles'])

        # 如果转换成功
        if mol is not None:
            # 获取原子数并更新最大原子数
            atoms = sum(1 for atom in mol.GetAtoms())
            max_atoms = max(max_atoms, atoms)
        else:
            # 如果转换失败，输出错误信息
            print(f"Error processing smiles: {row['smiles']}")

# 输出最大的原子数
print(f"The maximum number of atoms is: {max_atoms}")

import csv
from rdkit import Chem
import matplotlib.pyplot as plt
from collections import Counter

# 初始化原子数 Counter
atom_counts = Counter()

# 读取CSV文件
with open(filename, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 尝试将smiles字符串转换为Mol对象
        mol = Chem.MolFromSmiles(row['smiles'])

        # 如果转换成功
        if mol is not None:
            # 获取原子数
            atoms = mol.GetNumAtoms()
            atom_counts[atoms] += 1

# 转换为列表
atom_counts_list = list(atom_counts.items())
# 将列表转换为字典，键为原子序号，值为原子计数
atom_numbers = [x[0] for x in atom_counts_list]
frequencies = [x[1] for x in atom_counts_list]
# 绘制直方图
plt.bar(atom_numbers, frequencies, color='skyblue')
# 设置图表标题和坐标轴标签
plt.title('The number of atoms in the dataset')
plt.xlabel('Number of atoms')
plt.ylabel('frequency')
# 显示图表
plt.show()