import csv

from rdkit import Chem

import csv
from rdkit import Chem

from Module.datautils import motif_decomp

# 打开CSV文件进行阅读
with open('./dataset/bace_val.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    # 可能需要创建一个新的CSV文件来保存过滤后的行
    with open('./dataset/bace_val_del.csv', 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                # 尝试将SMILES字符串转换为Mol对象
                mol = Chem.MolFromSmiles(row['smiles'])

                # 如果转换成功
                if mol is not None:
                    # 获取分子中的原子数
                    atoms = mol.GetNumAtoms()
                    cliques = motif_decomp(mol)
                    num_motif = len(cliques)

                    # 如果原子数小于等于60，保留这一行
                    if atoms < 60 and num_motif <20:
                        writer.writerow(row)
                    else:
                        # 如果原子数大于60，可以在这里选择删除或者保留这一行
                        # writer.writerow(row)  # 如果保留，取消注释这行
                        print(f"原子数超过60: {row['smiles']}")
                else:
                    # 如果转换失败，输出错误信息
                    print(f"Error processing smiles: {row['smiles']}")
            except Exception as e:
                # 捕获并输出可能发生的任何异常
                print(f"处理过程中发生异常： {row['smiles']} - {str(e)}")
