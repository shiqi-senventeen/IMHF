import pandas as pd

# 读取CSV文件
df = pd.read_csv('../dataset/tox21_train.csv')

# 确保smiles列的数据唯一性，保留第一次出现的行
df = df.drop_duplicates(subset='smiles', keep='first')

# 保存到新的CSV文件
df.to_csv('../dataset/unique_smiles_file.csv', index=False)
