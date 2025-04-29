# -*-coding:utf-8-*-
import csv
import itertools
import os
import random
import time
from collections import Counter

import cv2 as cv

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from rdkit.Chem import BRICS

from Module.GNN import GraphPropagationAttention
from config import args

#可以在 python 文件头部使用如下函数打开 nan 检查：
torch.autograd.set_detect_anomaly(True)

# # 保证打印能看的完整
# torch.set_printoptions(threshold=np.inf)




ATOM_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == item) for item in allowable_set]


def atom_encodding(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ATOM_ELEMENTS)
                        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                        + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
                        + onek_encoding_unk(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'other'])
                        + [atom.GetIsAromatic()])


def bond_encodding(bond):
    bt = bond.GetBondType()
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(), bond.GetIsConjugated()]
    return torch.Tensor(fbond)


class SmilesBatch(Dataset):
    '''
    传入dataloader的dataset
    __getitem__方法返回support_x,support_y,query_x,query_y
    file应该是一整个csv表格,把所有的smiles分子及对应的label放在csv文件里面

    path:dataset整个的文件夹
     root :
        |- images/*.jpg 包括所有的分子图像
        |- pretrain.csv
        |- test.csv
        |- val.csv

    train/test/val.csv 应该有列task_label区分不同的任务

    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    '''

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, startidx=0, dim=768, c=3, h=args.imgsize,
                 w=args.imgsize):
        super(SmilesBatch, self).__init__()
        self.batchsz = batchsz  # 设置的采样批次，注意与args.batchsz的区别，那个是每次采样的任务数
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.setsz = self.n_way * self.k_shot
        self.querysz = self.n_way * self.k_query
        self.mode = mode
        self.startidx = startidx
        # self.n_atoms = n_atoms
        self.dim = dim
        self.c = c
        self.h = h
        self.w = w

        # 分子图像文件
        # self.image_path = os.path.join(root, 'images')
        '''
        task_dict应该是{task1:[smiles1...]...}
        label_dict应该是{smiles1:0,smile2:0...}
        '''
        self.task_dict, self.label_dict,self.num_dict = load_csv(os.path.join(root, mode + '.csv'))
        # 数据
        '''
        data:list[[smile1.....smile100],[smile101...smile200]...]   包括所有的smiles分子
        '''
        self.data = []
        self.task_label = {}

        '''
        task_label{"tox21":0,...}
        每个任务都对应了一个数
        '''
        for i, (k, v) in enumerate(self.task_dict.items()):
            self.data.append(v)  # 二维列表[[smile1, smile2, ...], [smile111, ...]]
            self.task_label[k] = i + self.startidx

        self.num_task = len(self.data)  # 返回的是列表最外层的长度，即任务数
        self.create_batch(self.batchsz)



    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        :param batchsz:
        :return:
        support_x_batch:创建一个列表，重复采样batchsz次，每次采样出n*way,k*shots个smiles分子,shape:[10000,nway,kshot]
        query_x_batch:创建一个列表，...,shape:[10000,nway,k,shot]
        """
        self.support_x_batch = []
        self.query_x_batch = []
        for batch in range(batchsz):
            '''
            在所有的分子中不放回的抽取n*way,k*shot个smiles分子
            '''
            # select_task表示从哪些任务中抽取smiles分子
            select_task = np.random.choice(self.num_task, 1, False)
            np.random.shuffle(select_task)
            support_x = []
            query_x = []

            for task_num in select_task:
                # 选取了meta-batch个任务编号，对于每个任务，应该一共采样nway*(k shot+kquery)个分子
                # 获取SMILES列表
                smiles_list = self.data[task_num]
                # 初始化两个列表
                smiles_list_0 = []
                smiles_list_1 = []
                # 遍历SMILES列表
                for smiles in smiles_list:
                    # 根据label_dict中的值将SMILES分到两个列表
                    if self.label_dict[smiles] == '0':
                        smiles_list_0.append(smiles)
                    elif self.label_dict[smiles] == '1':
                        smiles_list_1.append(smiles)

                # 每个任务对应的smiles分子分成两类，label为0和label为1
                # select_smiles_idx = np.random.choice((len(self.data[task_num])), self.k_shot + self.k_query, False)
                select_smiles_idx_0 = np.random.choice((len(smiles_list_0)), self.k_shot + self.k_query, False)
                select_smiles_idx_1 = np.random.choice((len(smiles_list_1)), self.k_shot + self.k_query, False)
                np.random.shuffle(select_smiles_idx_0)
                np.random.shuffle(select_smiles_idx_1)
                #   把刚才选取的数组第0~k_shot-1截取出来作为k_shot
                indexDtrain_0 = np.array(select_smiles_idx_0[:self.k_shot])
                indexDtrain_1 =np.array(select_smiles_idx_1[:self.k_shot])
                indexDquery_0 = np.array(select_smiles_idx_0[self.k_shot:])
                indexDquery_1 = np.array(select_smiles_idx_1[self.k_shot:])
                support_x.append((np.array(smiles_list_0)[indexDtrain_0].tolist())+(np.array(smiles_list_1)[indexDtrain_1].tolist()))
                # support_x.append(np.array(smiles_list)[indexDtrain_1].tolist())
                query_x.append((np.array(smiles_list_0)[indexDquery_0].tolist())+(np.array(smiles_list_1)[indexDquery_1].tolist()))
                # query_x.append(np.array(smiles_list)[indexDquery_1].tolist())

            # support_x:[smiles24,smiles31,....]
            random.shuffle(support_x)
            random.shuffle(query_x)

            # support_x_batch:[bz,nway,kshot]
            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)

        # print(np.array(self.support_x_batch).shape)
        # print(np.array(self.query_x_batch).shape)



    def __getitem__(self, index):
        '''

        :param index:
        :return: support_x=[setz,n_atoms,dim],[setz,c,h,w]  传两个张量
                support_y=[setz,label]
                query_x=[[setz,n_atoms,dim],[setz,c,h,w]]
                query_y=[setz,label]
        '''

        # support_x_batch[index] n_way*k_shot
        # 一次训练的smiles分子打包成list
        support_smile_batch_list = [item for sublist in self.support_x_batch[index] for item in sublist]
        # print(smile_batch_list):
        # ['CC(=O)OCCCC1:C:C:C:C:C:1', 'O=C(Cl)C1:C:C:C:C(C(=O)Cl):C:1', 'O=C(C1:C:C:C:C:C:1)C(O)C1:C:C:C:C:C:1','CNC(=O)C1:C:C:C:C:C:1', 'CCCCNC(N)=O', 'S=C=NCC1:C:C:C:C:C:1']
        query_smile_batch_list = [item for sublist in self.query_x_batch[index] for item in sublist]

        #   对support_x进行处理

        s_atom_features, s_edge_features, padding_mask_s , s_img_features = SmilesToTensor(support_smile_batch_list,self.num_dict)
        q_atom_features, q_edge_features, padding_mask_q, q_img_features= SmilesToTensor(query_smile_batch_list,self.num_dict)

        support_y = []
        query_y = []
        for smile in support_smile_batch_list:
            label = self.label_dict[smile]
            support_y.append(float(label))

        for smile in query_smile_batch_list:
            label = self.label_dict[smile]
            query_y.append(float(label))

        #
        return s_atom_features, s_edge_features,padding_mask_s, s_img_features, torch.LongTensor(
            support_y),  q_atom_features, q_edge_features, padding_mask_q,q_img_features, torch.LongTensor(
            query_y)

    def __len__(self):
        return self.batchsz


class ValDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with smiles and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.smiles_data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        num_dict = {}
        smiles = self.smiles_data.iloc[idx, 1]
        label = self.smiles_data.iloc[idx, 2]
        num = self.smiles_data.iloc[idx, 3]
        num_dict[smiles] = num
        smiles_list = [smiles]
        atom_features, edge_features, padding_mask, img_features = SmilesToTensor(smiles_list,
                                                                                          num_dict=num_dict)
        atom_features = atom_features.squeeze(0)
        edge_features = edge_features.squeeze(0)
        padding_mask = padding_mask.squeeze(0)
        img_features = img_features.squeeze(0)

        return atom_features, edge_features, padding_mask, img_features, torch.tensor(label, dtype=torch.float32)

class SmilesDuplicationError(Exception):
    """自定义的Smiles重复错误"""

    def __init__(self, smiles):
        self.smiles = smiles
        super().__init__(f"Smiles '{smiles}' 分子发生了重复。")


# 传入一个smiles列表,将他们转换成张量，具体有输入图神经的embedding和输入卷积神经网络的embedding
def SmilesToTensor(smilelist,num_dict):
    '''

 :param smilelist: 传入一个smiles列表，对smiles进行编码，返回元模型需要的张量

 :return:[setsz,natoms,dim]
 CNN :[3,224,224]
 '''

    s_img_features = []
    fatoms = []

    tensor_list = []

    # 生成分子图像特征
    for smiles in smilelist:
        mol = Chem.MolFromSmiles(smiles)
        setsz = len(smilelist)

        # with open(r"D:\Pyprograms\smiles-main\dataset\pretrain.csv", mode='r', encoding='utf-8') as csvfile:
        #     # 创建CSV阅读器
        #     reader = csv.DictReader(csvfile)
        #     # 遍历CSV文件中的每一行
        #     for row in reader:
        #         # 比较smiles列的值
        #         if row['smiles'] == smiles:
        #             # 找到了匹配项，记录code_num
        #             data = row['code_num']
        #             # 由于是最后一列，找到一个匹配项即可停止搜索
        #             break
        data=num_dict[smiles]

        # 根据任务更换文件夹名
        # 如果是预训练的话应该从下面这行拿到图片
        if(args.pretain):
            filename=os.path.join("./dataset/IMG/pretain",str(data)+".png")
        else:
            filename=os.path.join(f"./dataset/IMG/{args.task_name}",str(data)+".png")
        # 否则应该从这个地方拿图片

        transf = transforms.ToTensor()
        IMGfile = cv.imread(filename)
        # cv.imshow('Image Title', IMGfile)
        img_tensor = transf(IMGfile)
        img_tensor=img_tensor.unsqueeze(0)
        tensor_list.append(img_tensor)

    img_tensor = torch.cat(tensor_list, dim=0)
    # print(img_tensor.shape)  # [6,3,224,224]


# 分别创建[1,31][Max_atoms,31],[Max_motif,31]的张量分别作为分子层特征，原子层特征和基序层特征
# 内层concat成[Max_atoms+Max_motif+1,31]的张量


    atom_list=[]
    edge_list=[]
    padding_list=[]
    # 接下来对图神经网络部分进行编码
    for smiles in smilelist:
        # 获取原子数
        atom_num = mol.GetNumAtoms()
        # print("atom"+str(atom_num))

        # 1~Max_atom为原子层次,Max_atom+1~Max_motif为基序层次，0为分子层次
        super_node=torch.zeros(1,31)
        no_super = torch.zeros(args.Max_atoms, 31)
        motif_node=torch.zeros(args.Max_motif,31)


        for i, atom in enumerate(mol.GetAtoms()):
            # 第一步，对每个原子进行onehot编码，生成Max_atoms*31的特征矩阵
            # print(smiles)
            no_super[i] = atom_encodding(atom)


        cliques = motif_decomp(mol)
        # 分割基序
        # print(cliques)  # [[0, 1, 2], [4, 5, 6], [7, 8, 9, 10, 11, 12], [3]]

        # 第二步，对基序进行编码,初始化为1
        for k, motif in enumerate(cliques):
                motif_node[k]=torch.ones(1,31)



        num_motif = len(cliques)
        # print("motif"+str(num_motif))

        atom_features = torch.cat((super_node, no_super, motif_node), dim=0)


        # 接下来创建邻接矩阵，大小为(Max_atoms+Max_motif1)*(Max_atoms+Max_motif+1)*6

        # 先对原子层次键进行生成
        edge_feature = torch.zeros(args.Max_atoms + args.Max_motif+1 , args.Max_atoms + args.Max_motif+1, 6)

        # 首先是原子与原子之间
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            i = a1.GetIdx()
            j = a2.GetIdx()
            bond_feature = bond_encodding(bond)
            edge_feature[i+1][j+1] = bond_feature
            edge_feature[j+1][i+1] = bond_feature

        # 再对基序层次与原子直接连接

        for k, motif in enumerate(cliques):
            for i in motif:
                edge_feature[args.Max_atoms+k+1][i+1] = torch.tensor([1, 1, 1, 1, 1, 1])  # 回头研究一下怎么对它进行初始化
                edge_feature[i+1][args.Max_atoms+k+1] = torch.tensor([1, 1, 1, 1, 1, 1])

        # 再与分子层次进行连接
        for i in range(atom_num):
            edge_feature[0][i+1]=torch.tensor([1, 1, 1, 1, 1, 1])
            edge_feature[i+1][0] = torch.tensor([1, 1, 1, 1, 1, 1])

        for j in range(num_motif):
            edge_feature[0][args.Max_atoms+j+1]=torch.tensor([1, 1, 1, 1, 1, 1])
            edge_feature[args.Max_atoms + j+1][0] = torch.tensor([1, 1, 1, 1, 1, 1])

        # print(atom_features.size())
        # print(edge_feature.size())

        # 接下来生成padding_mask矩阵
        padding_mask = torch.full((1,args.Max_atoms+args.Max_motif+1,1), True, dtype=torch.bool)
        for i in range(atom_num):
            padding_mask[0][i+1][0]=False
        for i in range(num_motif):
            padding_mask[0][args.Max_atoms+i+1][0]=False
        padding_mask[0][0][0]=False
        # print(padding_mask)

        atom_list.append(atom_features.unsqueeze(0))
        edge_list.append(edge_feature.unsqueeze(0))
        padding_list.append(padding_mask)

    atom_features = torch.cat(atom_list, dim=0)
    edge_feature=torch.cat(edge_list,dim=0)
    padding_mask=torch.cat(padding_list,dim=0)

    return atom_features,edge_feature,padding_mask,img_tensor


def motif_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr) > 1:
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
            cliques[i] = []

    cliques = [c for c in cliques if n_atoms > len(c) > 0]
    return cliques


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_clique_mol(mol, atoms):
    # get the fragment of clique
    Chem.Kekulize(mol)
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    Chem.SanitizeMol(mol)
    return new_mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Chem.Kekulize(mol)
    return mol


def save_model(model: nn.Module, save_path: str, k_shot, meta_batchsz):
    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    name = f"{now}_meta_learner"
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f"{name}_state_dict.pth"),
    )


def load_csv(csv_path):
        """
        处理csv文件：
        格式：task_label,smiles,label
        return a dict saving the information of csv
        : param splitFile: csv file name
        : return: {task_label_1:[[smile_1,label]，[smile_2,label]...]}
    返回两个dict
    分别是task对应的smiles
    和smiles对应的label
        """
        task_dict = {}
        smiles_dict = {}
        num_dict={}
        with open(csv_path) as csv_file:
            csvreader = csv.reader(csv_file, delimiter=',')
            next(csvreader, None)

            for i, row in enumerate(csvreader):
                # if(i==872):
                #     print("1")
                task_name = row[0]
                smiles = row[1]
                label = row[2]
                num = row[3]

                if task_name in task_dict.keys():
                    task_dict[task_name].append(smiles)
                else:
                    task_dict[task_name] = [smiles]
                if smiles not in smiles_dict.keys():
                    smiles_dict[smiles] = label
                else:
                    raise SmilesDuplicationError(smiles)
                num_dict[smiles]=num

        # smiles_dict:{'CN(C)C(=O)OC1:C:C:C:[N+](C):C:1': '1'}
        # task_dict:{'NR-AR': ['CN(C)C(=O)OC1:C:C:C:[N+](C):C:1']}
        # 这里再返回一个编号的字典
        return task_dict, smiles_dict,num_dict

