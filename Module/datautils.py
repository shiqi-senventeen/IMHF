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

from Module import GraphPropagationAttention
from config import args


torch.autograd.set_detect_anomaly(True)






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

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, startidx=0, dim=768, c=3, h=args.imgsize,
                 w=args.imgsize):
        super(SmilesBatch, self).__init__()
        self.batchsz = batchsz
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



        self.task_dict, self.label_dict,self.num_dict = load_csv(os.path.join(root, mode + '.csv'))
        self.data = []
        self.task_label = {}
        for i, (k, v) in enumerate(self.task_dict.items()):
            self.data.append(v)
            self.task_label[k] = i + self.startidx

        self.num_task = len(self.data)
        self.create_batch(self.batchsz)



    def create_batch(self, batchsz):
        """
        Create batches for meta-learning.
        :param batchsz: The size of the batch.
        :return:
        support_x_batch: A list sampled batchsz times, each time sampling n*way, k*shots SMILES molecules, shape: [10000, nway, kshot]
        query_x_batch: A list sampled batchsz times, each time sampling n*way, k, shots SMILES molecules, shape: [10000, nway, k, shot]
        """

        self.support_x_batch = []
        self.query_x_batch = []
        for batch in range(batchsz):
            # select_task indicates from which tasks to extract SMILES molecules.
            select_task = np.random.choice(self.num_task, 1, False)
            np.random.shuffle(select_task)
            support_x = []
            query_x = []

            for task_num in select_task:
                # Selected meta-batch task numbers, for each task, a total of nway*(k shot+kquery) molecules should be sampled
                # Obtain the SMILES list

                smiles_list = self.data[task_num]
                # Initialize two lists
                smiles_list_0 = []
                smiles_list_1 = []
                # Iterate over the SMILES list

                for smiles in smiles_list:
                    # Divide SMILES into two lists based on the value in label_dict
                    if self.label_dict[smiles] == '0':
                        smiles_list_0.append(smiles)
                    elif self.label_dict[smiles] == '1':
                        smiles_list_1.append(smiles)

                # Each task corresponds to two types of SMILES molecules, with labels '0' and '1'
                select_smiles_idx_0 = np.random.choice((len(smiles_list_0)), self.k_shot + self.k_query, False)
                select_smiles_idx_1 = np.random.choice((len(smiles_list_1)), self.k_shot + self.k_query, False)
                np.random.shuffle(select_smiles_idx_0)
                np.random.shuffle(select_smiles_idx_1)
                # Take the first k_shot elements from the selected array as k_shot
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
        :return: support_x=[setz,n_atoms,dim],[setz,c,h,w]  
                support_y=[setz,label]
                query_x=[[setz,n_atoms,dim],[setz,c,h,w]]
                query_y=[setz,label]
        '''

        # support_x_batch[index] n_way*k_shot

        support_smile_batch_list = [item for sublist in self.support_x_batch[index] for item in sublist]

        query_smile_batch_list = [item for sublist in self.query_x_batch[index] for item in sublist]

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

    def __init__(self, smiles):
        self.smiles = smiles
        super().__init__(f"Smiles '{smiles}' The molecules are repeating.")

def SmilesToTensor(smilelist,num_dict):
    '''

 :param smilelist: list of smiles

 :return:[setsz,natoms,dim]
 CNN :[3,224,224]
 '''

    s_img_features = []
    fatoms = []

    tensor_list = []

    for smiles in smilelist:
        mol = Chem.MolFromSmiles(smiles)
        setsz = len(smilelist)


        data=num_dict[smiles]


        if(args.ismetatrain):
            filename=os.path.join("./dataset/IMG/pretain",str(data)+".png")
        else:
            filename=os.path.join(f"./dataset/IMG/{args.task_name}",str(data)+".png")


        transf = transforms.ToTensor()
        IMGfile = cv.imread(filename)

        if IMGfile is None:

            raise Exception("You should set the ismetatrain ")
        img_tensor = transf(IMGfile)
        img_tensor=img_tensor.unsqueeze(0)
        tensor_list.append(img_tensor)

    img_tensor = torch.cat(tensor_list, dim=0)



    atom_list=[]
    edge_list=[]
    padding_list=[]

    for smiles in smilelist:

        atom_num = mol.GetNumAtoms()

        # 1~Max_atom is the atomic level,Max_atom+1~Max_motif is the motif level, and 0 is the molecular level
        super_node=torch.zeros(1,31)
        no_super = torch.zeros(args.Max_atoms, 31)
        motif_node=torch.zeros(args.Max_motif,31)


        for i, atom in enumerate(mol.GetAtoms()):

            no_super[i] = atom_encodding(atom)


        cliques = motif_decomp(mol)


        for k, motif in enumerate(cliques):
                motif_node[k]=torch.ones(1,31)



        num_motif = len(cliques)

        atom_features = torch.cat((super_node, no_super, motif_node), dim=0)


        edge_feature = torch.zeros(args.Max_atoms + args.Max_motif+1 , args.Max_atoms + args.Max_motif+1, 6)


        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            i = a1.GetIdx()
            j = a2.GetIdx()
            bond_feature = bond_encodding(bond)
            edge_feature[i+1][j+1] = bond_feature
            edge_feature[j+1][i+1] = bond_feature



        for k, motif in enumerate(cliques):
            for i in motif:
                edge_feature[args.Max_atoms+k+1][i+1] = torch.tensor([1, 1, 1, 1, 1, 1])  # 回头研究一下怎么对它进行初始化
                edge_feature[i+1][args.Max_atoms+k+1] = torch.tensor([1, 1, 1, 1, 1, 1])

        for i in range(atom_num):
            edge_feature[0][i+1]=torch.tensor([1, 1, 1, 1, 1, 1])
            edge_feature[i+1][0] = torch.tensor([1, 1, 1, 1, 1, 1])

        for j in range(num_motif):
            edge_feature[0][args.Max_atoms+j+1]=torch.tensor([1, 1, 1, 1, 1, 1])
            edge_feature[args.Max_atoms + j+1][0] = torch.tensor([1, 1, 1, 1, 1, 1])


        padding_mask = torch.full((1,args.Max_atoms+args.Max_motif+1,1), True, dtype=torch.bool)
        for i in range(atom_num):
            padding_mask[0][i+1][0]=False
        for i in range(num_motif):
            padding_mask[0][args.Max_atoms+i+1][0]=False
        padding_mask[0][0][0]=False

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


        return task_dict, smiles_dict,num_dict

