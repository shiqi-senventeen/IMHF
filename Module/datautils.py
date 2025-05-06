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

# Define valid atom elements for molecular representation
ATOM_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']


def onek_encoding_unk(x, allowable_set):
    """
    One-hot encode the input with unknown value handling.

    Args:
        x: Input value to encode
        allowable_set: List of allowable values

    Returns:
        List of binary values (one-hot encoding)
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == item) for item in allowable_set]


def atom_encodding(atom):
    """
    Encode atom features into a tensor representation.

    Args:
        atom: RDKit atom object

    Returns:
        Tensor of atom features including element type, degree, charge, chirality, hybridization, and aromaticity
    """
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ATOM_ELEMENTS)
                        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                        + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
                        + onek_encoding_unk(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'other'])
                        + [atom.GetIsAromatic()])


def bond_encodding(bond):
    """
    Encode bond features into a tensor representation.

    Args:
        bond: RDKit bond object

    Returns:
        Tensor of bond features including bond type, ring membership, and conjugation
    """
    bt = bond.GetBondType()
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(), bond.GetIsConjugated()]
    return torch.Tensor(fbond)


class SmilesBatch(Dataset):
    """
    Dataset class for meta-learning with SMILES data.
    Implements few-shot learning with support and query sets.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, startidx=0, dim=768, c=3, h=args.imgsize,
                 w=args.imgsize):
        super(SmilesBatch, self).__init__()
        self.batchsz = batchsz  # Number of tasks
        self.n_way = n_way  # Number of "ways" (classes)
        self.k_shot = k_shot  # Number of examples per class in support set
        self.k_query = k_query  # Number of examples per class in query set
        self.setsz = self.n_way * self.k_shot
        self.querysz = self.n_way * self.k_query
        self.mode = mode
        self.startidx = startidx
        self.dim = dim  # Feature dimension
        self.c = c  # Number of channels
        self.h = h  # Image height
        self.w = w  # Image width

        # Load data from CSV file
        self.task_dict, self.label_dict, self.num_dict = load_csv(os.path.join(root, mode + '.csv'))
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

        Args:
            batchsz: The size of the batch.

        Returns:
            support_x_batch: A list sampled batchsz times, each time sampling n_way, k_shot SMILES molecules
                            shape: [batchsz, n_way, k_shot]
            query_x_batch: A list sampled batchsz times, each time sampling n_way, k_query SMILES molecules
                          shape: [batchsz, n_way, k_query]
        """
        self.support_x_batch = []
        self.query_x_batch = []
        for batch in range(batchsz):
            # select_task indicates from which tasks to extract SMILES molecules
            select_task = np.random.choice(self.num_task, 1, False)
            np.random.shuffle(select_task)
            support_x = []
            query_x = []

            for task_num in select_task:
                # Selected meta-batch task numbers, for each task, a total of n_way*(k_shot+k_query) molecules should be sampled
                # Obtain the SMILES list
                smiles_list = self.data[task_num]
                # Initialize two lists for binary classification
                smiles_list_0 = []
                smiles_list_1 = []

                # Divide SMILES into two lists based on the label
                for smiles in smiles_list:
                    if self.label_dict[smiles] == '0':
                        smiles_list_0.append(smiles)
                    elif self.label_dict[smiles] == '1':
                        smiles_list_1.append(smiles)

                # Each task corresponds to two types of SMILES molecules, with labels '0' and '1'
                # Randomly sample k_shot+k_query molecules from each class
                select_smiles_idx_0 = np.random.choice((len(smiles_list_0)), self.k_shot + self.k_query, False)
                select_smiles_idx_1 = np.random.choice((len(smiles_list_1)), self.k_shot + self.k_query, False)
                np.random.shuffle(select_smiles_idx_0)
                np.random.shuffle(select_smiles_idx_1)

                # Take the first k_shot elements as support set, rest as query set
                indexDtrain_0 = np.array(select_smiles_idx_0[:self.k_shot])
                indexDtrain_1 = np.array(select_smiles_idx_1[:self.k_shot])
                indexDquery_0 = np.array(select_smiles_idx_0[self.k_shot:])
                indexDquery_1 = np.array(select_smiles_idx_1[self.k_shot:])

                # Combine molecules from both classes for support and query sets
                support_x.append((np.array(smiles_list_0)[indexDtrain_0].tolist()) + (
                    np.array(smiles_list_1)[indexDtrain_1].tolist()))
                query_x.append((np.array(smiles_list_0)[indexDquery_0].tolist()) + (
                    np.array(smiles_list_1)[indexDquery_1].tolist()))

            # Shuffle the order of molecules in support and query sets
            random.shuffle(support_x)
            random.shuffle(query_x)

            # Add to batch
            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)

    def __getitem__(self, index):
        """
        Get a batch of data for meta-learning.

        Args:
            index: Batch index

        Returns:
            s_atom_features: Atom features for support set
            s_edge_features: Edge features for support set
            padding_mask_s: Padding mask for support set
            s_img_features: Image features for support set
            support_y: Labels for support set
            q_atom_features: Atom features for query set
            q_edge_features: Edge features for query set
            padding_mask_q: Padding mask for query set
            q_img_features: Image features for query set
            query_y: Labels for query set
        """
        # Flatten support and query sets
        support_smile_batch_list = [item for sublist in self.support_x_batch[index] for item in sublist]
        query_smile_batch_list = [item for sublist in self.query_x_batch[index] for item in sublist]

        # Convert SMILES to tensors
        s_atom_features, s_edge_features, padding_mask_s, s_img_features = SmilesToTensor(support_smile_batch_list,
                                                                                          self.num_dict)
        q_atom_features, q_edge_features, padding_mask_q, q_img_features = SmilesToTensor(query_smile_batch_list,
                                                                                          self.num_dict)

        # Get labels for support and query sets
        support_y = []
        query_y = []
        for smile in support_smile_batch_list:
            label = self.label_dict[smile]
            support_y.append(float(label))

        for smile in query_smile_batch_list:
            label = self.label_dict[smile]
            query_y.append(float(label))

        return s_atom_features, s_edge_features, padding_mask_s, s_img_features, torch.LongTensor(
            support_y), q_atom_features, q_edge_features, padding_mask_q, q_img_features, torch.LongTensor(
            query_y)

    def __len__(self):
        """Return the number of batches"""
        return self.batchsz


class ValDataset(Dataset):
    """
    Dataset class for validation/testing.
    """

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with smiles and labels.
        """
        self.smiles_data = pd.read_csv(csv_file)

    def __len__(self):
        """Return the number of samples"""
        return len(self.smiles_data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            atom_features: Atom features
            edge_features: Edge features
            padding_mask: Padding mask
            img_features: Image features
            label: Classification label
        """
        num_dict = {}
        smiles = self.smiles_data.iloc[idx, 1]
        label = self.smiles_data.iloc[idx, 2]
        num = self.smiles_data.iloc[idx, 3]
        num_dict[smiles] = num
        smiles_list = [smiles]

        # Convert SMILES to tensors
        atom_features, edge_features, padding_mask, img_features = SmilesToTensor(smiles_list, num_dict=num_dict)

        # Remove batch dimension
        atom_features = atom_features.squeeze(0)
        edge_features = edge_features.squeeze(0)
        padding_mask = padding_mask.squeeze(0)
        img_features = img_features.squeeze(0)

        return atom_features, edge_features, padding_mask, img_features, torch.tensor(label, dtype=torch.float32)


class SmilesDuplicationError(Exception):
    """Exception raised when duplicate SMILES are found in the dataset."""

    def __init__(self, smiles):
        self.smiles = smiles
        super().__init__(f"Smiles '{smiles}' The molecules are repeating.")


def SmilesToTensor(smilelist, num_dict):
    """
    Convert a list of SMILES strings to tensor representations for network input.

    Args:
        smilelist: List of SMILES strings
        num_dict: Dictionary mapping SMILES to numerical identifiers

    Returns:
        atom_features: Tensor of atom features [batch_size, max_atoms+max_motif+1, feature_dim]
        edge_feature: Tensor of edge features [batch_size, max_atoms+max_motif+1, max_atoms+max_motif+1, edge_dim]
        padding_mask: Tensor indicating which atoms/motifs are padding [batch_size, max_atoms+max_motif+1, 1]
        img_tensor: Tensor of molecular images [batch_size, channels, height, width]
    """
    s_img_features = []
    fatoms = []

    tensor_list = []

    # Process molecular images
    for smiles in smilelist:
        mol = Chem.MolFromSmiles(smiles)
        setsz = len(smilelist)

        data = num_dict[smiles]

        # Load molecular image based on training mode
        if (args.ismetatrain):
            filename = os.path.join("./dataset/IMG/pretain", str(data) + ".png")
        else:
            filename = os.path.join(f"./dataset/IMG/{args.task_name}", str(data) + ".png")

        transf = transforms.ToTensor()
        IMGfile = cv.imread(filename)

        if IMGfile is None:
            raise Exception("You should set the ismetatrain ")

        img_tensor = transf(IMGfile)
        img_tensor = img_tensor.unsqueeze(0)
        tensor_list.append(img_tensor)

    # Concatenate image tensors
    img_tensor = torch.cat(tensor_list, dim=0)

    # Process molecular graphs
    atom_list = []
    edge_list = []
    padding_list = []

    for smiles in smilelist:
        mol = Chem.MolFromSmiles(smiles)
        atom_num = mol.GetNumAtoms()

        # Initialize node features:
        # 1~Max_atom is the atomic level, Max_atom+1~Max_motif is the motif level, and 0 is the molecular level
        super_node = torch.zeros(1, 31)  # Molecular-level node (index 0)
        no_super = torch.zeros(args.Max_atoms, 31)  # Atom-level nodes
        motif_node = torch.zeros(args.Max_motif, 31)  # Motif-level nodes

        # Encode atom features
        for i, atom in enumerate(mol.GetAtoms()):
            no_super[i] = atom_encodding(atom)

        # Decompose molecule into motifs (functional groups)
        cliques = motif_decomp(mol)

        # Set motif node features
        for k, motif in enumerate(cliques):
            motif_node[k] = torch.ones(1, 31)

        num_motif = len(cliques)

        # Combine all node features
        atom_features = torch.cat((super_node, no_super, motif_node), dim=0)

        # Initialize edge features tensor
        edge_feature = torch.zeros(args.Max_atoms + args.Max_motif + 1, args.Max_atoms + args.Max_motif + 1, 6)

        # Encode bond features for atom-atom connections
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            i = a1.GetIdx()
            j = a2.GetIdx()
            bond_feature = bond_encodding(bond)
            edge_feature[i + 1][j + 1] = bond_feature
            edge_feature[j + 1][i + 1] = bond_feature  # Symmetric connection

        # Connect motifs to their atoms
        for k, motif in enumerate(cliques):
            for i in motif:
                edge_feature[args.Max_atoms + k + 1][i + 1] = torch.tensor([1, 1, 1, 1, 1, 1])
                edge_feature[i + 1][args.Max_atoms + k + 1] = torch.tensor([1, 1, 1, 1, 1, 1])

        # Connect super node (molecule node) to atoms
        for i in range(atom_num):
            edge_feature[0][i + 1] = torch.tensor([1, 1, 1, 1, 1, 1])
            edge_feature[i + 1][0] = torch.tensor([1, 1, 1, 1, 1, 1])

        # Connect super node (molecule node) to motifs
        for j in range(num_motif):
            edge_feature[0][args.Max_atoms + j + 1] = torch.tensor([1, 1, 1, 1, 1, 1])
            edge_feature[args.Max_atoms + j + 1][0] = torch.tensor([1, 1, 1, 1, 1, 1])

        # Create padding mask (True for padding positions, False for actual nodes)
        padding_mask = torch.full((1, args.Max_atoms + args.Max_motif + 1, 1), True, dtype=torch.bool)
        for i in range(atom_num):
            padding_mask[0][i + 1][0] = False  # Set actual atoms as non-padding
        for i in range(num_motif):
            padding_mask[0][args.Max_atoms + i + 1][0] = False  # Set actual motifs as non-padding
        padding_mask[0][0][0] = False  # Super node is not padding

        # Add batch dimension and append to lists
        atom_list.append(atom_features.unsqueeze(0))
        edge_list.append(edge_feature.unsqueeze(0))
        padding_list.append(padding_mask)

    # Concatenate tensors along batch dimension
    atom_features = torch.cat(atom_list, dim=0)
    edge_feature = torch.cat(edge_list, dim=0)
    padding_mask = torch.cat(padding_list, dim=0)

    return atom_features, edge_feature, padding_mask, img_tensor


def motif_decomp(mol):
    """
    Decompose a molecule into motifs (functional groups).
    Uses BRICS decomposition algorithm from RDKit.

    Args:
        mol: RDKit molecule object

    Returns:
        cliques: List of lists, where each inner list contains atom indices for a motif
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    # Initialize with all bonds as cliques
    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    # Find breaking points using BRICS algorithm
    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # Merge overlapping cliques
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

    # Filter out empty cliques and those equal to the whole molecule
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    # Handle ring structures
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

    # Filter final cliques
    cliques = [c for c in cliques if n_atoms > len(c) > 0]
    return cliques


def copy_edit_mol(mol):
    """
    Create an editable copy of a molecule.

    Args:
        mol: RDKit molecule object

    Returns:
        new_mol: Editable copy of the molecule
    """
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
    """
    Get the fragment molecule corresponding to a clique.

    Args:
        mol: RDKit molecule object
        atoms: List of atom indices

    Returns:
        new_mol: RDKit molecule object for the fragment
    """
    # Get the fragment of clique
    Chem.Kekulize(mol)
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    Chem.SanitizeMol(mol)
    return new_mol


def copy_atom(atom):
    """
    Create a copy of an atom.

    Args:
        atom: RDKit atom object

    Returns:
        new_atom: Copy of the atom
    """
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def sanitize(mol):
    """
    Sanitize a molecule for validity.

    Args:
        mol: RDKit molecule object

    Returns:
        mol: Sanitized molecule, or None if invalid
    """
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def get_smiles(mol):
    """
    Get SMILES string from molecule.

    Args:
        mol: RDKit molecule object

    Returns:
        SMILES string representation
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def get_mol(smiles):
    """
    Get molecule from SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        mol: RDKit molecule object, or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


def save_model(model: nn.Module, save_path: str, k_shot, meta_batchsz):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        save_path: Directory to save model
        k_shot: Number of shots in meta-learning
        meta_batchsz: Batch size for meta-learning
    """
    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    name = f"{now}_meta_learner"
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f"{name}_state_dict.pth"),
    )


def load_csv(csv_path):
    """
    Load data from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        task_dict: Dictionary mapping task names to lists of SMILES
        smiles_dict: Dictionary mapping SMILES to labels
        num_dict: Dictionary mapping SMILES to numerical identifiers
    """
    task_dict = {}
    smiles_dict = {}
    num_dict = {}
    with open(csv_path) as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',')
        next(csvreader, None)  # Skip header row

        for i, row in enumerate(csvreader):
            task_name = row[0]
            smiles = row[1]
            label = row[2]
            num = row[3]

            # Add SMILES to task dictionary
            if task_name in task_dict.keys():
                task_dict[task_name].append(smiles)
            else:
                task_dict[task_name] = [smiles]

            # Check for duplicate SMILES
            if smiles not in smiles_dict.keys():
                smiles_dict[smiles] = label
            else:
                raise SmilesDuplicationError(smiles)

            num_dict[smiles] = num

    return task_dict, smiles_dict, num_dict