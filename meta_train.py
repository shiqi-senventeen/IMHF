# -*- coding:utf-8 -*-

import os
import random
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import RDLogger

from Module import UMPredict, SmilesBatch
from Reptile import MetaLearner
from config import args

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(args.random_seed)

    # Prepare training data for pretraining stage
    smi_train = SmilesBatch(
        './dataset',
        mode='pretrain',
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        batchsz=1000
    )

    # Initialize meta-learner
    meta = MetaLearner(
        UMPredict,
        (31, 6, 0.4),
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        meta_batchsz=args.meta_batchsz,
        meta_lr=args.meta_lr,
        num_updates=args.num_updates
    ).to(args.device)

    # Load pretrained model or CNN backbone
    if os.path.exists(args.pretain_pth):
        state_dict = torch.load(args.pretain_pth)
        meta.learner.net.load_state_dict(state_dict)
    else:
        cnn_state_dict = torch.load('./checkpoints/CNN/cnn.pth')
        meta.learner.net.cnn.load_state_dict(cnn_state_dict)

    # Print model parameter count
    total_params = sum(p.numel() for p in meta.learner.net.parameters() if p.requires_grad)
    print(f"Total_params: {total_params}")

    # Initialize metrics
    list_train_roc, list_train_loss = [], []
    list_test_roc, list_test_loss = [], []
    episode = []

    for epoch in range(args.meta_epoch):
        meta.train()
        train_losses, test_losses = [], []

        dataloader = DataLoader(smi_train, batch_size=args.meta_batchsz, shuffle=True)
        loop = tqdm(enumerate(dataloader), total=len(dataloader))

        for step, batch in loop:
            (
                s_atom, s_edge, mask_s, s_img, support_y,
                q_atom, q_edge, mask_q, q_img, query_y
            ) = [x.to(args.device) for x in batch[:10]]

            support_y = support_y.float()
            query_y = query_y.float()

            # Meta-training step
            train_loss, rocs = meta(s_atom, s_edge, mask_s, s_img, support_y,
                                    q_atom, q_edge, mask_q, q_img, query_y)
            train_avg_roc = np.mean(rocs)

            # Evaluation step
            test_out = meta.pred(s_atom, s_edge, mask_s, s_img, support_y,
                                 q_atom, q_edge, mask_q, q_img, query_y)
            test_loss, acc, precision, recall, mcc, roc, f1 = test_out

            # Accumulate loss values
            train_losses.extend([l.item() for l in train_loss])
            test_losses.extend([l.item() for l in test_loss])

            # Log metrics
            train_avg_loss = np.mean(train_losses)
            test_avg_loss = np.mean(test_losses)

            results = [step, train_avg_loss, test_avg_loss, acc, precision, recall, mcc, roc, f1]
            with open(os.path.join(args.save_path, 'log.txt'), 'a') as f:
                f.write(','.join(map(str, results)) + '\n')

            list_train_loss.append(train_avg_loss)
            list_test_loss.append(test_avg_loss)
            list_train_roc.append(train_avg_roc)
            list_test_roc.append(roc)
            episode.append(step)

            # Print current episode summary
            print('\n')
            print(f"Episode: {step}\tTrain ROC: {train_avg_roc:.4f}\tTrain Loss: {train_avg_loss:.4f}"
                  f"\tValid Acc: {acc:.4f}\tValid Loss: {test_avg_loss:.4f}"
                  f"\tPrecision: {precision:.4f}\tRecall: {recall:.4f}"
                  f"\tMCC: {mcc:.4f}\tROC: {roc:.4f}\tF1: {f1:.4f}")


if __name__ == '__main__':
    main()
