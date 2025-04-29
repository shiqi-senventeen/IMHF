# -*-coding:utf-8-*-


import torch


from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from Module import UMPredict
from Module import SmilesBatch
from config import args

from Reptile import  MetaLearner
import random

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

import warnings
warnings.filterwarnings("ignore")


def main():

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.random_seed)


    smi_train = SmilesBatch('./dataset', mode='pretrain', n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query, batchsz=1000)

    meta=MetaLearner(UMPredict, (31, 6, 0.4), n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query, meta_batchsz=args.meta_batchsz, meta_lr=args.meta_lr, num_updates=args.num_updates).to(args.device)

    if os.path.exists(args.pretain_pth):
        model_state_dict = torch.load(args.pretain_pth)
        meta.learner.net.load_state_dict(model_state_dict)
    else:
        cnn_state_dict = torch.load('./checkpoints/CNN/cnn.pth')
        meta.learner.net.cnn.load_state_dict(cnn_state_dict)

    pytorch_total_params = sum(p.numel() for p in meta.learner.net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))


    list_train_roc = []
    list_train_loss = []
    list_test_roc = []
    list_test_loss = []
    episode = []
    for epoch in range(args.meta_epoch):
        meta.train()

        train_losses = []
        test_losses = []

        db = DataLoader(smi_train, batch_size=args.meta_batchsz, shuffle=True, num_workers=0, pin_memory=False)
        loop = tqdm(enumerate(db), total=len(db))

        for step, (s_atom_features, s_edge_features,padding_mask_s, s_img_features,
            support_y,  q_atom_features, q_edge_features, padding_mask_q,q_img_features,
            query_y) in loop:
            s_atom_features=s_atom_features.to(args.device)
            s_edge_features=s_edge_features.to(args.device)
            padding_mask_s=padding_mask_s.to(args.device)
            s_img_features=s_img_features.to(args.device)
            support_y=support_y.to(dtype=torch.float32).to(args.device)
            q_atom_features=q_atom_features.to(args.device)
            q_edge_features=q_edge_features.to(args.device)
            padding_mask_q=padding_mask_q.to(args.device)
            q_img_features=q_img_features.to(args.device)
            query_y=query_y.to(dtype=torch.float32).to(args.device)

            train_loss,rocs= meta(s_atom_features,s_edge_features,padding_mask_s,s_img_features,support_y,q_atom_features,q_edge_features,padding_mask_q,q_img_features,query_y)

            train_avg_roc = np.array(rocs).mean()

            test_loss, test_acc, test_pre_score, test_recall_score, test_mcc_score, test_roc_score, test_f1_score=meta.pred(s_atom_features,s_edge_features,padding_mask_s,s_img_features,support_y,q_atom_features,q_edge_features,padding_mask_q,q_img_features,query_y)

            for i in range(len(test_loss)):
                test_loss1 = test_loss[i].cpu().detach().numpy()
                test_losses.append(test_loss1)

            for i in range(len(train_loss)):
                train_loss1 = train_loss[i].cpu().detach().numpy()
                train_losses.append(train_loss1)

            train_avg_loss = np.array(train_losses).mean()
            test_avg_loss = np.array(test_losses).mean()

            results = [step, train_avg_loss, test_avg_loss, test_acc, test_pre_score, test_recall_score,
                       test_mcc_score, test_roc_score, test_f1_score]

            with open(os.path.join(args.save_path, 'log.txt'), 'a') as f:
                f.write(','.join([str(r) for r in results]))
                f.write('\n')

            list_test_loss.append(test_avg_loss)
            list_test_roc.append(test_roc_score)
            list_train_loss.append(train_avg_loss)
            list_train_roc.append(train_avg_roc)

            print('\n')
            print('episode:', step, '\ttrain roc:%.4f' % train_avg_roc,
                  '\t\ttrain_loss:%.4f' % train_avg_loss,
                  '\t\tvalid acc:%.4f' % test_acc, '\t\tloss:%.4f' % test_avg_loss,
                  '\t\tvalid pre:%.4f' % test_pre_score, '\t\tvalid recall:%.4f' % test_recall_score,
                  '\t\tvalid mcc:%.4f' % test_mcc_score, '\t\tvalid roc:%.4f' % test_roc_score,
                  '\t\tvalid f1:%.4f' % test_f1_score)
            episode.append(step)


if __name__ == '__main__':
    main()