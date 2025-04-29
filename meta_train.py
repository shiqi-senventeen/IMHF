# -*-coding:utf-8-*-
import argparse
import datetime

import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, auc

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from Module.Model import UMPredict
from Module.datautils import SmilesBatch, args
# 保证打印能看的完整
from Reptile.meta import Learner, MetaLearner
import random


# 忽略警告
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

import warnings
warnings.filterwarnings("ignore")


def main():


    # 设置随机数种子
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

    set_seed(args.random_seed)


    smi_train = SmilesBatch('./dataset', mode='pretrain', n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query, batchsz=1000)

    meta=MetaLearner(UMPredict, (31, 6, 0.4), n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query, meta_batchsz=args.meta_batchsz, meta_lr=args.meta_lr, num_updates=args.num_updates).to(args.device)

    print(meta)

    # 如果预训练就加载元模型参数，否则加载冻结的CNN参数
    if(args.pretain):
        model_state_dict = torch.load('checkpoints/pretain.pth')
        meta.learner.net.load_state_dict(model_state_dict)
    else:
        cnn_state_dict = torch.load('./checkpoints/CNN/cnn.pth')
        meta.learner.net.cnn.load_state_dict(cnn_state_dict)
    '''
    这两个模型参数必须选一个load
    '''
    # # # # '''
    # # # 加载CNN部分模型参数
    # # # '''
    # cnn_state_dict = torch.load('./checkpoints/CNN/SETR_ConvFormer_04121402_16_0.014175541458966445.pth')
    # meta.learner.net.cnn.load_state_dict(cnn_state_dict)
    # # # #
    # # 注意参数 Node dim=512
    # 预训练加载模型
    # model_state_dict = torch.load('./checkpoints/pretain.pth')
    # meta.learner.net.load_state_dict(model_state_dict)


    # 打印参数量
    pytorch_total_params = sum(p.numel() for p in meta.learner.net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    # print("CNN_Params: {}".format(sum(p.numel() for p in meta.learner.net.cnn.parameters() if p.requires_grad)))





    list_train_roc = []
    list_train_loss = []
    list_test_roc = []
    list_test_loss = []
    episode = []
    for epoch in range(args.meta_epoch):
        meta.train()

        train_losses = []
        test_losses = []
    # for epoch in range(2):
        # fetch meta_batchsz num of episode each time

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
            # num=num+1
            # print(f"采样了{num}次")
            # print(s_atom_features.shape, s_edge_features, padding_mask_s, s_img_features, support_y, q_atom_features, q_edge_features, padding_mask_q, q_img_features, query_y)


            # 传出两部分的loss,分别是CNN部分的重建损失和GNN部分的损失
            # cnn_loss,gnn_loss=meta(s_atom_features,s_edge_features,padding_mask_s,s_img_features,support_y,q_atom_features,q_edge_features,padding_mask_q,q_img_features,query_y)
            train_loss,rocs=meta(s_atom_features,s_edge_features,padding_mask_s,s_img_features,support_y,q_atom_features,q_edge_features,padding_mask_q,q_img_features,query_y)

            # # 截取roc的长度，保留两位小数
            # roc=[f"{x:.2f}" for x in roc]

            # 记录在所有采样任务上的roc的平均值
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
            print('episode:', step, '\ttrain roc:%.6f' % train_avg_roc,
                  '\t\ttrain_loss:%.6f' % train_avg_loss,
                  '\t\tvalid acc:%.6f' % test_acc, '\t\tloss:%.6f' % test_avg_loss,
                  '\t\tvalid pre:%.6f' % test_pre_score, '\t\tvalid recall:%.6f' % test_recall_score,
                  '\t\tvalid mcc:%.6f' % test_mcc_score, '\t\tvalid roc:%.6f' % test_roc_score,
                  '\t\tvalid f1:%.6f' % test_f1_score)
            episode.append(step)
            '''
            保存CNN部分的权重
            # torch.save(meta.learner.net.cnn.state_dict(),save_path)
            
            保存最终元模型的权重只需要torch.save(meta.learner.net.state_dict(),save_path)
            '''


        # 假设 meta 和 args 已经被定义
        save_path = args.save_path

        # 获取当前时间
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 将时间插入到文件名中
        save_path_with_time = save_path + '/' + current_time + '.pth'
        torch.save(meta.learner.net.state_dict(), save_path_with_time)
        print('save final meta-learner')







if __name__ == '__main__':
    main()