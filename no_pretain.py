# 微调，加载预训练模型
# 用train.csv进行训练，val.csv进行评估，test.csv进行测试
import argparse
import os
from tqdm import trange
import pandas as pd
from rdkit import Chem
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, auc
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from torchvision import transforms
# 忽略警告
from rdkit import RDLogger

from Module.datautils import ValDataset

RDLogger.DisableLog('rdApp.*')

from Module.Model import UMPredict
from Module.datautils import ValDataset, args
import warnings
warnings.filterwarnings("ignore")




def main():
    # 初始化一个列表用于存储roc-auc数据
    train_roc_aucs = []
    val_roc_aucs = []
    test_roc_aucs = []
    if (args.task_name=='BBBP'):
        train_csv_path=("./dataset/BBBP/BBBP_train.csv")
        val_csv_path=('./dataset/BBBP/BBBP_val.csv')
        test_csv_path=('./dataset/BBBP/BBBP_test.csv')

    elif (args.task_name=='bace'):
        train_csv_path=("./dataset/bace/bace_train.csv")
        val_csv_path=('./dataset/bace/bace_val.csv')
        test_csv_path=('./dataset/bace/bace_test.csv')

    elif (args.task_name == 'HIV'):
        train_csv_path=("./dataset/HIV/HIV_train.csv")
        val_csv_path=('./dataset/HIV/HIV_val.csv')
        test_csv_path=('./dataset/HIV/HIV_test.csv')

    elif (args.task_name=='clintox'):
        train_csv_path= "./dataset/clintox/clintox_train.csv"
        val_csv_path= './dataset/clintox/clintox_val.csv'
        test_csv_path= './dataset/clintox/clintox_test.csv'

    elif (args.task_name=='tox21'):
        train_csv_path= "./dataset/tox21/tox21_train.csv"
        val_csv_path= './dataset/tox21/tox21_val.csv'
        test_csv_path= './dataset/tox21/tox21_test.csv'

    valdaset=ValDataset(val_csv_path)
    val_dataloader = DataLoader(valdaset,batch_size=1)
    testdataset=ValDataset(test_csv_path)
    test_dataloader = DataLoader(testdataset,batch_size=1)
    net = UMPredict(31, 6, 0.).to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)


    # model_state_dict = torch.load(args.pretain_pth)
    # net.load_state_dict(model_state_dict)
    net.cnn.load_state_dict(torch.load("./checkpoints/CNN/cnn.pth"))
    train_dataset=ValDataset(train_csv_path)

    # 打印参数量
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    # print("CNN_Params: {}".format(sum(p.numel() for p in meta.learner.net.cnn.parameters() if p.requires_grad)))




    # 训练数据加载器
    train_dataloader = DataLoader(train_dataset,batch_size=20,shuffle=True,num_workers=0)

    # 切换到训练模式
    net.train()

    # 训练循环
    # with trange(iter , desc='Progress', unit='step', ncols=args.epoch) as pbar:
    for epoch in range(args.epoch):  # args.num_epochs 是你想要训练的轮数
    # for epoch in pbar:
        y_train = []
        predictions = []
        print("training:")
        for atom_features, edge_features, padding_mask, img_features, label in tqdm(train_dataloader):

            # 将数据移动到指定设备
            atom_features = atom_features.to(torch.float32).to(args.device)
            edge_features = edge_features.to(torch.float32).to(args.device)
            img_features = img_features.to(torch.float32).to(args.device)
            padding_mask = padding_mask.to(args.device)
            label = label.to(torch.float32).to(args.device)

            # 清除梯度
            optimizer.zero_grad()

            # 前向传播
            loss, predict, target = net(atom_features, edge_features, padding_mask, img_features, label)

            # 计算损失
            loss.backward()

            # 更新参数
            optimizer.step()

            # 收集真实标签和预测值
            y_train.extend(target.flatten().tolist())
            predictions.extend(predict.flatten().tolist())

        # 在每个 epoch 结束时，你可以打印出训练损失和性能指标
        predicted_labels = [1.0 if p >= 0.5 else 0.0 for p in predictions]
        roc_auc = roc_auc_score(y_train, predictions)
        train_roc_aucs.append(roc_auc)  # 将ROC-AUC值添加到列表中
        accuracy = accuracy_score(y_train, predicted_labels)
        f1 = f1_score(y_train, predicted_labels)
        roc_curve_values = roc_curve(y_train, predictions)
        auc_value = auc(roc_curve_values[0], roc_curve_values[1])


        print(f'epoch {epoch+1}/{args.epoch} : Loss: {loss.item():.3f}\t\t',
              f'roc-auc: {roc_auc:.3f}',
              f'\t\taccuracy: {accuracy:.3f}',
              f'\t\tf1: {f1:.3f}',
              f'\t\tauc_value: {auc_value:.3f}')

    # # 训练完成后，记得保存模型
    # torch.save(net.state_dict(), 'model.pth')

# 评估
        net.eval()
        with torch.no_grad():
            y_val=[]
            predictions = []
            for atom_features, edge_features, padding_mask, img_features, label in val_dataloader:

                atom_features=atom_features.to(torch.float32).to(args.device)
                edge_features=edge_features.to(torch.float32).to(args.device)
                img_features=img_features.to(torch.float32).to(args.device)
                padding_mask=padding_mask.to(args.device)
                label=label.to(torch.float32).to(args.device)
                loss,predict,target = net(atom_features, edge_features, padding_mask, img_features,
                    label)
                y_val.extend(target.flatten().tolist())
                predictions.extend(predict.flatten().tolist())

        # Convert predictions to labels
        predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]

        # Calculate the performance metrics
        roc_auc = roc_auc_score(y_val, predictions)

        val_roc_aucs.append(roc_auc)

        accuracy = accuracy_score(y_val, predicted_labels)
        f1 = f1_score(y_val, predicted_labels)
        roc_curve_values = roc_curve(y_val, predictions)
        auc_value = auc(roc_curve_values[0], roc_curve_values[1])

        # print("val:")
        print('val:\t\troc-auc:%.3f' % roc_auc,
              '\t\taccuracy:%.3f' % accuracy,
              '\t\tf1:%.3f' % f1,
              '\t\tauc_value:%.3f' % auc_value)
        # '\t\troc_curve_values: (FPR:', roc_curve_values[0], ', TPR:', roc_curve_values[1], ')',

# 测试
        net.eval()
        with torch.no_grad():
            y_test = []
            predictions = []
            for atom_features, edge_features, padding_mask, img_features, label in test_dataloader:
                atom_features = atom_features.to(torch.float32).to(args.device)
                edge_features = edge_features.to(torch.float32).to(args.device)
                img_features = img_features.to(torch.float32).to(args.device)
                padding_mask = padding_mask.to(args.device)
                label = label.to(torch.float32).to(args.device)
                loss, predict, target = net(atom_features, edge_features, padding_mask, img_features,
                                            label)
                y_test.extend(target.flatten().tolist())
                predictions.extend(predict.flatten().tolist())

        # Convert predictions to labels
        predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]

        # Calculate the performance metrics
        roc_auc = roc_auc_score(y_test, predictions)
        test_roc_aucs.append(roc_auc)
        accuracy = accuracy_score(y_test, predicted_labels)
        f1 = f1_score(y_test, predicted_labels)
        roc_curve_values = roc_curve(y_test, predictions)
        auc_value = auc(roc_curve_values[0], roc_curve_values[1])

        # print("test:")
        print('test:\t\troc-auc:%.3f' % roc_auc,
              '\t\taccuracy:%.3f' % accuracy,
              '\t\tf1:%.3f' % f1,
              '\t\tauc_value:%.3f' % auc_value)
        # '\t\troc_curve_values: (FPR:', roc_curve_values[0], ', TPR:', roc_curve_values[1], ')',

    print("Training ROC-AUC values per epoch:")
    for epoch, roc_auc in enumerate(train_roc_aucs, 1):
        print(f'Epoch {epoch}: {roc_auc:.3f}')
    print("Val ROC-AUC values per epoch:")
    for epoch, roc_auc in enumerate(val_roc_aucs,1):
        print(f'Epoch {epoch}: {roc_auc:.3f}')
    print("Test ROC-AUC values per epoch:")
    for epoch, roc_auc in enumerate(test_roc_aucs,1):
        print(f'Epoch {epoch}: {roc_auc:.3f}')
        
if __name__ == "__main__":
    main()