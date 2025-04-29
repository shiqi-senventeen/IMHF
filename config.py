import torch


class args:
    meta_batchsz =3   #2    #15每次采样的任务数
    meta_lr = 0.001
    num_updates = 5 #　元学习支持集的更新次数
    random_seed = 68
    gnn_num_layer=3 # 图神经网络的更新次数
    p_dropout = 0.2

    imgsize=224
    # 原子和键编码维度
    en_node_dim=31
    en_edge_dim=6


    # 原子和键的潜在维度
    node_dim=256   # 512
    edge_dim=64

    # 图像特征潜在维度
    img_dim=768
    img_channels=3

    # radius = 3

    # 二分类
    per_task_output_units_num = 2  # 每个任务都是二分类任务

    epoch = 100
    meta_epoch=10

    n_way = 2   # 2分类任务
    k_shot = 3  # 支持集每类样本采样3个分子
    k_query = 5     # 查询集每类样本采样5个分子
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Max_atoms = 60  # 允许的最大原子数
    Max_motif = 20  # 允许的最大基序数


    save_path = "./checkpoints"
    pretain_pth='./checkpoints/pretain.pth'
    #每次需要修改
    # 预训练过程中pretain=True,finetuning阶段设置pretain=False
    pretain=False
    task_name="HIV"   # BBBP,tox21,HIV
