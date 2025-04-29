import torch


class args:
    meta_batchsz = 2  # 2 Number of tasks sampled each time
    meta_lr = 0.001
    num_updates = 5  # Number of updates for the meta-learning support set
    random_seed = 68
    gnn_num_layer = 3  # Number of updates in the graph neural network
    p_dropout = 0.2

    imgsize = 224
    # Atom and bond encoding dimensions
    en_node_dim = 31
    en_edge_dim = 6

    # Latent dimensions for atoms and bonds
    node_dim = 256  # 512
    edge_dim = 64

    # Latent dimensions for image features
    img_dim = 768
    img_channels = 3

    # radius = 3

    # Binary classification
    per_task_output_units_num = 2  # Each task is a binary classification task

    epoch = 100
    meta_epoch = 10

    n_way = 2  # 2-way classification task
    k_shot = 2  # Sample 3 molecules per class for the support set
    k_query = 5  # Sample 5 molecules per class for the query set
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Max_atoms = 60  # Maximum number of atoms allowed
    Max_motif = 20  # Maximum number of motifs allowed

    save_path = "./checkpoints"
    pretain_pth = './checkpoints/pretain.pth'
    # Need to modify each time
    # Set pretain=True during pre-training, pretain=False during fine-tuning

    ismetatrain = True
    task_name = "tox21"  # BBBP, tox21, HIV, bace, clintox
    # BACE, BBBP, Tox21 completed
