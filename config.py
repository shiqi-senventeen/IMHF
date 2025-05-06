import torch


class args:
    """
    Configuration class for meta-learning with graph neural networks.
    Contains hyperparameters for model architecture, training process, and dataset handling.
    """
    # Meta-learning parameters
    meta_batchsz = 2  # Number of tasks sampled for each meta-batch
    meta_lr = 0.001  # Meta-learning rate for outer loop optimization
    num_updates = 5  # Number of gradient updates for the inner loop (support set)
    meta_epoch = 10  # Number of meta-training epochs

    # Model architecture parameters
    gnn_num_layer = 3  # Number of message passing layers in the graph neural network
    p_dropout = 0.2  # Dropout probability for regularization

    # Input feature dimensions
    imgsize = 224  # Size of input images (height=width)
    en_node_dim = 31  # Dimension of initial atom (node) features
    en_edge_dim = 6  # Dimension of initial bond (edge) features

    # Latent representation dimensions
    node_dim = 256  # Dimension of atom embeddings after encoding
    edge_dim = 64  # Dimension of bond embeddings after encoding
    img_dim = 768  # Dimension of image features
    img_channels = 3  # Number of image channels (RGB)

    # Task definition parameters
    per_task_output_units_num = 2  # Output dimension (binary classification)
    n_way = 2  # N-way classification task (binary)
    k_shot = 2  # K samples per class for support set
    k_query = 5  # Number of query samples per class for evaluation

    # Training parameters
    epoch = 100  # Number of training epochs
    random_seed = 68  # Seed for reproducibility

    # Hardware configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Molecule representation constraints
    Max_atoms = 60  # Maximum number of atoms per molecule
    Max_motif = 20  # Maximum number of structural motifs per molecule

    # File paths
    save_path = "./checkpoints"  # Directory to save model checkpoints
    pretain_pth = './checkpoints/pretain.pth'  # Path to pre-trained model weights

    # Training mode configuration
    ismetatrain = True  # Whether to use meta-training (True) or standard training (False)

    # Dataset selection
    task_name = "tox21"  # Dataset name: options include BBBP, tox21, HIV, bace, clintox
    # Note: BACE, BBBP, Tox21 have been implemented and tested