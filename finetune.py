# Import evaluation metrics from sklearn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, auc

import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from rdkit import RDLogger
from Module import UMPredict
from Module import ValDataset
import warnings
from config import args

# Suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def main():
    # Lists to store ROC-AUC scores for training, validation, and testing
    train_roc_aucs = []
    val_roc_aucs = []
    test_roc_aucs = []

    dataset_dir = "./dataset/"

    # Paths for different benchmark datasets
    task_paths = {
        'BBBP': 'BBBP/BBBP_',
        'bace': 'bace/bace_',
        'HIV': 'HIV/HIV_',
        'clintox': 'clintox/clintox_',
        'tox21': 'tox21/tox21_'
    }

    # Select the dataset path based on the task name
    if args.task_name in task_paths:
        base_path = dataset_dir + task_paths[args.task_name]
        train_csv_path = base_path + 'train.csv'
        val_csv_path = base_path + 'val.csv'
        test_csv_path = base_path + 'test.csv'
    else:
        raise ValueError(f"Unsupported task name: {args.task_name}")

    # Load validation and test datasets
    valdaset = ValDataset(val_csv_path)
    val_dataloader = DataLoader(valdaset, batch_size=1)
    testdataset = ValDataset(test_csv_path)
    test_dataloader = DataLoader(testdataset, batch_size=1)

    # Initialize the model
    net = UMPredict(31, 6, 0.).to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # Load pretrained model weights
    model_state_dict = torch.load(args.pretain_pth)
    net.load_state_dict(model_state_dict)

    # Load training dataset
    train_dataset = ValDataset(train_csv_path)

    # Print total number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # Create DataLoader for training set
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)

    net.train()

    # Training loop
    for epoch in range(args.epoch):
        y_train = []
        predictions = []

        print("training:")
        for atom_features, edge_features, padding_mask, img_features, label in tqdm(train_dataloader):
            atom_features = atom_features.to(torch.float32).to(args.device)
            edge_features = edge_features.to(torch.float32).to(args.device)
            img_features = img_features.to(torch.float32).to(args.device)
            padding_mask = padding_mask.to(args.device)
            label = label.to(torch.float32).to(args.device)

            optimizer.zero_grad()

            # Forward pass and compute loss
            loss, predict, target = net(atom_features, edge_features, padding_mask, img_features, label)

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()

            y_train.extend(target.flatten().tolist())
            predictions.extend(predict.flatten().tolist())

        # Compute training metrics
        predicted_labels = [1.0 if p >= 0.5 else 0.0 for p in predictions]
        roc_auc = roc_auc_score(y_train, predictions)
        train_roc_aucs.append(roc_auc)
        accuracy = accuracy_score(y_train, predicted_labels)
        f1 = f1_score(y_train, predicted_labels)
        roc_curve_values = roc_curve(y_train, predictions)
        auc_value = auc(roc_curve_values[0], roc_curve_values[1])

        print(f'epoch {epoch + 1}/{args.epoch} : Loss: {loss.item():.3f}\t\t',
              f'roc-auc: {roc_auc:.6f}',
              f'\t\taccuracy: {accuracy:.6f}',
              f'\t\tf1: {f1:.6f}',
              f'\t\tauc_value: {auc_value:.6f}')

        # Validation phase
        net.eval()
        with torch.no_grad():
            y_val = []
            predictions = []

            for atom_features, edge_features, padding_mask, img_features, label in val_dataloader:
                atom_features = atom_features.to(torch.float32).to(args.device)
                edge_features = edge_features.to(torch.float32).to(args.device)
                img_features = img_features.to(torch.float32).to(args.device)
                padding_mask = padding_mask.to(args.device)
                label = label.to(torch.float32).to(args.device)

                loss, predict, target = net(atom_features, edge_features, padding_mask, img_features, label)
                y_val.extend(target.flatten().tolist())
                predictions.extend(predict.flatten().tolist())

        # Compute validation metrics
        predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]
        roc_auc = roc_auc_score(y_val, predictions)
        val_roc_aucs.append(roc_auc)
        accuracy = accuracy_score(y_val, predicted_labels)
        f1 = f1_score(y_val, predicted_labels)
        roc_curve_values = roc_curve(y_val, predictions)
        auc_value = auc(roc_curve_values[0], roc_curve_values[1])

        print('val:\t\troc-auc:%.6f' % roc_auc,
              '\t\taccuracy:%.6f' % accuracy,
              '\t\tf1:%.6f' % f1,
              '\t\tauc_value:%.6f' % auc_value)

        # Test phase
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

                loss, predict, target = net(atom_features, edge_features, padding_mask, img_features, label)
                y_test.extend(target.flatten().tolist())
                predictions.extend(predict.flatten().tolist())

        # Compute test metrics
        predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]
        roc_auc = roc_auc_score(y_test, predictions)
        test_roc_aucs.append(roc_auc)
        accuracy = accuracy_score(y_test, predicted_labels)
        f1 = f1_score(y_test, predicted_labels)
        roc_curve_values = roc_curve(y_test, predictions)
        auc_value = auc(roc_curve_values[0], roc_curve_values[1])

        print('test:\t\troc-auc:%.6f' % roc_auc,
              '\t\taccuracy:%.6f' % accuracy,
              '\t\tf1:%.6f' % f1,
              '\t\tauc_value:%.6f' % auc_value)

    # Print metrics for each epoch
    print("Training ROC-AUC values per epoch:")
    for epoch, roc_auc in enumerate(train_roc_aucs, 1):
        print(f'Epoch {epoch}: {roc_auc:.4f}')

    print("Val ROC-AUC values per epoch:")
    for epoch, roc_auc in enumerate(val_roc_aucs, 1):
        print(f'Epoch {epoch}: {roc_auc:.4f}')

    print("Test ROC-AUC values per epoch:")
    for epoch, roc_auc in enumerate(test_roc_aucs, 1):
        print(f'Epoch {epoch}: {roc_auc:.4f}')

    # Save final model
    print("save model")
    path = "./checkpoints/" + args.task_name + ".pth"
    torch.save(net.state_dict(), path)


if __name__ == "__main__":
    main()
