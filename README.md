# Integrated Multimodal Hierarchical Fusion and Meta-Learning for Enhanced Molecular Property Prediction

This repository contains the full code and partial sample data for our paper:
**"Integrated Multimodal Hierarchical Fusion and Meta-Learning for Enhanced Molecular Property Prediction."**

We propose a multimodal hierarchical fusion framework that integrates molecular graphs and images, enhanced with meta-learning to improve molecular property prediction performance.

> ⚠️ Due to storage constraints, only a subset of molecular images is included here. The complete dataset, including molecular images, SMILES files, and pretrained weights, can be downloaded via the following links:

* **Baidu Netdisk**: [IMHF.zip (extraction code: 8qfp)](https://pan.baidu.com/s/1pbiOSKRX3QwnWk5qNJC1pA?pwd=8qfp)
* **Google Drive**: [Download Link](https://drive.google.com/file/d/1uuCCjxb9eG9-uTKsxl02--fgUL1iA8NF/view?usp=sharing)

---

## 📁 Project Structure

```
.
├── Module/                # Model components
├── Reptile/              # Meta-learning (Reptile algorithm)
├── checkpoints/          # Folder for pretrained or saved model weights
├── datasets/             # Data loading and processing
├── utils/                # Utility scripts
├── cnn_pretrain.py       # CNN pretraining script
├── config.py             # Global configuration
├── environment.yaml      # Conda environment definition
├── finetune.py           # Fine-tuning on downstream tasks
├── meta_train.py         # Meta-learning training script
└── README.md             # Project documentation
```

---

## ⚙️ Environment Setup

You can install required packages by either using the provided `environment.yaml` or manually installing the following dependencies:

### Option 1: Conda environment (recommended)

```bash
conda env create -f environment.yaml
conda activate mols
```

### Option 2: Manual installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge rdkit
pip install opencv-python
pip install scikit-learn
pip install seaborn
pip install tqdm
pip install einops
pip install nvitop
pip install hausdorff
pip install numba
pip install -U scikit-image
pip install tensorboard==1.1
```

---

## 🚀 Usage Instructions

### Step 1: Prepare Dataset

Download and extract the dataset and the checkpoints. Place all files according to the project’s directory structure. Make sure datasets are inside the `datasets/` folder and follow the correct hierarchy.

---

### Step 2: Pretrain the CNN (Optional)

Pretrain the image feature extractor (CNN):

```bash
python cnn_pretrain.py
```

Alternatively, use the provided pretrained CNN weights:

```
./checkpoints/CNN/cnn.pth
```

---

### Step 3: Meta-Learning Pretraining

Train the model using meta-learning:

```bash
python meta_train.py
```

Ensure the following is set in `config.py`:

```python
ismetatrain = True
```

Or load pretrained meta-model:

```
./checkpoints/pretain.pth
```

---

### Step 4: Fine-Tune on Downstream Tasks

Edit the configuration:

```python
ismetatrain = False
task_name = "BBBP"  # Options: bace, tox21, clintox, etc.
```

Then run:

```bash
python finetune.py
```

Results including ROC-AUC, accuracy, and F1 score will be printed and saved.

> Note: Downstream datasets are disjoint from meta-training datasets.

---

## ⚙️ Configuration Guide (`config.py`)

The `config.py` file contains all hyperparameters. Here's how to adjust them for different phases:

### ✅ General Settings

| Parameter     | Description                                       | Value                         |
| ------------- | ------------------------------------------------- | ----------------------------- |
| `ismetatrain` | `True` for meta-training, `False` for fine-tuning | `True` / `False`              |
| `task_name`   | Dataset name (for fine-tuning)                    | `"bace"`, `"tox21"` etc.      |
| `pretain_pth` | Pretrained model path                             | `"./checkpoints/pretain.pth"` |

### 📘 Meta-Learning Settings

| Parameter      | Description              | Default |
| -------------- | ------------------------ | ------- |
| `meta_batchsz` | Tasks per meta-batch     | 2       |
| `meta_lr`      | Outer loop learning rate | 0.001   |
| `num_updates`  | Inner loop updates       | 5       |
| `meta_epoch`   | Meta-training epochs     | 10      |

### 📕 Fine-Tuning Settings

| Parameter     | Description           | Default |
| ------------- | --------------------- | ------- |
| `epoch`       | Total training epochs | 100     |
| `random_seed` | Random seed           | 68      |

### 🧠 Model Architecture

| Parameter       | Description                    | Default |
| --------------- | ------------------------------ | ------- |
| `gnn_num_layer` | Number of GNN layers           | 3       |
| `p_dropout`     | Dropout rate                   | 0.2     |
| `imgsize`       | Input image size               | 224     |
| `img_dim`       | Image feature dimension        | 768     |
| `node_dim`      | Node latent dimension          | 256     |
| `edge_dim`      | Edge latent dimension          | 64      |
| `en_node_dim`   | Initial node feature dimension | 31      |
| `en_edge_dim`   | Initial edge feature dimension | 6       |

### ⚠️ Constraints

| Parameter   | Description                       | Default |
| ----------- | --------------------------------- | ------- |
| `Max_atoms` | Max number of atoms per molecule  | 60      |
| `Max_motif` | Max number of motifs per molecule | 20      |

---

## 📫 Contact

For questions, suggestions, or issues, please open an issue or contact the authors.
