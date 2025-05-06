
# Integrated Multimodal Hierarchical Fusion and Meta-Learning for Enhanced Molecular Property Prediction

![Overview](overview.png)
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
├── Reptile/               # Meta-learning (Reptile algorithm)
├── checkpoints/           # Folder for pretrained or saved model weights
├── datasets/              # Data loading and processing
├── utils/                 # Utility scripts
├── cnn_pretrain.py        # CNN pretraining script
├── config.py              # Global configuration
├── environment.yaml       # Conda environment definition
├── finetune.py            # Fine-tuning on downstream tasks
├── meta_train.py          # Meta-learning training script
└── README.md              # Project documentation
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

Download and extract the dataset and checkpoints. Place all files according to the project’s directory structure. Make sure datasets are inside the `datasets/` folder and follow the correct hierarchy.

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

---

## 🧪 Implementation Details

Our training process consists of two main stages: **meta-learning pretraining** and **fine-tuning on downstream property prediction tasks**.

### 🔁 Meta-Learning Pretraining

We adopt the Reptile meta-learning algorithm. This phase helps the model learn transferable molecular representations from auxiliary datasets.

| Parameter      | Value | Description                    |
| -------------- | ----- | ------------------------------ |
| `meta_batchsz` | 2     | Number of tasks per meta-batch |
| `meta_lr`      | 0.001 | Outer loop learning rate       |
| `num_updates`  | 5     | Number of inner-loop updates   |
| `meta_epoch`   | 10    | Total meta-training epochs     |

Run the training using:

```bash
python meta_train.py
```

Ensure the following flag is set in `config.py`:

```python
ismetatrain = True
```

### 🎯 Fine-Tuning on Downstream Tasks

After meta-training, we fine-tune the model on disjoint benchmark datasets such as BBBP, BACE, Tox21, etc.

| Parameter     | Value  | Description                  |
| ------------- | ------ | ---------------------------- |
| `epoch`       | 100    | Total fine-tuning epochs     |
| `random_seed` | 68     | Seed for reproducibility     |
| `task_name`   | "BBBP" | Target task name             |
| `ismetatrain` | False  | Switches to fine-tuning mode |

Run fine-tuning with:

```bash
python finetune.py
```

### 🧩 Model Architecture

Our IMHF framework integrates multimodal information from both molecular graphs and images. The architecture uses a hierarchical fusion mechanism with the following major modules:

| Component       | Value | Description                          |
| --------------- | ----- | ------------------------------------ |
| `gnn_num_layer` | 3     | Number of GNN layers                 |
| `node_dim`      | 256   | Node hidden dimension                |
| `edge_dim`      | 64    | Edge hidden dimension                |
| `en_node_dim`   | 31    | Initial node feature dimension       |
| `en_edge_dim`   | 6     | Initial edge feature dimension       |
| `p_dropout`     | 0.2   | Dropout rate                         |
| `imgsize`       | 224   | Input molecular image size           |
| `img_dim`       | 768   | Image feature dimension (CNN output) |

---

## 📫 Contact

For questions, suggestions, or issues, please open an issue or contact the authors.
