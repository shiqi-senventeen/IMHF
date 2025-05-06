
# Integrated Multimodal Hierarchical Fusion and Meta-Learning for Enhanced Molecular Property Prediction

This repository contains the full code and partial sample data for our paper:  
**"Integrated Multimodal Hierarchical Fusion and Meta-Learning for Enhanced Molecular Property Prediction."**

We propose a multimodal hierarchical fusion framework that integrates molecular graphs and images, enhanced with meta-learning to improve molecular property prediction performance.

> ⚠️ Due to storage constraints, only a subset of molecular images is included here. The complete dataset, including molecular images, SMILES files, and pretrained weights, can be downloaded via the following links:

- **Baidu Netdisk**: [dataset.7z (extraction code: 8qfp)](https://pan.baidu.com/s/1pbiOSKRX3QwnWk5qNJC1pA?pwd=8qfp)  
- **Google Drive**: [Download Link](https://drive.google.com/file/d/1RcgFieKUMTJ31is0x8uc7cUqGEbsUSqJ/view?usp=sharing)

---

## 📁 Project Structure

```

.
├── Module/                # Model components
├── Reptile/               # Meta-learning (Reptile algorithm)
├── checkpoints/           # Folder for pretrained or saved model weights
├── datasets/              # Data loading and processing
├── utils/                 # Utility scripts
├── cnn\_pretrain.py        # CNN pretraining script
├── config.py              # Global configuration
├── environment.yaml       # Conda environment definition
├── finetune.py            # Fine-tuning on downstream tasks
├── meta\_train.py          # Meta-learning training script
└── README.md              # Project documentation

````

---

## ⚙️ Environment Setup

To create the required environment, run:

```bash
conda env create -f environment.yaml
conda activate multimodal-molecule
````

---

## 🚀 Usage Instructions

### Step 1: Pretrain the CNN (Optional)

Pretrain the image feature extractor (CNN) using:

```bash
python cnn_pretrain.py
```

Alternatively, use the provided pretrained weight:

```
./checkpoints/CNN/cnn.pth
```

---

### Step 2: Meta-Learning Pretraining

Train the base model using the meta-learning framework:

```bash
python meta_train.py
```

Alternatively, load the pretrained meta-model from:

```
./checkpoints/pretain.pth
```


### Step 3: Fine-Tune on Downstream Molecular Property Prediction Tasks

To fine-tune the model on a specific molecular property prediction dataset (e.g., **BACE**, **BBBP**, **SIDER**, **Tox21**), edit the following parameters in `config.py`:

```python
# config.py
````

Replace `bace` with the desired dataset name. Supported dataset options include:

* `bace`
* `bbbp`
* `clintox`
* `tox21`
* *(other datasets depending on your setup)*


Evaluation metrics such as accuracy, ROC-AUC, F1 score, and others will be printed to the console and saved for later review.

**Note**: All downstream datasets are disjoint from the meta-training set to ensure fair evaluation.



## 📫 Contact

For questions, suggestions, or issues, please open an issue or contact the authors.

