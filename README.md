# Forewarned is Forearmed: Joint Prediction and Classification of Optical Transceiver Failures in Large-Scale LLM Training Clusters

## Project Introduction

This project, **OptProphet**, presents a unified framework for predicting and classifying failures of optical transceivers in hyperscale LLM training clusters. It leverages temporal signal processing, transformer-based representation learning, and multi-task learning with MMOE (Multi-gate Mixture-of-Experts) to jointly model **soft**, **hard**, and **normal** transceiver behavior. The pipeline includes automatic TSFEL-based feature extraction, transformer encoding, class-balanced data augmentation, and MMOE-based failure prediction and classification.

---

## Directory Structure

```
OptProphet-model 
├── data                                        # Original and intermediate data
│   ├── offline                                 # Raw offline CSV data (historical labeled samples)
│   ├── offline_tsfel                           # TSFEL features extracted from offline data
│   ├── offline_feature                         # Transformer features from offline data
│   ├── online                                  # Raw online CSV data (real-time samples)
│   ├── online_tsfel                            # TSFEL features extracted from online data
│   ├── online_feature                          # Transformer features from online data
│   ├── feature_two_step_train.npz              # Intermediate stacked feature file
│   └── train_task                              # Augmented training data for MMOE
├── result                                      # MMOE final inference results
├── tsfel_feature_extractor.py                  # TSFEL feature extraction module
├── transformer.py                              # Transformer encoder and feature stacking
├── data_augmentation.py                        # Feature-space sampling for class balancing
├── MOE.py                                      # MMOE training and inference module
├── main.py                                     # Full pipeline: TSFEL → Transformer → Augmentation → MMOE
├── model.pth                                   # Saved MMOE model
└── requirements.txt                            # Python package requirements
```

---

## Data Description

### Dataset Overview

All data in this project are synthetic time-series readings generated to demonstrate pipeline functionality rather than represent actual transceiver measurements. Each sequence captures seventeen electrical and optical metrics (e.g. current, temperature, voltage, TX/RX power and multi-channel bias readings) along with two labels—`subhealth_label` (healthy vs. sub-healthy) and `failure_type` (soft failure, hard failure).


### Directory Description

The top‐level `data` folder organizes both raw and processed files into separate subdirectories for offline (historical) and online (real‐time) streams, plus intermediate outputs. Within each of these (`offline`, `offline_tsfel`, `offline_feature`, `online`, `online_tsfel`, `online_feature`, `train_task`), there are three class‐specific subfolders:

  ```
  <stage_folder>/
  ├── normal                                    # healthy samples
  ├── hard                                      # hard‐failure samples
  └── soft                                      # soft‐failure samples
  ```

### Sample Data Format

Each CSV in offline/ or online/ begins with a header containing a timestamp, a set of metric columns (typically several electrical and optical features), and two label columns. For example:

  ```csv
  timestamp,Current,Temperature,Voltage,CurrentTXPower,…,CurrentMultiBias4,subhealth_label,failure_type
  2025-05-05 16:35:14.419676,12.08,47.54,3.13,…,12.64,0,0
  ```

---

## Installation

### 1. Clone and Install Dependencies

1. Download the project
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: this project was tested on Python 3.9.21.*

### 2. Configuration

All pipeline parameters can be customized either via command-line flags or by editing a YAML configuration file. By default, `main.py` accepts the following options:

| Parameter                | Description                                    | Default                                                                                              |
| ------------------------ | ---------------------------------------------- |------------------------------------------------------------------------------------------------------|
| `--window_size`          | Sliding window size for TSFEL                  | `24`                                                                                                 |
| `--step_size`            | Step size for TSFEL window                     | `1`                                                                                                  |
| `--first_feature`        | First column index used in data augmentation   | `0`                                                                                                  |
| `--last_feature`         | Last column index used in data augmentation    | `2311`                                                                                               |
| `--train_name`           | Output folder name for augmented training data | `train_task`                                                                                         |
| `--offline_input_dirs`   | Raw offline data directories                   | `['./data/offline/hard/', './data/offline/soft/', './data/offline/normal/']`                         |
| `--offline_output_dirs`  | Output dirs for offline TSFEL features         | `['./data/offline_tsfel/hard/', './data/offline_tsfel/soft/', './data/offline_tsfel/normal/']`       |
| `--online_input_dirs`    | Raw online data directories                    | `['./data/online/hard/', './data/online/soft/', './data/online/normal/']`                            |
| `--online_output_dirs`   | Output dirs for online TSFEL features          | `['./data/online_tsfel/hard/', './data/online_tsfel/soft/', './data/online_tsfel/normal/']`          |
| `--offline_feature_dirs` | Output dirs for offline Transformer features   | `['./data/offline_feature/hard/', './data/offline_feature/soft/', './data/offline_feature/normal/']` |
| `--online_feature_dirs`  | Output dirs for online Transformer features    | `['./data/online_feature/hard/', './data/online_feature/soft/', './data/online_feature/normal/']`    |
| `--stacked_output_path`  | Path for offline stacked feature file          | `./data/feature_two_step_train.npz`                                                                  |
| `--result_dirs`          | Directories to save results                    | `['./result/hard/', './result/soft/', './result/normal/']`                                           |

---

## How to Use

### 1. Prepare Input Files

Under both `data/offline` and `data/online`, create three class folders—`normal`, `hard`, and `soft`. In each folder place a `data.csv` containing the time-series samples for that class:

```
data/
├── offline/
│   ├── normal/    # healthy samples
│   │   └── data.csv
│   ├── hard/      # hard-failure samples
│   │   └── data.csv
│   └── soft/      # soft-failure samples
│       └── data.csv
└── online/
    ├── normal/    # healthy samples
    │   └── data.csv
    ├── hard/      # hard-failure samples
    │   └── data.csv
    └── soft/      # soft-failure samples
        └── data.csv
```

Each `data.csv` must use the schema shown in **Sample Data Format**, with a timestamp column, metric columns, and the two labels.

### 2. Run the Pipeline

Once your CSV files are in place, execute the full workflow:

```bash
python main.py
```

### 3. Run with Custom Parameters (Optional)

You can override the default arguments using command-line options. For example, to modify the TSFEL window size, specify custom feature column indices, and change output directories:

```bash
python main.py \
  --window_size 32 \
  --step_size 4 \
  --first_feature 0 \
  --last_feature 2345 \
  --train_name custom_aug \
  --offline_input_dirs ./mydata/offline/hard ./mydata/offline/soft ./mydata/offline/normal \
  --online_input_dirs ./mydata/online/hard ./mydata/online/soft ./mydata/online/normal \
  --stacked_output_path ./outputs/custom_features.npz \
  --result_dirs ./outputs/results/hard ./outputs/results/soft ./outputs/results/normal
```

---



