# 🧠 Brain Disease Classification using EEG Signals

<p align="center">
  <strong>An End-to-End Brain-Computer Interface (BCI) Project for EEG-Based Classification of Alzheimer's Disease, Frontotemporal Dementia, and Cognitively Normal Subjects</strong>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![MNE-Python](https://img.shields.io/badge/MNE-Python-green)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![EEG](https://img.shields.io/badge/EEG-Brain%20Signals-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

</p>

---

# 📖 Overview

This project is an **EEG-based Brain Disease Classification System** developed as part of a **Brain-Computer Interface (BCI)** project. It focuses on analyzing **resting-state Electroencephalography (EEG)** signals to assist in the classification of neurodegenerative disorders.

The system processes EEG recordings from **88 subjects**, performs signal preprocessing, extracts meaningful features, visualizes brain activity, and prepares the data for **Machine Learning** and **Deep Learning** models capable of classifying:

* 🧠 Alzheimer's Disease (AD)
* 🧠 Frontotemporal Dementia (FTD)
* 🧠 Cognitively Normal (CN)

The project is designed with a modular pipeline, making it easy to extend with advanced feature extraction techniques and deep learning architectures.

---

# ✨ Features

* 🧠 EEG Signal Processing
* ⚡ Bandpass Filtering
* 📈 Multi-channel EEG Visualization
* 📊 Power Spectral Density (PSD) Analysis
* 🔥 Brain Activity Heatmap Generation
* 📉 Principal Component Analysis (PCA)
* 🌐 t-SNE Visualization
* 🤖 Machine Learning Ready Feature Extraction
* 🧩 Modular Deep Learning Pipeline
* 📂 Automatic Plot Generation

---

# 🏗️ Project Architecture

```text
                    EEG Dataset (.set)
                           │
                           ▼
                 Load EEG using MNE-Python
                           │
                           ▼
                Bandpass Filtering (1–50 Hz)
                           │
                           ▼
               EEG Signal Visualization
                           │
                           ▼
          Power Spectral Density (PSD)
                           │
                           ▼
               EEG Feature Extraction
                           │
                           ▼
           Brain Activity Heatmap Generation
                           │
                           ▼
                  PCA Visualization
                           │
                           ▼
                 t-SNE Visualization
                           │
                           ▼
        Machine Learning / Deep Learning
                           │
                           ▼
       Brain Disease Classification
```

---

# 📂 Dataset

This project utilizes the publicly available **OpenNeuro Dataset (ds004504)** containing resting-state **eyes-closed EEG recordings**.

### Dataset Summary

| Category                      | Number of Subjects |
| ----------------------------- | -----------------: |
| Alzheimer's Disease (AD)      |             **36** |
| Frontotemporal Dementia (FTD) |             **23** |
| Cognitively Normal (CN)       |             **29** |
| **Total Subjects**            |             **88** |

---

# 📊 EEG Recording Details

The EEG recordings were acquired using a **Nihon Kohden EEG-2100 Clinical EEG System** with **19 scalp electrodes** placed according to the **International 10–20 System**.

### Recording Specifications

| Parameter             | Value                       |
| --------------------- | --------------------------- |
| Recording Type        | Resting-State (Eyes Closed) |
| EEG Channels          | 19                          |
| Reference Electrodes  | A1 & A2                     |
| Sampling Frequency    | **500 Hz**                  |
| Recording Resolution  | **10 μV/mm**                |
| High Frequency Filter | **70 Hz**                   |
| Skin Impedance        | **< 5 kΩ**                  |

---

# 🧹 Preprocessing Pipeline

The dataset provides both **raw** and **preprocessed** EEG recordings.

The preprocessing pipeline includes:

* Butterworth Bandpass Filter (0.5–45 Hz)
* Re-referencing (A1–A2)
* Artifact Subspace Reconstruction (ASR)
* Independent Component Analysis (ICA)
* Automatic Artifact Removal using ICLabel

These preprocessing steps reduce noise and improve signal quality for downstream analysis.

---

# ⚙️ Workflow

```text
EEG Dataset
      │
      ▼
Load EEG (.set)
      │
      ▼
Bandpass Filtering
      │
      ▼
Signal Visualization
      │
      ▼
Power Spectral Density (PSD)
      │
      ▼
Feature Extraction
      │
      ▼
Brain Activity Heatmap
      │
      ▼
PCA Analysis
      │
      ▼
t-SNE Visualization
      │
      ▼
Machine Learning / Deep Learning
      │
      ▼
Brain Disease Classification
```

---

# 📁 Repository Structure

```text
Brain-Disease-Classification-EEG/

├── EEG_Dataset/
│   ├── sub-001/
│   ├── sub-002/
│   ├── ...
│   └── derivatives/
│
├── outputs/
│
├── 3a.py
├── final.py
├── xyz.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

> **Note:** The EEG dataset is **not included** in this repository due to its size. Please download the **OpenNeuro ds004504** dataset separately before running the project.

---

# 🚀 Getting Started

## Clone the Repository

```bash
git clone https://github.com/<your-username>/Brain-Disease-Classification-EEG.git
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download the Dataset

Download the **OpenNeuro ds004504** dataset and update the dataset path in the source code.

---

## Run the Project

```bash
python final.py
```

Generated visualizations will be automatically saved inside the **outputs/** directory.

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Libraries

* MNE-Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Techniques

* EEG Signal Processing
* Power Spectral Density (PSD)
* Principal Component Analysis (PCA)
* t-SNE
* Feature Engineering

---

# 🔮 Future Enhancements

* Continuous Wavelet Transform (CWT)
* EEG Time-Frequency Image Generation
* Data Augmentation
* CNN-Based Classification
* EEGConvNeXt Architecture
* Transfer Learning
* Grad-CAM Explainability
* Hyperparameter Optimization
* Model Performance Evaluation
* Real-Time EEG Classification

---

# 📚 Dataset Citation

If you use this dataset in your research, please cite:

> **A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG**
>
> *Data*, **2023**
>
> DOI: **https://doi.org/10.3390/data8060095**

---

# 👨‍💻 Author

**Harshavardhan Korlepara**

B.Tech – Artificial Intelligence & Machine Learning

**Areas of Interest**

* Brain-Computer Interface (BCI)
* EEG Signal Processing
* Machine Learning
* Deep Learning
* Biomedical Data Analysis
* Explainable AI

---

# ⭐ Support

If you found this project useful, please consider **starring ⭐ this repository**.
