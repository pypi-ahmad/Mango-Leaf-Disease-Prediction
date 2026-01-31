# MangoGuard: Multi-Model Agricultural AI Lab 🥭

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)
![Grad-CAM](https://img.shields.io/badge/Grad--CAM-Explainable%20AI-blue?style=for-the-badge)
![Models](https://img.shields.io/badge/EfficientNet%20|%20DenseNet%20|%20ResNet%20|%20MobileNet-green?style=for-the-badge)

> **State-of-the-Art Leaf Disease Diagnosis with Consensus Voting & Explainable AI.**

---

## 🌍 Project Overview

**Problem**: Early detection of mango crop diseases like **Anthracnose**, **Bacterial Canker**, and **Powdery Mildew** is critical for food security and farmer livelihood. Traditional manual inspection is slow and prone to error.

**Solution**: MangoGuard employs a **"Committee of Machines"** approach. Instead of relying on a single AI model, we orchestrate **4 distinct Deep Learning architectures** (DenseNet121, MobileNetV3, ResNet18, EfficientNet-B0) to vote on the diagnosis. This ensemble method reduces bias and increases reliability.

**Key Value**: We don't just give you a label; we show you **why**. Using **Grad-CAM (Gradient-weighted Class Activation Mapping)**, we visualize the exact regions of the leaf that triggered the diagnosis, creating a "Trust Layer" between the AI and the agronomist.

---

## ✨ Key Features

*   **🗳️ Consensus Diagnosis**: Four SOTA models analyze the image independently and vote. The system aggregates these votes to provide a high-confidence consensus prediction.
*   **🧠 Visual Explainability**: See what the AI sees. Compare side-by-side **Grad-CAM heatmaps** from different architectures to verify that the model is looking at the disease lesions, not the background.
*   **📊 Performance Radar**: Interactive **Plotly radar charts** visualize the strengths and weaknesses of each model (Precision vs. Recall vs. F1-Score).
*   **⚡ Lightweight & Fast**: Includes **MobileNetV3-Large**, optimized for mobile and edge deployment, proving that high accuracy doesn't always need heavy compute.

---

## 🏆 Performance Benchmarks

Our "Committee" was trained on the MangoLeafBD Dataset. Here are the validation accuracy results:

| Model Rank | Architecture | Accuracy | Role |
| :--- | :--- | :--- | :--- |
| 🥇 **1st** | **DenseNet121** | **99.88%** | The Deep Expert |
| 🥈 **2nd** | **MobileNetV3** | **99.75%** | The Speedster ⚡ |
| 🥉 **3rd** | **ResNet18** | **99.38%** | The Classic |
| 4th | EfficientNet-B0 | 98.88% | The Balanced |

---

## 🛠️ Tech Stack

*   **Core AI Engine**: [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/stable/index.html)
*   **User Interface**: [Streamlit](https://streamlit.io/)
*   **Visualization**: [Plotly Express](https://plotly.com/python/plotly-express/), [Matplotlib](https://matplotlib.org/)
*   **Explainable AI (XAI)**: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---

## 🚀 Quick Start

### 1. Installation
Clone the repo and install dependencies (supports CUDA 13.0).
```bash
pip install -r requirements.txt
```

### 2. Train the "Committee" (The Factory)
This script automatically downloads the dataset, trains all 4 models, and saves them into a single optimized bundle (`multi_model_bundle.pth`).
```bash
python train.py
```

### 3. Enter the Lab
Launch the interactive dashboard to diagnose images and explore the models.
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
📦 MangoGuard-AI-Lab
 ┣ 📂 data/                   # Dataset (Downloaded automatically)
 ┣ 📜 app.py                  # Streamlit Dashboard (The Lab)
 ┣ 📜 train.py                # Multi-Model Training Pipeline (The Factory)
 ┣ 📜 multi_model_bundle.pth  # The Trained Committee (Models + Metrics)
 ┣ 📜 requirements.txt        # Dependencies
 ┗ 📜 README.md               # You are here
```

---
*Created for the Future of Agriculture.* 🥭🤖
