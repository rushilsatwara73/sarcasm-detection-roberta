# 🎭 Detecting Sarcasm as Sentiment Incongruence using RoBERTa

This project investigates **Sarcasm Detection in Natural Language Processing (NLP)** by analyzing contextual sentiment incongruence. We compare traditional baseline models (TF-IDF + Logistic Regression) against deep learning transformer models (**RoBERTa**).

---

## 📌 Project Highlights
- **Contextual Representation**: Harnesses pre-trained `RoBERTa` transformer embeddings to capture nuanced context and implicit sarcasm.
- **Sentiment Incongruence Analysis**: Explores tone disparities (e.g., positive sentiment paired with negative context) as an interpretability signal.
- **Comparative Benchmarking**: Evaluates TF-IDF baseline against RoBERTa across key classification metrics (Accuracy, Precision, Recall, Macro F1, ROC-AUC).

---

## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | Macro F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 77.13% | 77.40% | 77.35% | 0.7713 | 0.8581 |
| **RoBERTa (Fine-Tuned)** | **81.06%** | **81.15%** | **80.87%** | **0.8094** | **0.8951** |

> **Key Finding**: RoBERTa achieves superior performance across all evaluation metrics, yielding a ~3.7% absolute gain in accuracy and an ROC-AUC of **0.8951**.

---

## 📁 Repository Structure

```
.
├── Detecting_Sarcasm_as_Sentiment_Incongruence_UPDATED.ipynb   # Main Jupyter Notebook
├── Sarcasm_Headlines_Dataset_v2.json                           # News Headlines Sarcasm Dataset
├── Sarcasm_Detection_Using_RoBERTa_Technical_Report.pdf        # Full Technical Report (PDF)
├── Sarcasm_Detection_Using_RoBERTa_Technical_Report.docx       # Technical Report (Word)
├── Arch.png                                                     # Model Architecture Diagram
├── pipeline.png                                                 # Data & Model Pipeline Diagram
├── real.png                                                     # Real-world inference example
├── sarcasm_outputs/                                             # Plots, confusion matrix, ROC curve
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_curve.png
│   └── model_comparison.csv
├── .gitignore                                                   # Git Ignore Configuration
└── README.md                                                    # Project Documentation
```

---

## ⚙️ Datasets Used
1. **News Headlines Sarcasm Dataset v2**: 28,619 news headlines labeled with binary sarcasm targets (`is_sarcastic`). Primary supervised training source.
2. **Amazon Unlocked Mobile Dataset**: Auxiliary dataset used for evaluating sentiment analysis and contextual incongruence across user ratings. *(Note: Download separately due to size limits)*.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/sarcasm-detection-roberta.git
cd sarcasm-detection-roberta
```

### 2. Install dependencies
```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn
```

### 3. Run the Notebook
Open `Detecting_Sarcasm_as_Sentiment_Incongruence_UPDATED.ipynb` in Jupyter Notebook or Google Colab to reproduce training, validation, and evaluation.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
