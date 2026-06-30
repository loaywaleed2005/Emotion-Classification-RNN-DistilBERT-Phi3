# Emotion Classification using RNN, DistilBERT, and Phi-3

An advanced natural language processing project that compares traditional deep learning, transformer fine-tuning, and large language model prompting for six-class emotion classification.

## Project Overview

The project classifies text into one of six emotions:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

Three different NLP approaches are implemented and compared:

1. A bidirectional LSTM trained from scratch
2. Fine-tuned DistilBERT
3. Phi-3 using zero-shot and few-shot prompting

## Dataset

The project uses the `dair-ai/emotion` dataset.

Dataset splits:

- Training: 16,000 samples
- Validation: 2,000 samples
- Test: 2,000 samples

## Data Preparation

The workflow includes:

- Loading the dataset
- Inspecting class distributions
- Cleaning and tokenizing text
- Building a vocabulary for the RNN
- Padding sequences
- DistilBERT tokenization
- Creating PyTorch DataLoaders

## RNN Model

The baseline deep learning model uses:

- Word embeddings
- Bidirectional LSTM
- Dropout regularization
- Fully connected classification layer
- Cross-entropy loss
- Adam optimizer

The model is trained from scratch on the emotion dataset.

## DistilBERT Model

The transformer approach uses:

- `distilbert-base-uncased`
- Six-class classification head
- Full-model fine-tuning
- AdamW optimizer
- Linear learning-rate scheduling
- Gradient clipping

DistilBERT benefits from pretrained contextual language representations.

## Phi-3 Prompting

The project uses:

- `microsoft/Phi-3-mini-4k-instruct`
- 4-bit quantization
- Zero-shot prompting
- Three-example few-shot prompting
- Eight-example few-shot prompting
- Chain-of-thought-style prompting
- Safe output parsing

Phi-3 is evaluated without gradient updates or fine-tuning.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro F1-score
- Per-class F1-score
- Classification reports
- Confusion matrices
- Training time
- Inference time

## Model Comparison

The project compares:

- Predictive performance
- Training requirements
- Inference speed
- Model size
- Performance in low-data settings
- Benefits of pretrained representations

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Microsoft Phi-3
- LSTM
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Google Colab

## Project File

Open `Emotion_Classification_RNN_DistilBERT_Phi3.ipynb` to view the complete preprocessing, training, prompting, evaluation, and model comparison workflow.

## Author

Loay Waleed
