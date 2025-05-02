# 🤗 Empathetic Response Generation with Transformer-Based Seq2Seq

> A Transformer-powered dialogue system that generates **emotionally intelligent responses** for mental health-related conversations. Built using PyTorch and the EmpatheticDialogues dataset.

---

## 🧠 Motivation

With the rising importance of mental health and the role AI can play in providing support, our goal was to build a conversational model that understands emotional context and responds empathetically using a state-of-the-art Transformer-based Sequence-to-Sequence (Seq2Seq) architecture.

---

## 📌 Objectives

- ✅ Implement a Transformer-based Seq2Seq model from scratch.
- ✅ Train it on emotionally grounded conversations.
- ✅ Generate empathetic, context-aware responses.
- ✅ Evaluate different embedding strategies: Trainable, GloVe, and FastText.

---

## 📚 Dataset

We used the [EmpatheticDialogues dataset](https://huggingface.co/datasets/empathetic_dialogues) released by Facebook AI, consisting of:

- 💬 24,000+ emotion-labeled conversations
- 🎭 32 emotion categories
- 📂 Split into train (19,533), validation (2,770), test (2,547)

---

## ⚙️ Model Architecture

Built from scratch using PyTorch:

- ✨ **Transformer Encoder-Decoder** design
- 🧩 **Multi-head attention**, **Positional encoding**
- 🔢 Embedding choices:
  - 🔹 Trainable (Best BLEU: 0.84)
  - 🔹 GloVe (BLEU: 0.18)
  - 🔹 FastText (BLEU: 0.82)

![Transformer](https://upload.wikimedia.org/wikipedia/commons/3/3f/Transformer.png)

> *Architecture details: Encoder + Decoder stacks with attention masks, embedding layers, and autoregressive decoding using greedy, top-k, and nucleus sampling.*

---

## 🛠️ Preprocessing & Training

- 🧹 WordPiece Tokenizer with `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- ✂ Max length: 60 tokens
- 📦 Batch size: 32
- 🔁 Early stopping with patience: 5
- 🎯 Optimizer: Adam
- ⏱️ Max Epochs: 50
- 🔒 Reproducibility ensured by seeding `random`, `torch`, and `PYTHONHASHSEED`

---

## 🔍 Evaluation

| Embedding Type       | BLEU Score | Convergence Epoch |
|----------------------|------------|-------------------|
| Trainable Embeddings | **0.84**   | 50                |
| GloVe                | 0.18       | 40                |
| FastText             | 0.82       | **30**            |

- 📉 Loss and validation tracked per epoch
- 📊 BLEU scores calculated on test set
- 🔍 Sampling: Greedy, Top-k, Top-p

---

## 🚀 Inference

```python
generate_response("I’m feeling lonely today", strategy="topk", k=10, temperature=0.8)
