# 🤖 ZENDS AI Customer Support Copilot

> An end-to-end AI-powered customer support system for ZENDS Communications — a virtual telecom company.
> Built using NLP, Deep Learning, HuggingFace Transformers, RAG, and Streamlit.

---

## 🏢 Company Overview

**ZENDS Communications** is a virtual telecom company providing:

* 📱 5G Mobile Plans
* 🌐 Fiber Broadband
* ☁️ Cloud Storage
* 🔌 IoT Devices
* 🏢 Enterprise Network Solutions

---

## ❗ Problem Statement

Support agents face challenges handling large volumes of customer queries across:

* Billing issues
* Refund requests
* Technical problems
* Complaints
* Product inquiries

Manual responses are slow, inconsistent, and resource-intensive.

---

## ✅ Solution

An AI Customer Support Copilot that:

* 🎯 Detects **customer intent** automatically
* 😊 Analyzes **customer sentiment**
* 📚 Retrieves **relevant information** using RAG
* 🤖 Generates **context-aware responses**
* 💬 Provides an **interactive Streamlit interface**

---

## 🧠 Tech Stack

| Technology               | Usage                            |
| ------------------------ | -------------------------------- |
| Python                   | Core programming                 |
| HuggingFace Transformers | DistilBERT Intent Classification |
| Sentence Transformers    | Text Embeddings                  |
| FAISS                    | Vector Search                    |
| Streamlit                | Web Application                  |
| PyTorch                  | Deep Learning                    |
| Scikit-learn             | Label Encoding, Metrics          |
| Pandas / NumPy           | Data Processing                  |

---

## 📁 Project Structure

```
zends_ai_copilot/
│
├── src/
│   ├── data_generator.py
│   ├── train.py
│   ├── sentiment.py
│   ├── rag.py
│   └── llm_response.py
│
├── models/
├── data/
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow

```
Dataset → Training → Intent Prediction → Sentiment Analysis → RAG Retrieval → Response Generation → Streamlit UI
```

---

## 📊 Model Performance

* Accuracy: **100%**
* F1 Score: **100%**

> Note: High accuracy due to synthetic dataset.

---

## 🚀 How to Run

```bash
git clone https://github.com/tamilarasirajendran/zends-ai-copilot
cd zends-ai-copilot
pip install -r requirements.txt
streamlit run app.py
```

---

## 💡 Features

* Intent Classification
* Sentiment Analysis
* RAG-based Retrieval
* Context-aware Responses
* Clean Chat UI

---

## 🙋 Author

**Tamilarasi Rajendran**

---

## 📄 License

For educational purposes only.
