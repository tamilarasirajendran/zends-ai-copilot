<<<<<<< HEAD
# 🤖 ZENDS AI Customer Support Copilot

> An end-to-end AI-powered customer support system for ZENDS Communications — a virtual telecom company.
> Built using NLP, Deep Learning, HuggingFace Transformers, RAG, and Streamlit.

---

## 🏢 Company Overview

**ZENDS Communications** is a virtual telecom company providing:
- 📱 5G Mobile Plans
- 🌐 Fiber Broadband
- ☁️ Cloud Storage
- 🔌 IoT Devices
- 🏢 Enterprise Network Solutions

---

## ❗ Problem Statement

Support agents face challenges handling large volumes of customer queries across:
- Billing issues
- Refund requests
- Technical problems
- Complaints
- Product inquiries

Manual responses are slow, inconsistent, and resource-intensive.

---

## ✅ Solution

An AI Customer Support Copilot that:
- 🎯 Detects **customer intent** automatically
- 😊 Analyzes **customer sentiment**
- 📚 Retrieves **relevant policy information** using RAG
- 🤖 Generates **accurate responses**
- 💬 Provides an **interactive Streamlit dashboard**

---

## 🧠 Tech Stack

| Technology | Usage |
|------------|-------|
| Python | Core programming |
| HuggingFace Transformers | DistilBERT Intent Classification |
| Sentence Transformers | Text Embeddings for RAG |
| FAISS | Vector Database |
| Streamlit | Web Application |
| PyTorch | Deep Learning Framework |
| Scikit-learn | Label Encoding, Evaluation |
| Pandas / NumPy | Data Processing |
| Matplotlib | EDA Visualization |

---

## 📁 Project Structure
```
zends_ai_copilot/
│
├── src/
│   ├── data_generator.py      # Synthetic dataset generation
│   ├── train.py               # DistilBERT model training (Colab)
│   ├── sentiment.py           # Sentiment analysis
│   ├── rag.py                 # FAISS vector DB + retrieval
│   └── llm_response.py        # Response generation
│
├── models/
│   ├── intent_model/          # Fine-tuned DistilBERT model
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── label_encoder.pkl      # Intent label encoder
│
├── data/
│   └── zends_dataset_v2.csv   # Generated synthetic dataset
│
├── app.py                     # Streamlit web application
├── config.py                  # Configuration & settings
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

---

## 🔄 Project Workflow
```
Step 1: Synthetic Dataset Generation (20,000 records)
        ↓
Step 2: EDA — Intent, Sentiment, Text Length Analysis
        ↓
Step 3: DistilBERT Fine-Tuning (Google Colab GPU)
        ↓
Step 4: Sentiment Analysis (HuggingFace Pre-trained)
        ↓
Step 5: RAG Pipeline (FAISS Vector DB + Retrieval)
        ↓
Step 6: Response Generation
        ↓
Step 7: Streamlit Web App Deployment
```

---

## 📊 Dataset Details

| Property | Value |
|----------|-------|
| Total Records | 20,000 |
| Intents | 5 (Billing, Refund, Technical, Complaint, Product) |
| Sentiments | 3 (Angry, Neutral, Happy) |
| Type | Synthetic (Template + Entity Injection) |
| Format | CSV |

---

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **100%** |
| F1 Score (Weighted) | **100%** |
| Precision | **100%** |
| Recall | **100%** |

### Classification Report:

| Intent | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Billing | 1.00 | 1.00 | 1.00 |
| Complaint | 1.00 | 1.00 | 1.00 |
| Product | 1.00 | 1.00 | 1.00 |
| Refund | 1.00 | 1.00 | 1.00 |
| Technical | 1.00 | 1.00 | 1.00 |

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/zends-ai-copilot.git
cd zends-ai-copilot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

### 5. Open Browser
```
http://localhost:8501
```

---

## ⚠️ Training Note

> Model training requires **GPU** and is done in **Google Colab**.
> Pre-trained model files are saved in `models/intent_model/`.
> No need to retrain — just run `streamlit run app.py` directly.

---

## 💡 Features

- ✅ Real-time intent detection
- ✅ Sentiment classification (Angry / Neutral / Happy)
- ✅ RAG-based policy retrieval
- ✅ Context-aware response generation
- ✅ Chat history with clear option
- ✅ Clean and interactive UI

---

## 📈 Business Use Cases

- 🚀 Reduce customer support response time
- 👥 Help agents handle large query volumes
- 📋 Ensure policy-compliant communication
- 😊 Improve customer satisfaction
- 🏢 Enterprise AI simulation without real data

---

Google Colab Notebook

Training notebook available here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OBL6i_mxsR8PtDa4pZeYaFgpRhJrimEe?usp=sharing)

## 🙋 Author

**Tamilarasi Rajendran**
- Domain: Telecom AI / NLP
- Project: ZENDS AI Customer Support Copilot

---

## 📄 License

This project is for educational purposes.
=======
# zends-ai-copilot
>>>>>>> 8553c651e7687f526b6ef77344337561427ade3a
