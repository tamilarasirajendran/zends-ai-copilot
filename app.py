import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.rag import retrieve
from src.sentiment import analyze_sentiment
from src.llm_response import generate_response
from config import MODEL_PATH, LABEL_ENCODER_PATH

# ===== Page Config =====
st.set_page_config(
    page_title="ZENDS AI Copilot",
    page_icon="🤖",
    layout="centered"
)

# ===== Custom CSS =====
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .title { text-align: center; color: #1a1a2e; font-size: 2.5rem; font-weight: bold; }
    .subtitle { text-align: center; color: #555; margin-bottom: 30px; }
    .chat-box {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .user-msg {
        background-color: #dfe6fd;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        text-align: right;
        font-weight: bold;
        color: #1a1a2e;
    }
    .bot-msg {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        color: #333;
    }
    .intent-badge {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-right: 8px;
    }
    .sentiment-badge {
        display: inline-block;
        background-color: #FF5722;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Title =====
st.markdown('<div class="title">🤖 ZENDS AI Customer Support</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by DistilBERT + RAG</div>', unsafe_allow_html=True)
st.markdown("---")

# ===== Load Model =====
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

model, tokenizer, le = load_model()

# ===== Chat History =====
if "history" not in st.session_state:
    st.session_state.history = []

# ===== Input =====
query = st.text_input("💬 Type your query here:", placeholder="e.g. My internet is not working...")
submit = st.button("Send 🚀")

if submit and query.strip() != "":
    # Intent
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax().item()
    intent = le.inverse_transform([pred])[0]

    # Sentiment
    sentiment = analyze_sentiment(query)

    # Response
    response = generate_response(query)

    # Save to history
    st.session_state.history.append({
        "query": query,
        "intent": intent,
        "sentiment": sentiment,
        "response": response
    })

# ===== Show Chat History =====
if st.session_state.history:
    st.markdown("### 💬 Conversation History")
    for chat in reversed(st.session_state.history):

        # Sentiment color
        sentiment_color = "#FF5722" if chat['sentiment'] == "angry" else "#4CAF50" if chat['sentiment'] == "happy" else "#2196F3"

        st.markdown(f"""
        <div class="chat-box">
            <div class="user-msg">🧑 {chat['query']}</div>
            <div style="margin: 8px 0;">
                <span class="intent-badge">🎯 {chat['intent']}</span>
                <span style="display:inline-block; background-color:{sentiment_color};
                color:white; padding:4px 12px; border-radius:20px; font-size:0.85rem;">
                😊 {chat['sentiment']}</span>
            </div>
            <div class="bot-msg">🤖 {chat['response']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Clear button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()