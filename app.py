# ==============================
# 🤖 ZENDS AI Copilot - FINAL APP
# ==============================

import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.sentiment import analyze_sentiment
from src.llm_response import generate_response
from config import MODEL_PATH, LABEL_ENCODER_PATH

# ------------------------------
# 🎨 Page Config
# ------------------------------
st.set_page_config(
    page_title="ZENDS AI Copilot",
    page_icon="🤖",
    layout="centered"
)

# ------------------------------
# 💅 Title
# ------------------------------
st.title("🤖 ZENDS AI Copilot")
st.caption("AI Customer Support Assistant (DistilBERT + RAG)")

# ------------------------------
# ⚡ Load Model
# ------------------------------
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

model, tokenizer, le = load_model()

# ------------------------------
# 🧠 Predict Intent
# ------------------------------
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return le.inverse_transform([pred])[0]

# ------------------------------
# 💬 Chat Memory
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------
# 📝 Input
# ------------------------------
query = st.text_input("💬 Ask your question:", placeholder="e.g. My internet is not working")

# ------------------------------
# 🔄 Processing
# ------------------------------
if st.button("Send 🚀") and query.strip() != "":

    with st.spinner("Thinking... 🤖"):

        intent = predict_intent(query)
        sentiment = analyze_sentiment(query)

        # 🔥 Smart response
        if intent == "refund" and sentiment == "angry":
            response = "I understand your frustration. Don’t worry — your refund request is being processed and should be completed within 5–7 business days."

        elif intent == "billing" and sentiment == "angry":
            response = "I understand your concern about the high bill. It may be due to recent usage or additional charges. Please check your recent transactions, and if anything seems incorrect, I can help you resolve it."

        elif intent == "technical" and sentiment == "angry":
            response = "I’m sorry you're facing this issue. Please restart your router and check if the connection improves. If the problem continues, I can guide you through further troubleshooting."

        else:
            try:
                response = generate_response(query, intent)
                if not response or response.strip() == "":
                    raise ValueError("Empty")
            except:
                response = "Our support team will assist you shortly."

    st.session_state.history.append({
        "query": query,
        "intent": intent,
        "sentiment": sentiment,
        "response": response
    })

# ------------------------------
# 💬 Display Chat (CLEAN UI)
# ------------------------------
if st.session_state.history:

    for chat in st.session_state.history:

        with st.chat_message("user"):
            st.write(chat["query"])

        with st.chat_message("assistant"):
            st.write(chat["response"])
            st.caption(f"🎯 Intent: {chat['intent'].capitalize()}   |   😡 Sentiment: {chat['sentiment'].capitalize()}")

# ------------------------------
# 🧹 Clear Chat
# ------------------------------
if st.button("🗑️ Clear Chat"):
    st.session_state.history = []
    st.rerun()