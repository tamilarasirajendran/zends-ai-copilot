# Main Streamlit App for ZENDS AI Copilot
# Imports
import streamlit as st # UI
import torch # Deep Learning
import pickle # file loading
from transformers import AutoTokenizer, AutoModelForSequenceClassification # NLP Models
from src.rag import retrieve # connect to othere modules
from src.sentiment import analyze_sentiment # connect to othere modules
from src.llm_response import generate_response # connect to othere modules
from config import MODEL_PATH, LABEL_ENCODER_PATH # Configurations

# ===== Page Config =====
# App title + icon
st.set_page_config(
    page_title="ZENDS AI Copilot",
    page_icon="🤖",
    layout="centered"
)

# ===== Custom CSS =====
# I customized the UI using CSS inside Streamlit.
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
# App heading
st.markdown('<div class="title">🤖 ZENDS AI Customer Support</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by DistilBERT + RAG</div>', unsafe_allow_html=True)
st.markdown("---")

# ===== Load Model =====
# I used caching to improve performance by avoiding repeated model loading.
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

# Initialize Model
model, tokenizer, le = load_model()

# ===== Chat History =====
# I used session state to maintain chat history.
if "history" not in st.session_state:
    st.session_state.history = []

# ===== Input =====
# Input + button
query = st.text_input("💬 Type your query here:", placeholder="e.g. My internet is not working...")
submit = st.button("Send 🚀")

if submit and query.strip() != "":
    # Intent Prediction
    # I used DistilBERT to predict the intent of the query.
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True) # Query → tokens
    with torch.no_grad():
        outputs = model(**inputs) # Prediction
    pred = outputs.logits.argmax().item()
    intent = le.inverse_transform([pred])[0] #Number → intent label

    # Sentiment
    sentiment = analyze_sentiment(query) # detect the emotion 

    # Response
    response = generate_response(query) # Generate response using RAG + LLM

    # Save to history
    st.session_state.history.append({
        "query": query,
        "intent": intent,
        "sentiment": sentiment,
        "response": response
    })

# ===== Show Chat History =====
# I displayed intent and sentiment along with the response for better user understanding.
if st.session_state.history:  # If chats exist
    st.markdown("### 💬 Conversation History")
    for chat in reversed(st.session_state.history): # Latest first show 

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
    # Reset chat history when clicked
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()