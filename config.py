# config.py

MODEL_PATH = "models/intent_model"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
DATA_PATH = "data/zends_dataset_v2.csv"

INTENTS = ["billing", "refund", "technical", "complaint", "product"]

PRODUCTS = [
    "5G mobile plan",
    "fiber broadband",
    "cloud storage",
    "IoT device",
    "enterprise network"
]

DOCUMENTS = [
    "Refund is allowed within 7 days if usage is less than 10 percent.",
    "Billing is monthly and late payment may suspend services.",
    "Technical support is available 24/7 for all services.",
    "Fiber broadband plans include 100 Mbps, 300 Mbps, and 1 Gbps.",
    "If internet is slow, restart your router or contact technical support."
]
