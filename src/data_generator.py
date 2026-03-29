import pandas as pd
import random

intents = ["billing", "refund", "technical", "complaint", "product"]

templates = {
    "billing": [
        "Why is my bill so high for {product}?",
        "I was overcharged for my {product}",
        "Explain my latest bill for {product}",
        "Billing amount seems incorrect"
    ],
    "refund": [
        "I want a refund for {product}",
        "When will I get my refund?",
        "Refund not processed yet",
        "Cancel and refund my payment"
    ],
    "technical": [
        "My {product} is not working",
        "Facing network issue with {product}",
        "Slow speed problem in {product}",
        "Connection keeps dropping"
    ],
    "complaint": [
        "Very bad service for {product}",
        "I am not happy with your service",
        "Worst experience ever",
        "Customer support is terrible"
    ],
    "product": [
        "Tell me about {product}",
        "What is the price of {product}?",
        "Explain features of {product}",
        "Do you have unlimited plans?"
    ]
}

products = [
    "5G mobile plan",
    "fiber broadband",
    "cloud storage",
    "IoT device",
    "enterprise network"
]

def assign_sentiment(intent):
    if intent in ["complaint", "refund"]:
        return random.choice(["angry", "neutral"])
    elif intent == "billing":
        return random.choice(["neutral", "angry"])
    else:
        return random.choice(["happy", "neutral"])

def generate_dataset(n_samples=20000):
    data = []
    for _ in range(n_samples):
        intent = random.choice(intents)
        template = random.choice(templates[intent])
        product = random.choice(products)
        text = template.format(product=product)
        sentiment = assign_sentiment(intent)
        data.append({
            "text": text,
            "intent": intent,
            "sentiment": sentiment
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset(20000)
    df.to_csv("data/zends_dataset_v2.csv", index=False)
    print("✅ Dataset saved to data/zends_dataset_v2.csv")
