from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(query):
    result = sentiment_pipeline(query)[0]['label']
    return map_sentiment(result)

def map_sentiment(label):
    if label == "NEGATIVE":
        return "angry"
    elif label == "POSITIVE":
        return "happy"
    else:
        return "neutral"
