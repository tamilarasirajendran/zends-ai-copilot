from transformers import pipeline

# Create Sentiment Pipeline
# I used a pretrained sentiment analysis model to classify user emotions.
sentiment_pipeline = pipeline("sentiment-analysis")

#The pipeline returns sentiment labels like POSITIVE or NEGATIVE.
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
