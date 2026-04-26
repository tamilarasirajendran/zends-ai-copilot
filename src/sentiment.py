# Detect user emotion (sentiment) from the query
# I used a pre-trained sentiment analysis model to classify user emotions as positive, negative, or neutral.

from transformers import pipeline

# Create Sentiment Pipeline
# I used a pretrained sentiment analysis model to classify user emotions.

sentiment_pipeline = pipeline("sentiment-analysis")

#The pipeline returns sentiment labels like POSITIVE or NEGATIVE.
# The user query is passed to a pretrained sentiment model, which returns a label and confidence score. Based on this, 
# I classify the emotion as angry, happy, or neutral.

def analyze_sentiment(query):  #Takes user input
    result = sentiment_pipeline(query)[0]
    label = result['label']
    score = result['score']

    if label == "NEGATIVE" and score > 0.75:
        return "angry"
    elif label == "POSITIVE" and score > 0.75:
        return "happy"
    else:
        return "neutral"