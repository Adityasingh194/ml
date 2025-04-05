import os
import time
import json
import pandas as pd
from transformers import pipeline
from pymongo import MongoClient
import spacy
import re
from collections import Counter

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# Connect to MongoDB Atlas
MONGO_URI = os.getenv("mongodb+srv://sadityakumar194:12345@cluster0.hdmpeoz.mongodb.net/") or "your_mongo_uri_here"
client = MongoClient(MONGO_URI)
db = client["chiler"]
collection = db["reviews"]

# Load sentiment analysis model
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'(?:#\w+\s*)+$', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(texts):
    words = []
    for text in texts:
        doc = nlp(text)
        words += [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"]]
    counter = Counter(words)
    most_common = counter.most_common(5)
    return [word for word, count in most_common]

def analyze_sentiment(texts):
    results = classifier(texts)
    sentiments = [r["label"].lower() for r in results]
    return sentiments

def write_json(sentiment_counts, keywords):
    output = {
        "sentiments": sentiment_counts,
        "keywords": keywords
    }
    os.makedirs("public/data", exist_ok=True)
    with open("public/data/sentiment.json", "w") as f:
        json.dump(output, f)
    print("✅ Updated public/data/sentiment.json")

# Main loop
while True:
    try:
        print("🔄 Running sentiment pipeline...")
        docs = list(collection.find().sort("createdAt", -1).limit(200))
        if not docs:
            print("⚠️ No data found in MongoDB")
            time.sleep(60)
            continue

        texts = [doc["Content"] for doc in docs if "Content" in doc]
        cleaned = [clean_text(t) for t in texts]
        sentiments = analyze_sentiment(cleaned)

        sentiment_counts = {
            "positive": sentiments.count("positive"),
            "neutral": sentiments.count("neutral"),
            "negative": sentiments.count("negative")
        }

        top_keywords = extract_keywords(cleaned)

        write_json(sentiment_counts, top_keywords)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("⏳ Sleeping for 60 seconds...")
    time.sleep(60)

