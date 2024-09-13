import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Replace spacy.prefer_gpu() with:
if torch.backends.mps.is_available():
    spacy.require_gpu()
    torch.set_default_device('mps')
elif torch.cuda.is_available():
    spacy.require_gpu()
    torch.set_default_device('cuda')
else:
    spacy.require_cpu()

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

    def calculate_sentiment(self, doc):
        text = " ".join([token.text for token in doc])
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        scores = output.logits[0].cpu().numpy()
        scores = softmax(scores)
        
        sentiment_score = scores[1] - scores[0]
        
        if sentiment_score > 0.2:
            return "Positive", sentiment_score
        elif sentiment_score < -0.2:
            return "Negative", sentiment_score
        else:
            return "Neutral", sentiment_score

sentiment_analyzer = SentimentAnalyzer()

def extract_interests(user_messages, top_n=10):
    # Combine all messages for the user
    text = " ".join([msg['message'] for msg in user_messages])

    # Process the text with spaCy
    doc = nlp(text)

    # Extract nouns, proper nouns, and entities as potential interests
    interests = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    interests.extend([ent.text.lower() for ent in doc.ents])

    # Use TF-IDF to identify important words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform([text])

    # Get the top N words based on TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    word_scores = list(zip(feature_names, tfidf_scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    top_words = [word for word, score in word_scores[:top_n]]

    # Use the new sentiment analysis
    sentiment = calculate_sentiment(doc)

    return {
        'top_words': top_words,
        'sentiment': sentiment
    }

def calculate_sentiment(doc):
    sentiment, score = sentiment_analyzer.calculate_sentiment(doc)
    return f"{sentiment} (score: {score:.2f})"

def calculate_user_similarity(user1_interests, user2_interests):
    # This function remains unchanged
    all_interests = list(set(user1_interests['top_words'] + user2_interests['top_words']))
    vector1 = [1 if word in user1_interests['top_words'] else 0 for word in all_interests]
    vector2 = [1 if word in user2_interests['top_words'] else 0 for word in all_interests]

    return cosine_similarity([vector1], [vector2])[0][0]

def export_user_interests_to_csv(user_interests, output_file='user_interests.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['User', 'Top Words', 'Sentiment'])
        for user, interests in user_interests.items():
            writer.writerow([user, ', '.join(interests['top_words']), interests['sentiment']])
    print(f"User interests exported to {output_file}")
