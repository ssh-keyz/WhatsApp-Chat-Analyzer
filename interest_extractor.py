import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import torch

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

    # Simple sentiment analysis
    sentiment = calculate_sentiment(doc)

    return {
        'top_words': top_words,
        'sentiment': sentiment
    }

def calculate_sentiment(doc):
    # Simple rule-based sentiment analysis
    positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy'])
    negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'disappointed'])

    sentiment_score = sum(1 if token.text.lower() in positive_words else -1
                          if token.text.lower() in negative_words else 0
                          for token in doc)

    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

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
