import spacy
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import warnings
import os
from datetime import datetime
from itertools import combinations

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Force CPU usage
def setup_device():
    spacy.require_cpu()
    print("Using CPU for both PyTorch and spaCy")
    return 'cpu'

# device = setup_device()
device = torch.device("mps") 

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

# Update SentimentAnalyzer to use CPU if GPU fails
class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):   
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # self.device = 'cpu'
        self.device = 'mps'
        self.model.to(self.device)

    def calculate_sentiment(self, doc):
        text = " ".join([token.text for token in doc])
        try:
            encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                output = self.model(**encoded_input)
            scores = output.logits[0].cpu().numpy()
            scores = softmax(scores)
            sentiment_score = scores[1] - scores[0]
            if sentiment_score > 0.1:
                return "Positive", sentiment_score
            elif sentiment_score < -0.1:
                return "Negative", sentiment_score
            else:
                return "Neutral", sentiment_score
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return "Error", 0.0

sentiment_analyzer = SentimentAnalyzer()

def read_csv_file(file_path):
    chat_data = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            date_time_str = row['date_time']
            user = row['user']
            message = row['message']

            try:
                timestamp = datetime.strptime(date_time_str, "%m/%d/%y %I:%M:%S %p")
            except ValueError:
                timestamp = datetime.strptime(date_time_str, "%m/%d/%Y %I:%M:%S %p")

            chat_data[user].append({
                'timestamp': timestamp,
                'message': message
            })
    return dict(chat_data)

def extract_interests(user_messages, top_n=10):
    # Combine all messages for the user
    text = " ".join([msg['message'] for msg in user_messages])

    # Process the text with spaCy
    doc = nlp(text)

    # Extract nouns, proper nouns, and entities as potential interests
    interests = [token.text.lower() for token in doc if token.pos_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT', 'NOUN', 'PROPN', 'GPE', 'LOC', 'MISC']]
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

    # Use the sentiment analysis
    sentiment = calculate_sentiment(doc)

    return {
        'top_words': top_words,
        'sentiment': sentiment,
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': vectorizer
    }

def analyze_chat(csv_file_path):
    chat_data = read_csv_file(csv_file_path)
    all_user_interests = {user: extract_interests(messages) for user, messages in chat_data.items()}
    group_interests = identify_group_interests(all_user_interests)
    user_clusters = cluster_users(all_user_interests)
    
    results = {
        'user_interests': all_user_interests,
        'group_interests': group_interests,
        'user_clusters': user_clusters,
    }
    
    # Calculate user similarities
    user_similarities = {}
    for user1 in all_user_interests:
        for user2 in all_user_interests:
            if user1 != user2:
                try:
                    similarity = calculate_user_similarity(all_user_interests[user1], all_user_interests[user2])
                    user_similarities[(user1, user2)] = similarity
                except Exception as e:
                    print(f"Error calculating similarity between {user1} and {user2}: {str(e)}")
    
    results['user_similarities'] = user_similarities
    
    # Score users by each group interest
    interest_scores = {}
    for interest in group_interests:
        interest_scores[interest] = score_users_by_interest(all_user_interests, interest)
    
    results['interest_scores'] = interest_scores
    
    return results

def calculate_sentiment(doc):
    sentiment, score = sentiment_analyzer.calculate_sentiment(doc)
    return f"{sentiment} (score: {score:.2f})"

def calculate_user_similarity(user1_interests, user2_interests):
    # Align feature spaces
    user1_matrix, user2_matrix = align_feature_spaces(user1_interests, user2_interests)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(user1_matrix, user2_matrix)
    
    return similarity[0][0]

def align_feature_spaces(user1_interests, user2_interests):
    # Combine vocabularies
    combined_vocab = set(user1_interests['vectorizer'].get_feature_names_out()) | set(user2_interests['vectorizer'].get_feature_names_out())
    
    # Create a new vectorizer with the combined vocabulary
    combined_vectorizer = TfidfVectorizer(vocabulary=combined_vocab)
    
    # Transform both users' interests using the combined vocabulary
    user1_matrix = combined_vectorizer.fit_transform(user1_interests['vectorizer'].get_feature_names_out())
    user2_matrix = combined_vectorizer.transform(user2_interests['vectorizer'].get_feature_names_out())
    
    return user1_matrix, user2_matrix

def identify_group_interests(all_user_interests, top_n=10):
    combined_interests = Counter()
    for user_interests in all_user_interests.values():
        combined_interests.update(user_interests['top_words'])
    return [word for word, _ in combined_interests.most_common(top_n)]

def score_users_by_interest(all_user_interests, interest):
    scores = {}
    for user, interests in all_user_interests.items():
        if interest in interests['top_words']:
            score = interests['top_words'].index(interest)
            scores[user] = 1 / (score + 1)  # Higher score for earlier appearance
        else:
            scores[user] = 0
    return scores

def suggest_new_interests(user_interests, all_user_interests, top_n=5):
    user_vector = user_interests['tfidf_matrix']
    all_interests = set()
    for interests in all_user_interests.values():
        all_interests.update(interests['top_words'])
    
    new_interests = all_interests - set(user_interests['top_words'])
    
    vectorizer = user_interests['vectorizer']
    new_interest_vectors = vectorizer.transform(new_interests)
    
    similarities = cosine_similarity(user_vector, new_interest_vectors)
    
    top_suggestions = sorted(zip(new_interests, similarities[0]), key=lambda x: x[1], reverse=True)[:top_n]
    return [interest for interest, score in top_suggestions]

def cluster_users(all_user_interests):
    # Get the maximum number of features across all users
    max_features = max(interests['tfidf_matrix'].shape[1] for interests in all_user_interests.values())

    # Pad each user's TF-IDF matrix to have the same number of features
    padded_matrices = []
    for interests in all_user_interests.values():
        matrix = interests['tfidf_matrix'].toarray()
        padded_matrix = np.pad(matrix, ((0, 0), (0, max_features - matrix.shape[1])), mode='constant')
        padded_matrices.append(padded_matrix)

    tfidf_matrix = np.vstack(padded_matrices)

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    return dict(zip(all_user_interests.keys(), cluster_labels))

def export_user_interests_to_csv(user_interests, output_file='user_interests.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['User', 'Top Words', 'Sentiment', 'Suggested Interests'])
        for user, interests in user_interests.items():
            suggested = suggest_new_interests(interests, user_interests)
            writer.writerow([user, ', '.join(interests['top_words']), interests['sentiment'], ', '.join(suggested)])
    print(f"User interests exported to {output_file}")

def analyze_chat(chat_data):
    all_user_interests = {user: extract_interests(messages) for user, messages in chat_data.items()}
    group_interests = identify_group_interests(all_user_interests)
    user_clusters = cluster_users(all_user_interests)
    
    results = {
        'user_interests': all_user_interests,
        'group_interests': group_interests,
        'user_clusters': user_clusters,
    }
    
    # Calculate user similarities
    user_similarities = {}
    for user1 in all_user_interests:
        for user2 in all_user_interests:
            if user1 != user2:
                try:
                    similarity = calculate_user_similarity(all_user_interests[user1], all_user_interests[user2])
                    user_similarities[(user1, user2)] = similarity
                except Exception as e:
                    print(f"Error calculating similarity between {user1} and {user2}: {str(e)}")
    
    results['user_similarities'] = user_similarities
    
    # Score users by each group interest
    interest_scores = {}
    for interest in group_interests:
        interest_scores[interest] = score_users_by_interest(all_user_interests, interest)
    
    results['interest_scores'] = interest_scores
    
    return results

# Example usage
# chat_data = parse_chat_file('chat.txt')  # You need to implement this function
# results = analyze_chat(chat_data)
# export_user_interests_to_csv(results['user_interests'])

def cleanup():
    torch.cuda.empty_cache()

if __name__ == "__main__":
    csv_file_path = './chats/formatted_chat.csv'  # Update this path to match your CSV file location
    results = analyze_chat(read_csv_file(csv_file_path))
    export_user_interests_to_csv(results['user_interests'])
    
    # Call cleanup at the end
    cleanup()
