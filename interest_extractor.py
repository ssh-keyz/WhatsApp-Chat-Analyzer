import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def extract_interests(user_messages, top_n=10):
    # Combine all messages for the user
    text = " ".join([msg['message'] for msg in user_messages])
    
    # Tokenize and get model output
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class (positive or negative sentiment)
    predicted_class = torch.argmax(outputs.logits).item()
    
    # Use TF-IDF to identify important words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform([text])
    
    # Get the top N words based on TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    word_scores = list(zip(feature_names, tfidf_scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    top_words = [word for word, score in word_scores[:top_n]]
    
    return {
        'top_words': top_words,
        'sentiment': 'positive' if predicted_class == 1 else 'negative'
    }

def calculate_user_similarity(user1_interests, user2_interests):
    # Create binary vectors for each user's interests
    all_interests = list(set(user1_interests['top_words'] + user2_interests['top_words']))
    vector1 = [1 if word in user1_interests['top_words'] else 0 for word in all_interests]
    vector2 = [1 if word in user2_interests['top_words'] else 0 for word in all_interests]
    
    # Calculate cosine similarity
    return cosine_similarity([vector1], [vector2])[0][0]

def analyze_chat_data(chat_data):
    user_interests = {}
    for user, messages in chat_data.items():
        user_interests[user] = extract_interests(messages)
    
    # Calculate similarities between users
    similarities = {}
    users = list(user_interests.keys())
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            user1, user2 = users[i], users[j]
            similarity = calculate_user_similarity(user_interests[user1], user_interests[user2])
            similarities[(user1, user2)] = similarity
    
    return user_interests, similarities

# Usage
chat_data = parse_chat_file('chat_file.txt')  # Assuming this function is available from the previous implementation
user_interests, user_similarities = analyze_chat_data(chat_data)

# Print results
for user, interests in user_interests.items():
    print(f"User: {user}")
    print(f"Top interests: {', '.join(interests['top_words'])}")
    print(f"Overall sentiment: {interests['sentiment']}")
    print()

print("User Similarities:")
for (user1, user2), similarity in user_similarities.items():
    print(f"{user1} - {user2}: {similarity:.2f}")
