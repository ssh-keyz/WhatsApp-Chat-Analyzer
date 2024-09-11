import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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
