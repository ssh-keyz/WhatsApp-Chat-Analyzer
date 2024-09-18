import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Replace spacy.prefer_gpu() with:
# if torch.backends.mps.is_available():
#     spacy.require_gpu()
#     torch.set_default_device('mps')
# elif torch.cuda.is_available():
#     spacy.require_gpu()
#     torch.set_default_device('cuda')
# else:
#     spacy.require_cpu()
torch.set_default_device('mps')
nlp = spacy.load("en_core_web_trf")

def calculate_user_similarity(user1_interests, user2_interests):
    """
    Calculate the similarity between two users based on their interests.

    :param user1_interests: dict containing 'top_words' and 'sentiment' for user1
    :param user2_interests: dict containing 'top_words' and 'sentiment' for user2
    :return: float representing the similarity score
    """
    # Combine top words from both users
    all_words = set(user1_interests['top_words'] + user2_interests['top_words'])

    # Create binary vectors for each user's interests
    vector1 = [1 if word in user1_interests['top_words'] else 0 for word in all_words]
    vector2 = [1 if word in user2_interests['top_words'] else 0 for word in all_words]

    # Calculate cosine similarity
    similarity = cosine_similarity([vector1], [vector2])[0][0]

    # Adjust similarity based on sentiment
    sentiment_factor = 1 if user1_interests['sentiment'] == user2_interests['sentiment'] else 0.8

    return similarity * sentiment_factor

def calculate_user_similarity_tfidf(user1_messages, user2_messages):
    # Process messages with spaCy
    user1_text = ' '.join([token.lemma_ for doc in nlp.pipe(user1_messages) for token in doc if not token.is_stop and not token.is_punct])
    user2_text = ' '.join([token.lemma_ for doc in nlp.pipe(user2_messages) for token in doc if not token.is_stop and not token.is_punct])

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user1_text, user2_text])

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity

def calculate_multiple_user_similarities(user_interests):
    """
    Calculate similarities between all pairs of users.

    :param user_interests: dict where keys are usernames and values are interest dicts
    :return: dict of user pairs and their similarity scores
    """
    similarities = {}
    users = list(user_interests.keys())

    for i in range(len(users)):
        for j in range(i+1, len(users)):
            user1, user2 = users[i], users[j]
            similarity = calculate_user_similarity(user_interests[user1], user_interests[user2])
            similarities[(user1, user2)] = similarity

    return similarities

def find_most_similar_users(similarities, top_n=5):
    """
    Find the most similar pairs of users.

    :param similarities: dict of user pairs and their similarity scores
    :param top_n: number of top similar pairs to return
    :return: list of tuples (user1, user2, similarity_score)
    """
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [(users[0], users[1], score) for users, score in sorted_similarities[:top_n]]

if __name__ == "__main__":
    # Example usage
    user_interests = {
        "Alice": {"top_words": ["python", "data", "machine learning"], "sentiment": "positive"},
        "Bob": {"top_words": ["java", "web", "database"], "sentiment": "neutral"},
        "Charlie": {"top_words": ["python", "web", "api"], "sentiment": "positive"}
    }

    similarities = calculate_multiple_user_similarities(user_interests)
    print("User Similarities:")
    for (user1, user2), score in similarities.items():
        print(f"{user1} - {user2}: {score:.2f}")

    print("\nMost Similar Users:")
    for user1, user2, score in find_most_similar_users(similarities):
        print(f"{user1} and {user2}: {score:.2f}")

    # Example with TF-IDF
    user1_messages = ["I love Python programming", "Data science is fascinating"]
    user2_messages = ["Java is my favorite language", "Web development is fun"]
    tfidf_similarity = calculate_user_similarity_tfidf(user1_messages, user2_messages)
    print(f"\nTF-IDF Similarity between user1 and user2: {tfidf_similarity:.2f}")
