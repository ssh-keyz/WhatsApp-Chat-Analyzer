import sys
import csv
import json
from collections import Counter
from chat_parser import parse_chat_file
from interest_extractor import extract_interests, calculate_sentiment
from similarity_calculator import calculate_user_similarity
from interest_extractor import export_user_interests_to_csv
from collections import defaultdict
import spacy

# Load the larger spaCy model at the beginning of the script
nlp = spacy.load("en_core_web_trf")

def analyze_sentiment_distribution(chat_data):
    sentiments = []
    for user, messages in chat_data.items():
        for msg in messages:
            doc = nlp(msg['message'])  # Process the text with spaCy
            sentiment = calculate_sentiment(doc)  # Pass the spaCy Doc object
            sentiments.append(sentiment)
    return Counter(sentiments)

def export_sentiment_distribution(distribution, output_file='sentiment_distribution.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sentiment', 'Count'])
        for sentiment, count in distribution.items():
            writer.writerow([sentiment, count])
    print(f"Sentiment distribution exported to {output_file}")

def analyze_chat_data(chat_data):
    user_interests = {}
    for user, messages in chat_data.items():
        if messages:  # Check if the user has any messages
            user_interests[user] = extract_interests(messages)
        else:
            user_interests[user] = {'top_words': [], 'sentiment': 'N/A'}  # Add users with no messages

    similarities = {}
    users = list(user_interests.keys())
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            user1, user2 = users[i], users[j]
            similarity = calculate_user_similarity(user_interests[user1], user_interests[user2])
            similarities[(user1, user2)] = similarity

    sentiment_distribution = analyze_sentiment_distribution(chat_data)

    return user_interests, similarities, sentiment_distribution

def print_results(user_interests, similarities):
    print("User Interests:")
    for user, interests in user_interests.items():
        print(f"\nUser: {user}")
        print(f"Top interests: {', '.join(interests['top_words'])}")
        print(f"Overall sentiment: {interests['sentiment']}")

    print("\nUser Similarities:")
    for (user1, user2), similarity in similarities.items():
        print(f"{user1} - {user2}: {similarity:.2f}")

def export_results(user_interests, similarities, output_file):
    results = {
        "user_interests": user_interests,
        "user_similarities": {f"{u1}-{u2}": sim for (u1, u2), sim in similarities.items()}
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to {output_file}")

def export_similarities_to_csv(similarities, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['User1', 'User2', 'Similarity'])
        for (user1, user2), similarity in similarities.items():
            csvwriter.writerow([user1, user2, f"{similarity:.2f}"])
    print(f"\nSimilarity scores exported to {output_file}")

def export_user_interests_to_csv(user_interests, output_file='user_interests.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['User', 'Top Words', 'Sentiment'])
        for user, interests in user_interests.items():
            writer.writerow([user, ', '.join(interests.get('top_words', [])), interests.get('sentiment', 'N/A')])
    print(f"User interests exported to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_chat.py <chat_file_path> [output_file_path]")
        sys.exit(1)

    chat_file_path = sys.argv[1]
    output_file_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        chat_data = parse_chat_file(chat_file_path)
        user_interests, similarities, sentiment_distribution = analyze_chat_data(chat_data)

        print_results(user_interests, similarities)
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_distribution.items():
            print(f"{sentiment}: {count}")

        if output_file_path:
            export_results(user_interests, similarities, output_file_path)

        # Export similarity scores to CSV
        export_similarities_to_csv(similarities, 'user_similarities.csv')
        # Export user interests to CSV
        export_user_interests_to_csv(user_interests, 'user_interests.csv')
        # Export sentiment distribution to CSV
        export_sentiment_distribution(sentiment_distribution, 'sentiment_distribution.csv')

    except FileNotFoundError:
        print(f"Error: File '{chat_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
