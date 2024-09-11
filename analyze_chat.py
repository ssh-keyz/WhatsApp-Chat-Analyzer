import sys
from chat_parser import parse_chat_file
from interest_extractor import extract_interests, calculate_user_similarity
from collections import defaultdict
import json

def analyze_chat_data(chat_data):
    user_interests = {}
    for user, messages in chat_data.items():
        user_interests[user] = extract_interests(messages)

    similarities = {}
    users = list(user_interests.keys())
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            user1, user2 = users[i], users[j]
            similarity = calculate_user_similarity(user_interests[user1], user_interests[user2])
            similarities[(user1, user2)] = similarity

    return user_interests, similarities

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_chat.py <chat_file_path> [output_file_path]")
        sys.exit(1)

    chat_file_path = sys.argv[1]
    output_file_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        chat_data = parse_chat_file(chat_file_path)
        user_interests, similarities = analyze_chat_data(chat_data)

        print_results(user_interests, similarities)

        if output_file_path:
            export_results(user_interests, similarities, output_file_path)

    except FileNotFoundError:
        print(f"Error: File '{chat_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
