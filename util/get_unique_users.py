import csv
from collections import Counter

def parse_chat_csv(input_file, output_file):
    unique_users = set()
    user_message_count = Counter()

    # Read input CSV and extract unique users
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            user = row['user'].strip()
            if user:
                unique_users.add(user)
                user_message_count[user] += 1

    # Write unique users to output CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['user', 'message_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for user in sorted(unique_users):
            writer.writerow({'user': user, 'message_count': user_message_count[user]})

    print(f"Found {len(unique_users)} unique users. Results written to {output_file}")

# Usage
input_file = '../chats/formatted_chat.csv'
output_file = '../chats/unique_users.csv'
parse_chat_csv(input_file, output_file)