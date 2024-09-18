import csv
import re
from datetime import datetime
from collections import defaultdict
import emoji

def parse_chat_file(file_path):
    chat_data = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            try:
                date_time_str = row['date_time']
                user = row['user']
                message = row['message']

                # Parse the date_time string
                try:
                    timestamp = datetime.strptime(date_time_str, "%m/%d/%y %I:%M:%S %p")
                except ValueError:
                    # Try alternative format if the first one fails
                    timestamp = datetime.strptime(date_time_str, "%m/%d/%Y %I:%M:%S %p")

                # Skip messages about joining, adding, or requesting to join
                if any(action in message.lower() for action in ['joined', 'added', 'requested to join']):
                    continue

                chat_data[user].append({
                    'timestamp': timestamp,
                    'message': clean_message(message),
                    'line_number': csv_reader.line_num
                })

            except Exception as e:
                print(f"Warning: Error processing line {csv_reader.line_num}: {e}")

    return dict(chat_data)

def clean_message(message):
    # Remove URLs
    message = re.sub(r'http\S+', '', message)
    # Remove emojis
    message = emoji.replace_emoji(message, '')
    # Remove special characters and extra whitespace
    message = re.sub(r'[^\w\s]', '', message)
    message = ' '.join(message.split())
    return message

def get_chat_statistics(chat_data):
    stats = {
        'total_messages': sum(len(messages) for messages in chat_data.values()),
        'user_message_counts': {user: len(messages) for user, messages in chat_data.items()},
        'date_range': {
            'start': min(message['timestamp'] for messages in chat_data.values() for message in messages),
            'end': max(message['timestamp'] for messages in chat_data.values() for message in messages)
        },
        'active_users': sorted(chat_data.keys(), key=lambda x: len(chat_data[x]), reverse=True),
        'message_frequency': {},
        'most_common_words': get_most_common_words(chat_data)
    }

    # Calculate message frequency
    for user, messages in chat_data.items():
        timestamps = [msg['timestamp'] for msg in messages]
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 for i in range(len(timestamps)-1)]
            stats['message_frequency'][user] = sum(time_diffs) / len(time_diffs)

    return stats

def get_most_common_words(chat_data, top_n=10):
    word_count = defaultdict(int)
    for user, messages in chat_data.items():
        for msg in messages:
            words = msg['message'].lower().split()
            for word in words:
                if len(word) > 3:  # Ignore short words
                    word_count[word] += 1
    return sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:top_n]

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python chat_parser.py <chat_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    try:
        parsed_data = parse_chat_file(file_path)
        stats = get_chat_statistics(parsed_data)
        
        print("Chat parsing completed successfully.")
        print(f"Total messages: {stats['total_messages']}")
        print("\nMessages per user:")
        for user, count in stats['user_message_counts'].items():
            print(f"  {user}: {count}")
        print(f"\nDate range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print("\nTop 5 most active users:")
        for user in stats['active_users'][:5]:
            print(f"  {user}: {stats['user_message_counts'][user]} messages")
        print("\nAverage time between messages (in hours):")
        for user, freq in stats['message_frequency'].items():
            print(f"  {user}: {freq:.2f}")
        print("\nMost common words:")
        for word, count in stats['most_common_words']:
            print(f"  {word}: {count}")
        
        # Save the parsed data to a JSON file
        with open('parsed_chat_data.json', 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, indent=2, default=str)
        print("\nParsed data saved to 'parsed_chat_data.json'")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
