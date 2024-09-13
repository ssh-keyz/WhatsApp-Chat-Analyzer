# python chat_parser.py <path_to_your_chat_log>
import re
from datetime import datetime
from collections import defaultdict
import emoji

def parse_chat_file(file_path):
    chat_data = defaultdict(list)
    date_pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2}\s?(?:AM|PM)?)\]'
    user_pattern = r'(.*?):'

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            try:
                date_match = re.match(date_pattern, line)
                if date_match:
                    date_str, time_str = date_match.groups()
                    
                    # Handle different date formats
                    try:
                        if len(date_str.split('/')[-1]) == 2:
                            date_format = "%m/%d/%y"
                        else:
                            date_format = "%m/%d/%Y"
                        
                        # Handle different time formats
                        if 'AM' in time_str or 'PM' in time_str:
                            time_format = "%I:%M:%S %p"
                        else:
                            time_format = "%H:%M:%S"
                        
                        timestamp = datetime.strptime(f"{date_str} {time_str}", f"{date_format} {time_format}")
                    except ValueError as e:
                        print(f"Warning: Unable to parse date/time on line {line_number}: {e}")
                        continue
                    
                    user_match = re.search(user_pattern, line[date_match.end():].strip())
                    if user_match:
                        user = user_match.group(1).strip()
                        message = line[date_match.end() + user_match.end() + 1:].strip()
                        
                        # Skip messages about joining, adding, or requesting to join
                        if any(action in message.lower() for action in ['joined', 'added', 'requested to join']):
                            continue
                        
                        chat_data[user].append({
                            'timestamp': timestamp,
                            'message': clean_message(message),
                            'line_number': line_number
                        })
                    else:
                        # Handle system messages or other non-user messages
                        message = line[date_match.end():].strip()
                        
                        # Skip system messages about joining, adding, or requesting to join
                        if any(action in message.lower() for action in ['joined', 'added', 'requested to join']):
                            continue
                        
                        chat_data['SYSTEM'].append({
                            'timestamp': timestamp,
                            'message': clean_message(message),
                            'line_number': line_number
                        })
                else:
                    # Handle lines without a timestamp (e.g., continued messages)
                    if chat_data:
                        last_user = list(chat_data.keys())[-1]
                        chat_data[last_user][-1]['message'] += f" {clean_message(line.strip())}"
            except Exception as e:
                print(f"Warning: Error processing line {line_number}: {e}")

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
