import re
from datetime import datetime
from collections import defaultdict

def parse_chat_file(file_path):
    chat_data = defaultdict(list)
    date_pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2}\s?(?:AM|PM)?)\]'
    user_pattern = r'(.*?):'
    system_message_patterns = [
        r'.*joined from the community',
        r'.*requested to join',
        r'.*joined using this group\'s invite link',
        r'.*added .*',
        r'.*changed the subject to ".*"',
        r'.*changed this group\'s icon',
        r'.*changed the group description',
        r'.*changed the group settings to allow .*',
        r'.*changed the group settings to only allow .*',
        r'.*created group ".*"'
    ]
    important_system_messages = [
        r'.*removed .*',
        r'.*left'
    ]

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
                    
                    message_content = line[date_match.end():].strip()
                    
                    # Check if the message is a system message to be ignored
                    if any(re.match(pattern, message_content) for pattern in system_message_patterns):
                        continue  # Skip these system messages
                    
                    # Check if the message is an important system message
                    if any(re.match(pattern, message_content) for pattern in important_system_messages):
                        chat_data['SYSTEM'].append({
                            'timestamp': timestamp,
                            'message': message_content,
                            'line_number': line_number
                        })
                        continue
                    
                    user_match = re.match(user_pattern, message_content)
                    if user_match:
                        user = user_match.group(1).strip()
                        message = message_content[user_match.end():].strip()
                        
                        chat_data[user].append({
                            'timestamp': timestamp,
                            'message': message,
                            'line_number': line_number
                        })
                else:
                    # Handle lines without a timestamp (e.g., continued messages)
                    if chat_data:
                        last_user = list(chat_data.keys())[-1]
                        chat_data[last_user][-1]['message'] += f" {line.strip()}"
            except Exception as e:
                print(f"Warning: Error processing line {line_number}: {e}")

    return dict(chat_data)

def get_chat_statistics(chat_data):
    stats = {
        'total_messages': sum(len(messages) for messages in chat_data.values()),
        'user_message_counts': {user: len(messages) for user, messages in chat_data.items()},
        'date_range': {
            'start': min(message['timestamp'] for messages in chat_data.values() for message in messages),
            'end': max(message['timestamp'] for messages in chat_data.values() for message in messages)
        }
    }
    return stats

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
        print("Messages per user:")
        for user, count in stats['user_message_counts'].items():
            print(f"  {user}: {count}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        
        # Optionally, you can uncomment the following lines to save the parsed data to a JSON file
        with open('parsed_chat_data.json', 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, indent=2, default=str)
        # print("Parsed data saved to 'parsed_chat_data.json'")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
