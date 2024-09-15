import re
import csv
from datetime import datetime
from collections import defaultdict
from datetime import datetime
import spacy
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

# Load BERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertModel.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

class ChatAnalyzer:
    def __init__(self):
        self.users = set()
        self.daily_active_users = defaultdict(set)
        self.groups = set()
        self.interests = defaultdict(set)
        self.events = []
        self.messages = []

    def preprocess_chat_to_csv(self, input_file_path, output_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as input_file, \
            open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
            
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(['date_time', 'user', 'message'])  # Write header

            # Updated regex pattern to match the date, time, user, and message
            pattern = r'\[(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}:\d{2}\s[AP]M)\]\s(.+?):\s(.+?)(?=\[|$)'
            
            for line in input_file:
                # Find all matches in the line
                matches = re.finditer(pattern, line)
                for match in matches:
                    date, time, user, message = match.groups()
                    date_time = f"{date} {time}"
                    
                    # Clean and write the message
                    cleaned_message = self.clean_message(message)  # Use self.clean_message instead of clean_message
                    if cleaned_message:  # Only write non-empty messages
                        csv_writer.writerow([date_time, user, cleaned_message])

        print(f"Preprocessed chat data has been written to {output_file_path}")

    def clean_message(self, message):
        # Remove any leading/trailing whitespace
        message = message.strip()
        
        # Remove any "attached" messages
        message = re.sub(r'‎?<attached:.*?>', '', message)
        
        # Remove any other unwanted patterns (add more as needed)
        message = re.sub(r'‎', '', message)  # Remove invisible separator character
        
        return message.strip()

    def parse_chat_file(self, file_path):
        chat_data = defaultdict(list)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row_number, row in enumerate(csv_reader, 2):  # Start at 2 to account for header row
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

                    if user.lower() == 'system':
                        chat_data['SYSTEM'].append({
                            'timestamp': timestamp,
                            'message': self.clean_message(message),
                            'line_number': row_number
                        })
                    else:
                        chat_data[user].append({
                            'timestamp': timestamp,
                            'message': self.clean_message(message),
                            'line_number': row_number
                        })

                except Exception as e:
                    print(f"Warning: Error processing line {line_number}: {e}")
        
        self.chat_data = dict(chat_data)
        self.users = set(self.chat_data.keys()) - {'SYSTEM'}
        self.messages = [msg for user_msgs in self.chat_data.values() for msg in user_msgs]
        self.daily_active_users = self.calculate_daily_active_users()
        self.extract_interests_and_events()

        print(f"Parsed {len(self.messages)} messages")
        print(f"Found {len(self.users)} users")
        print(f"Found {len(self.events)} events")

    def clean_message(self, message):
        # Implement any necessary message cleaning logic here
        return message.strip()

    def calculate_daily_active_users(self):
        daily_active = defaultdict(set)
        for user, messages in self.chat_data.items():
            for msg in messages:
                date = msg['timestamp'].date()
                daily_active[date].add(user)
        return daily_active

    def extract_interests_and_events(self):
        self.interests = defaultdict(set)
        self.events = []
        for user, messages in self.chat_data.items():
            for msg in messages:
                # Extract interests using NLP
                doc = nlp(msg['message'])
                interests = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT', 'NOUN', 'PROPN', 'GPE', 'LOC', 'MISC']]
                self.interests[user].update(interests)

                # Detect events
                if 'event' in msg['message'].lower() or 'rsvp' in msg['message'].lower():
                    self.events.append((msg['timestamp'].strftime('%m/%d/%y'), 
                                        msg['timestamp'].strftime('%I:%M:%S %p'), 
                                        user, 
                                        msg['message']))

    def export_to_csv(self):
        # Export daily active users
        with open('daily_active_users.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', 'Active Users'])
            for date, users in self.daily_active_users.items():
                writer.writerow([date, len(users)])

        # Export total users and groups
        with open('total_users_groups.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Total Users', 'Total Groups'])
            writer.writerow([len(self.users), len(self.groups)])

        # Export user interests
        with open('user_interests.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['User', 'Interests', 'Total Interests'])
            for user, interests in self.interests.items():
                writer.writerow([user, ', '.join(interests), len(interests)])

        # Export events
        with open('events.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', 'Time', 'User', 'Event Description'])
            writer.writerows(self.events)

        print("CSV files exported successfully")

    def analyze_event_attendance(self):
        # This is a placeholder for event attendance analysis
        # In a real implementation, you'd need to track RSVPs and actual attendance
        pass

    def analyze_event_timing(self):
        event_days = defaultdict(int)
        event_times = defaultdict(int)

        for date, time, _, _ in self.events:
            day = datetime.strptime(date, '%m/%d/%y').strftime('%A')
            hour = datetime.strptime(time, '%I:%M:%S %p').hour
            event_days[day] += 1
            event_times[hour] += 1

        with open('event_timing_analysis.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Day', 'Count'])
            writer.writerows(event_days.items())
            writer.writerow([])
            writer.writerow(['Hour', 'Count'])
            writer.writerows(event_times.items())

    def analyze_user_similarity(self):
        if not self.interests:
            print("No user interests found. Skipping similarity analysis.")
            return

        user_embeddings = {}

        for user, interests in self.interests.items():
            # Combine all interests into a single text
            text = ' '.join(interests)
            
            # Tokenize and get BERT embeddings
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Use the mean of the last hidden state as the embedding
            embedding = outputs.last_hidden_state.mean(dim=1)
            user_embeddings[user] = embedding

        # Calculate cosine similarity between users
        similarity_matrix = {}
        for user1 in user_embeddings:
            similarity_matrix[user1] = {}
            for user2 in user_embeddings:
                if user1 != user2:
                    similarity = torch.cosine_similarity(user_embeddings[user1], user_embeddings[user2])
                    similarity_matrix[user1][user2] = similarity.item()

        # Export similarity matrix to CSV
        with open('user_similarity.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            users = list(user_embeddings.keys())
            writer.writerow(['User'] + users)
            for user1 in users:
                row = [user1]
                for user2 in users:
                    if user1 == user2:
                        row.append(1.0)  # Self-similarity is always 1
                    else:
                        row.append(similarity_matrix[user1][user2])
                writer.writerow(row)

        print("User similarity analysis complete")

# Usage
analyzer = ChatAnalyzer()
analyzer.preprocess_chat_to_csv('./chats/_chat2.txt', './chats/formatted_chat.csv')
analyzer.parse_chat_file('./chats/formatted_chat.csv')
analyzer.export_to_csv()
analyzer.analyze_event_timing()
analyzer.analyze_user_similarity()