import pandas as pd
import random
import requests
import json
import csv
from datetime import datetime
import re

# Custom date parser function
def custom_date_parser(date_string):
    try:
        return pd.to_datetime(date_string, format='%m/%d/%y %I:%M:%S%p')
    except ValueError:
        # If the above fails, try without seconds
        try:
            return pd.to_datetime(date_string, format='%m/%d/%y %I:%M%p')
        except ValueError:
            # If both fail, return NaT (Not a Time)
            print(f"Unable to parse date: {date_string}")
            return pd.NaT

# Update the read_csv function to use the new parser
df = pd.read_csv('../chats/formatted_chat.csv', 
                 parse_dates=['date_time'], 
                 date_parser=custom_date_parser,
                 encoding='utf-8',
                 quotechar='"',
                 escapechar='\\',
                 on_bad_lines='skip')

# Read the CSV file with custom date parsing
df = pd.read_csv('../chats/formatted_chat.csv', 
                 parse_dates=['date_time'], 
                 date_parser=custom_date_parser,
                 encoding='utf-8',
                 quotechar='"',  # Specify the quote character
                 escapechar='\\',  # Specify the escape character
                 on_bad_lines='skip')  # Skip bad lines

API_ENDPOINT = "http://0.0.0.0:8080/v1/chat/completions"

def clean_message(message):
    # Remove @mentions
    cleaned = re.sub(r'@\d+', '', message)
    # Remove attachment notifications
    cleaned = re.sub(r'<attached: .*?>', '', cleaned)
    return cleaned.strip()

def get_user_messages(user):
    user_messages = df[df['user'] == user]['message'].dropna().tolist()
    return [clean_message(msg) for msg in user_messages if clean_message(msg)]

def create_tagging_prompt(messages):
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "category": {"type": "string", "enum": ["Topic", "Generic", "Question", "Personal", "Announcement", "Other"]}
            },
            "required": ["message", "category"]
        }
    }
    
    system_message = f"""<|im_start|>system
You are a helpful assistant that categorizes messages. Here's the json schema you must adhere to:
<schema>
{json.dumps(schema, indent=2)}
</schema><|im_end|>"""

    user_message = f"""<|im_start|>user
Categorize each of the following messages as either "Topic" (related to a specific subject or interest), "Generic" (general conversation), "Question" (asking for information), "Personal" (about the user's life), "Announcement" (sharing news or updates), or "Other" (if it doesn't fit the previous categories). Return the categorizations in JSON format according to the provided schema.

Messages:
{chr(10).join(f"- {msg}" for msg in messages)}

Remember to strictly follow the JSON schema provided in the system message.<|im_end|>
<|im_start|>assistant"""
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def create_interests_prompt(topic_messages):
    schema = {
        "type": "object",
        "properties": {
            "interest1": {"type": "string"},
            "interest2": {"type": "string"},
            "interest3": {"type": "string"},
            "interest4": {"type": "string"},
            "interest5": {"type": "string"},
            "suggested_group": {"type": "string"},
            "explanation": {"type": "string"}
        },
        "required": ["interest1", "interest2", "interest3", "interest4", "interest5", "suggested_group", "explanation"]
    }
    
    system_message = f"""<|im_start|>system
You are a helpful assistant that analyzes user interests. Here's the json schema you must adhere to:
<schema>
{json.dumps(schema, indent=2)}
</schema><|im_end|>"""

    user_message = f"""<|im_start|>user
Based on the following topic-related messages from a user, identify their top 5 interests and suggest a possible interest or activity group they might enjoy. Provide a brief explanation for your suggestion. Return the information in JSON format according to the provided schema.

Topic Messages:
{chr(10).join(f"- {msg}" for msg in topic_messages)}

Remember to strictly follow the JSON schema provided in the system message.<|im_end|>
<|im_start|>assistant"""
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def get_response(prompt, max_tokens):
    payload = {
        "model": "Hermes-3-Llama-3.1-8B.Q8_0.gguf",
        "messages": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        json_response = result['choices'][0]['message']['content']
        
        # Try to find valid JSON within the response
        try:
            start = json_response.index('[')
            end = json_response.rindex(']') + 1
            valid_json = json_response[start:end]
            return json.loads(valid_json)
        except ValueError:
            try:
                start = json_response.index('{')
                end = json_response.rindex('}') + 1
                valid_json = json_response[start:end]
                return json.loads(valid_json)
            except (ValueError, json.JSONDecodeError):
                print(f"Invalid JSON. Raw response: {json_response}")
                return None
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        if response.text:
            print(f"Response content: {response.text}")
        return None

def process_user(user):
    messages = get_user_messages(user)
    
    # Tag messages
    tagging_prompt = create_tagging_prompt(messages)
    tagged_messages = get_response(tagging_prompt, max_tokens=2000)
    
    if not tagged_messages:
        return None
    
    # Filter topic messages
    topic_messages = [msg['message'] for msg in tagged_messages if msg['category'] == 'Topic']
    
    if not topic_messages:
        return {
            'user': user,
            'tagged_messages': tagged_messages,
            'interests_data': None
        }
    
    # Generate interests based on topic messages
    interests_prompt = create_interests_prompt(topic_messages)
    interests_data = get_response(interests_prompt, max_tokens=1000)
    
    return {
        'user': user,
        'tagged_messages': tagged_messages,
        'interests_data': interests_data
    }

# Process all users
all_users = df['user'].unique()
user_data = {}

for user in all_users:
    print(f"Processing user: {user}")
    result = process_user(user)
    if result:
        user_data[user] = result
        print(f"Completed processing for user: {user}")
    else:
        print(f"Failed to process user: {user}")

# Write data to CSV
with open('../chats/user_data_with_tags_and_interests.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['User', 'Message', 'Category', 'Interest 1', 'Interest 2', 'Interest 3', 'Interest 4', 'Interest 5', 'Suggested Group', 'Explanation'])
    for user, data in user_data.items():
        interests = data['interests_data'] or {}
        for message in data['tagged_messages']:
            writer.writerow([
                user,
                message['message'],
                message['category'],
                interests.get('interest1', ''),
                interests.get('interest2', ''),
                interests.get('interest3', ''),
                interests.get('interest4', ''),
                interests.get('interest5', ''),
                interests.get('suggested_group', ''),
                interests.get('explanation', '')
            ])

print("All user data has been written to user_data_with_tags_and_interests.csv")

# Calculate statistics
total_users = len(user_data)
users_with_interests = sum(1 for data in user_data.values() if data['interests_data'])
percentage_with_interests = (users_with_interests / total_users) * 100 if total_users > 0 else 0

print(f"Total Users Processed: {total_users}")
print(f"Users with Interests: {users_with_interests}")
print(f"Percentage of Users with Interests: {percentage_with_interests:.2f}%")