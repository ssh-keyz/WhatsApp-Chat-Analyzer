import pandas as pd
import random
import requests
import json
import csv
from datetime import datetime
import re
import time
import os

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

def get_user_messages(user, max_messages=1000):
    user_messages = df[df['user'] == user]['message'].dropna().tolist()
    cleaned_messages = [clean_message(msg) for msg in user_messages if clean_message(msg)]
    return cleaned_messages[:max_messages]  # Limit the number of messages

def create_tagging_prompt(messages):
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "category": {"type": "string", "enum": ["Topic", "Event", "Generic", "Question", "Personal", "Announcement", "Too Many Tokens", "Other"]}
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
Categorize each of the following messages as either "Topic" (related to a specific non-person-subject, location, hobby, activity, or interest. A message like 'thinking of having a chill board game spooky movie night at [users] place', this is a topic), "Generic" (general conversation), "Event" (mentions that they are on the way to an event, plan to go, etc. if an event is interest based, this would be the Topic category), "Personal" (about the user's life unless they mention a topic, location, or interest), "Announcement" (sharing news or updates, a person was added, a person joined from the community, a person left the community, a person was removed, a person left), "Too Many Tokens" (if the message is too long to process), or "Other" (if it doesn't fit the previous categories). Return the categorizations in JSON format according to the provided schema.

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
        "temperature": 0.3,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        json_response = result['choices'][0]['message']['content']
        print(json_response)
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

def chunk_messages(messages, chunk_size=7500):
    """Split messages into chunks of specified size."""
    return [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]

def process_user(user):
    start_time = time.time()
    api_calls = 0
    
    print(f"Processing user: {user}")
    messages = get_user_messages(user)
    
    if not messages:
        print(f"No valid messages found for user: {user}")
        return None
    
    print(messages)
    print(f"Processing {len(messages)} messages for user: {user}")
    
    # Check if tagged messages file exists
    tagged_messages_file = f'tagged_messages_{user}.json'
    if os.path.exists(tagged_messages_file):
        print(f"Loading tagged messages from file for user: {user}")
        with open(tagged_messages_file, 'r') as f:
            tagged_messages = json.load(f)
    else:
        # Process messages based on count
        if len(messages) <= 100:
            # Process all messages in one go
            tagging_prompt = create_tagging_prompt(messages)
            tagged_messages = get_response(tagging_prompt, max_tokens=40000)
            api_calls += 1
        else:
            # Use chunking for users with more messages
            tagged_messages = []
            for chunk in chunk_messages(messages):
                chunk_start = time.time()
                tagging_prompt = create_tagging_prompt(chunk)
                chunk_tags = get_response(tagging_prompt, max_tokens=40000)
                api_calls += 1
                chunk_time = time.time() - chunk_start
                print(f"Chunk processing time: {chunk_time:.2f} seconds")
                if chunk_tags:
                    tagged_messages.extend(chunk_tags)
                else:
                    print(f"Failed to tag a chunk of messages for user: {user}")
        
        # Save tagged messages to file
        with open(tagged_messages_file, 'w') as f:
            json.dump(tagged_messages, f)
    
    if not tagged_messages:
        print(f"Failed to tag any messages for user: {user}")
        return None
    
    # Process tagged messages
    message_counts = {'Topic': 0, 'Generic': 0, 'Question': 0, 'Personal': 0, 'Announcement': 0, 'Other': 0}
    topic_messages = []
    
    for msg in tagged_messages:
        message_counts[msg['category']] += 1
        if msg['category'] == 'Topic':
            topic_messages.append(msg['message'])
    
    if not topic_messages:
        total_time = time.time() - start_time
        print(f"No topic messages found for user {user}. Total processing time: {total_time:.2f} seconds")
        print(f"Total API calls for user {user}: {api_calls}")
        return {
            'user': user,
            'message_counts': message_counts,
            'interests_data': None,
            'processing_time': total_time,
            'api_calls': api_calls
        }
    
    # Generate interests based on topic messages
    interests_start = time.time()
    interests_prompt = create_interests_prompt(topic_messages[:100])  # Limit to 100 topic messages
    interests_data = get_response(interests_prompt, max_tokens=80000)
    api_calls += 1
    interests_time = time.time() - interests_start
    print(f"Interests generation time: {interests_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"Total processing time for user {user}: {total_time:.2f} seconds")
    print(f"Total API calls for user {user}: {api_calls}")
    
    return {
        'user': user,
        'message_counts': message_counts,
        'interests_data': interests_data,
        'processing_time': total_time,
        'api_calls': api_calls
    }

def process_all_users(all_users):
    fieldnames = ['User', 'Topic_Count', 'Generic_Count', 'Question_Count', 'Personal_Count', 'Announcement_Count', 'Other_Count', 
                  'Interest1', 'Interest2', 'Interest3', 'Interest4', 'Interest5', 'Suggested_Group', 'Processing_Time', 'API_Calls']
    
    # Write the header once
    with open('user_data_summary.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    total_processing_time = 0
    total_api_calls = 0
    users_with_interests = 0
    
    for user in all_users:
        result = process_user(user)
        if result:
            interests = result['interests_data'] or {}
            
            # Open the file in append mode, write the row, and close it
            with open('user_data_summary.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'User': user,
                    'Topic_Count': result['message_counts']['Topic'],
                    'Generic_Count': result['message_counts']['Generic'],
                    'Question_Count': result['message_counts']['Question'],
                    'Personal_Count': result['message_counts']['Personal'],
                    'Announcement_Count': result['message_counts']['Announcement'],
                    'Other_Count': result['message_counts']['Other'],
                    'Interest1': interests.get('interest1', ''),
                    'Interest2': interests.get('interest2', ''),
                    'Interest3': interests.get('interest3', ''),
                    'Interest4': interests.get('interest4', ''),
                    'Interest5': interests.get('interest5', ''),
                    'Suggested_Group': interests.get('suggested_group', ''),
                    'Processing_Time': result['processing_time'],
                    'API_Calls': result['api_calls']
                })
            
            total_processing_time += result['processing_time']
            total_api_calls += result['api_calls']
            if result['interests_data']:
                users_with_interests += 1
        
        print(f"Processed user: {user}")
    
    print(f"Total processing time for all users: {total_processing_time:.2f} seconds")
    print(f"Total API calls for all users: {total_api_calls}")
    print(f"Users with generated interests: {users_with_interests}")

# Main execution
all_users = df['user'].unique()
process_all_users(all_users)