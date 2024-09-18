# import torch
# from transformers import AutoTokenizer, LlamaForCausalLM
# import pandas as pd
# import random
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# import torch
# torch.cuda.is_available = lambda : False

# # Load the model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-3-Llama-3.1-8B', trust_remote_code=True)
# model = LlamaForCausalLM.from_pretrained(
#     "NousResearch/Hermes-3-Llama-3.1-8B",
#     torch_dtype=torch.float32,
#     device_map="cpu",
#     load_in_8bit=False,
#     load_in_4bit=False,
#     use_flash_attention_2=False
# )

# # Read the CSV file
# df = pd.read_csv('formatted_chat.csv')

# def get_user_messages(user, n=10):
#     user_messages = df[df['user'] == user]['message'].tolist()
#     return random.sample(user_messages, min(n, len(user_messages)))

# def create_prompt(messages):
#     prompt = """<|im_start|>system
# You are a sentient, superintelligent artificial general intelligence, here to analyze user interests based on their messages.
# <|im_end|>
# <|im_start|>user
# Based on the following messages from a user, identify and list their top 5 interests or hobbies. Here are the messages:

# """
#     for msg in messages:
#         prompt += f"- {msg}\n"
#     prompt += "\nWhat are this user's top 5 interests or hobbies based on these messages?<|im_end|>\n<|im_start|>assistant"
#     return prompt

# # def get_interests(user):
# #     messages = get_user_messages(user)
# #     prompt = create_prompt(messages)
    
# #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
# #     generated_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.7, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
# #     response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
    
# #     return response
# def get_interests(user):
#     messages = get_user_messages(user)
#     prompt = create_prompt(messages)
    
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     generated_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.7, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
    
#     return response
# # Example usage
# user = "~ MJ Morrison"  # Replace with an actual user from your CSV
# interests = get_interests(user)
# print(f"Interests for {user}:")
# # Write interests to CSV
# import csv

# with open('interests.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['User', 'Interests'])
#     writer.writerow([user, interests])

# print(f"Interests for {user} have been written to interests.csv")


# import pandas as pd
# import random
# import requests
# import json
# import csv

# # Read the CSV file
# df = pd.read_csv('formatted_chat.csv')

# API_ENDPOINT = "http://0.0.0.0:8080/v1/chat/completions"

# def get_user_messages(user, n=10):
#     user_messages = df[df['user'] == user]['message'].tolist()
#     return random.sample(user_messages, min(n, len(user_messages)))

# def create_prompt(messages):
#     system_message = "You are an AI assistant tasked with analyzing user interests based on their messages. Provide a concise, specific list of interests."
#     user_message = "Based on the following messages, identify the user's top 5 specific interests or hobbies. List them in order of apparent importance, separated by commas:\n\n"
#     for msg in messages:
#         user_message += f"- {msg}\n"
#     user_message += "\nUser's top 5 specific interests or hobbies:"
    
#     return [
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": user_message}
#     ]

# def get_interests(user):
#     messages = get_user_messages(user)
#     prompt = create_prompt(messages)
    
#     payload = {
#         "model": "Hermes-3-Llama-3.1-8B.Q8_0.gguf",
#         "messages": prompt,
#         "max_tokens": 100,
#         "temperature": 0.3
#     }
    
#     headers = {
#         "Content-Type": "application/json"
#     }
    
#     try:
#         response = requests.post(API_ENDPOINT, json=payload, headers=headers)
#         response.raise_for_status()
#         result = response.json()
#         interests = result['choices'][0]['message']['content']
#         return interests.strip()
#     except requests.exceptions.RequestException as e:
#         print(f"Error for user {user}: {str(e)}")
#         if response.text:
#             print(f"Response content: {response.text}")
#         return f"Error: {str(e)}"

# # Process all users
# all_users = df['user'].unique()
# user_interests = {}

# for user in all_users:
#     interests = get_interests(user)
#     user_interests[user] = interests
#     print(f"Processed interests for {user}: {interests}")

# # Write interests to CSV
# with open('all_user_interests.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['User', 'Interests'])
#     for user, interests in user_interests.items():
#         writer.writerow([user, interests])

# print("All user interests have been written to all_user_interests.csv")

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
df = pd.read_csv('formatted_chat.csv', 
                 parse_dates=['date_time'], 
                 date_parser=custom_date_parser,
                 encoding='utf-8',
                 quotechar='"',
                 escapechar='\\',
                 on_bad_lines='skip')

# Read the CSV file with custom date parsing
df = pd.read_csv('formatted_chat.csv', 
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

def get_user_messages(user, n=10):
    user_messages = df[df['user'] == user]['message'].dropna().tolist()
    cleaned_messages = [clean_message(msg) for msg in user_messages if clean_message(msg)]
    return random.sample(cleaned_messages, min(n, len(cleaned_messages)))

def create_prompt(user, messages):
    schema = {
        "type": "object",
        "properties": {
            "username": {"type": "string"},
            "interest1": {"type": "string"},
            "interest2": {"type": "string"},
            "interest3": {"type": "string"},
            "suggested_possible_interest_or_activity_group": {"type": "string"},
            "explanation": {"type": "string"}
        },
        "required": ["username", "interest1", "interest2", "interest3", "suggested_possible_interest_or_activity_group", "explanation"]
    }
    
    system_message = f"""<|im_start|>system
You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:
<schema>
{json.dumps(schema, indent=2)}
</schema><|im_end|>"""

    user_message = f"""<|im_start|>user
Based on the following messages from a user, identify their top 3 interests and suggest a possible interest or activity group they might enjoy. Provide a brief explanation for your suggestion. Return the information in JSON format according to the provided schema. Ignore any @mentions or attachment notifications in the messages.

Username: {user}

User's messages:
{chr(10).join(f"- {msg}" for msg in messages)}

Remember to strictly follow the JSON schema provided in the system message.<|im_end|>
<|im_start|>assistant"""
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def get_interests(user):
    messages = get_user_messages(user)
    prompt = create_prompt(user, messages)
    
    payload = {
        "model": "Hermes-3-Llama-3.1-8B.Q8_0.gguf",
        "messages": prompt,
        "max_tokens": 300,
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
            start = json_response.index('{')
            end = json_response.rindex('}') + 1
            valid_json = json_response[start:end]
            return json.loads(valid_json)
        except (ValueError, json.JSONDecodeError):
            print(f"Invalid JSON for user {user}. Raw response: {json_response}")
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"Request error for user {user}: {str(e)}")
        if response.text:
            print(f"Response content: {response.text}")
        return None

# Process all users
all_users = df['user'].unique()
user_data = {}

for user in all_users:
    user_df = df[df['user'] == user]
    
    # Find start date
    start_messages = user_df[user_df['message'].str.contains('was added|joined from the community', case=False, na=False)]
    start_date = start_messages['date_time'].min() if not start_messages.empty else None
    
    # Find end date
    end_messages = user_df[user_df['message'].str.contains('left', case=False, na=False)]
    # end_date = end_messages['date_time'].max() if not end_messages.empty else None
    removed_messages = df[df['message'].str.contains(f'removed {user}', case=False, na=False)]
    
    if not end_messages.empty:
        end_date = end_messages['date_time'].max()
    elif not removed_messages.empty:
        end_date = removed_messages['date_time'].min()
    else:
        end_date = None
    
    # Count engagement (including photo/video attachments)
    engagement_count = len(user_df) + sum(user_df['message'].str.contains('<attached:', case=False, na=False))
    
    # Get interests
    interests_data = get_interests(user)
    
    if interests_data:
        user_data[user] = {
            'start_date': start_date,
            'end_date': end_date,
            'engagement_count': engagement_count,
            'interests_data': interests_data
        }
        print(f"Processed data for {user}: Start: {start_date}, End: {end_date}, Engagement: {engagement_count}, Interests: {interests_data}")

# Write data to CSV
with open('../chats/user_data_with_interests.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['User', 'Start Date', 'End Date', 'Engagement Count', 'Interest 1', 'Interest 2', 'Interest 3', 'Suggested Group', 'Explanation'])
    for user, data in user_data.items():
        interests = data['interests_data']
        writer.writerow([
            user,
            data['start_date'],
            data['end_date'],
            data['engagement_count'],
            interests.get('interest1', ''),
            interests.get('interest2', ''),
            interests.get('interest3', ''),
            interests.get('suggested_possible_interest_or_activity_group', ''),
            interests.get('explanation', '')
        ])

print("All user data has been written to user_data_with_interests.csv")

# Calculate retention metrics
total_users = len(user_data)
active_users = sum(1 for data in user_data.values() if data['end_date'] is None)
retention_rate = (active_users / total_users) * 100 if total_users > 0 else 0

print(f"Total Users: {total_users}")
print(f"Active Users: {active_users}")
print(f"Retention Rate: {retention_rate:.2f}%")