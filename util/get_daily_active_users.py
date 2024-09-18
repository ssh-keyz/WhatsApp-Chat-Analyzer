import csv
from collections import defaultdict
from datetime import datetime

def count_daily_active_users(file_path, start_date, end_date):
    daily_users = defaultdict(set)
    
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date_str = row['date_time'].split()[0]
            date = datetime.strptime(date_str, '%m/%d/%y').date()
            if start_date <= date <= end_date:
                daily_users[date].add(row['user'])
    
    return {date: len(users) for date, users in daily_users.items()}

# Define the date range
start_date = datetime(2024, 9, 8).date()
end_date = datetime(2024, 9, 16).date()

# Count daily active users
daily_active_users = count_daily_active_users('../chats/formatted_chat.csv', start_date, end_date)

# Generate CSV output
csv_output = "date,active_users\n"
for date in sorted(daily_active_users.keys()):
    csv_output += f"{date.strftime('%Y-%m-%d')},{daily_active_users[date]}\n"

print(csv_output)