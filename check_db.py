import sqlite3
import os

# Connect to the chat history database
db_path = 'app/chat_history.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check for existing tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(f"Tables in {db_path}:")
for table in tables:
    print(f"- {table[0]}")
    # Get schema for each table
    cursor.execute(f"PRAGMA table_info({table[0]})")
    columns = cursor.fetchall()
    print("  Columns:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

conn.close() 