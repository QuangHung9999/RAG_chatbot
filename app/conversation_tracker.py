import sqlite3
import os
import time
import json
from datetime import datetime
import re
from app.config import CHAT_HISTORY_DB_PATH

# Added token counting utilities
def estimate_tokens(text):
    """
    Estimate the number of tokens in a text string.
    This is a simple approximation - for production use, consider using a proper tokenizer.
    """
    # Rough approximation: ~4 chars per token for English text
    if not text:
        return 0
    return len(text) // 4

def count_tokens_in_messages(messages):
    """Count tokens in a list of messages"""
    total = 0
    for msg in messages:
        if isinstance(msg, dict) and 'content' in msg:
            total += estimate_tokens(msg['content'])
        elif isinstance(msg, str):
            total += estimate_tokens(msg)
    return total

class ConversationTracker:
    """
    Tracks and stores conversation metrics including token usage, latency,
    and other relevant statistics.
    """
    
    def __init__(self, db_path=CHAT_HISTORY_DB_PATH):
        """Initialize the tracker with path to SQLite database"""
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Ensure the database and required tables exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                total_messages INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_latency_ms INTEGER DEFAULT 0,
                metadata TEXT
            )
            ''')
            
            # Create messages table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                message_type TEXT NOT NULL, 
                timestamp TIMESTAMP NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                embedding_time_ms INTEGER DEFAULT 0,
                retrieval_time_ms INTEGER DEFAULT 0,
                generation_time_ms INTEGER DEFAULT 0,
                documents_retrieved INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
            ''')
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def start_conversation(self, user_id):
        """Start tracking a new conversation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_time = datetime.now()
            cursor.execute(
                "INSERT INTO conversations (user_id, start_time) VALUES (?, ?)",
                (user_id, start_time)
            )
            conn.commit()
            conversation_id = cursor.lastrowid
            
            return conversation_id
        except sqlite3.Error as e:
            print(f"Error starting conversation: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def end_conversation(self, conversation_id):
        """Mark a conversation as ended"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            end_time = datetime.now()
            cursor.execute(
                "UPDATE conversations SET end_time = ? WHERE id = ?",
                (end_time, conversation_id)
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error ending conversation: {e}")
        finally:
            if conn:
                conn.close()
    
    def add_message(self, conversation_id, message_type, content, metrics=None):
        """
        Add a message to the conversation history with metrics
        
        Args:
            conversation_id: ID of the conversation
            message_type: 'user' or 'assistant'
            content: The message content
            metrics: Dictionary containing metrics like tokens, latency, etc.
        """
        if metrics is None:
            metrics = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract metrics with defaults
            tokens = metrics.get('tokens', 0)
            # If tokens not provided, estimate them
            if tokens == 0 and content:
                tokens = estimate_tokens(content)
                
            latency_ms = metrics.get('latency_ms', 0)
            embedding_time_ms = metrics.get('embedding_time_ms', 0)
            retrieval_time_ms = metrics.get('retrieval_time_ms', 0)
            generation_time_ms = metrics.get('generation_time_ms', 0)
            documents_retrieved = metrics.get('documents_retrieved', 0)
            
            # Add message
            timestamp = datetime.now()
            cursor.execute(
                """
                INSERT INTO messages 
                (conversation_id, message_type, timestamp, content, 
                tokens, latency_ms, embedding_time_ms, retrieval_time_ms, 
                generation_time_ms, documents_retrieved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, message_type, timestamp, content, 
                tokens, latency_ms, embedding_time_ms, retrieval_time_ms, 
                generation_time_ms, documents_retrieved)
            )
            
            # Update conversation stats
            cursor.execute(
                """
                UPDATE conversations SET 
                total_messages = total_messages + 1,
                total_tokens = total_tokens + ?,
                total_latency_ms = total_latency_ms + ?
                WHERE id = ?
                """,
                (tokens, latency_ms, conversation_id)
            )
            
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error adding message: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def update_conversation_metadata(self, conversation_id, metadata):
        """Update metadata for a conversation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert dict to JSON string
            metadata_json = json.dumps(metadata)
            
            cursor.execute(
                "UPDATE conversations SET metadata = ? WHERE id = ?",
                (metadata_json, conversation_id)
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error updating conversation metadata: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_conversation_stats(self, conversation_id=None, user_id=None, start_date=None, end_date=None):
        """Get statistics for conversations with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            query = "SELECT * FROM conversations WHERE 1=1"
            params = []
            
            if conversation_id:
                query += " AND id = ?"
                params.append(conversation_id)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if start_date:
                query += " AND start_time >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND start_time <= ?"
                params.append(end_date)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting conversation stats: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_conversation_details(self, conversation_id):
        """Get detailed information about a specific conversation, including all messages"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get conversation data
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            conversation = dict(cursor.fetchone() or {})
            
            if not conversation:
                return None
            
            # Get messages
            cursor.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp", 
                          (conversation_id,))
            messages = [dict(row) for row in cursor.fetchall()]
            
            # Add messages to conversation data
            conversation['messages'] = messages
            
            return conversation
        except sqlite3.Error as e:
            print(f"Error getting conversation details: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_aggregated_stats(self, user_id=None, start_date=None, end_date=None):
        """Get aggregated statistics across conversations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT 
                COUNT(id) as conversation_count,
                SUM(total_messages) as total_messages,
                SUM(total_tokens) as total_tokens,
                AVG(total_tokens) as avg_tokens_per_conversation,
                SUM(total_latency_ms) as total_latency_ms,
                AVG(total_latency_ms) as avg_latency_ms_per_conversation
            FROM conversations
            WHERE 1=1
            """
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if start_date:
                query += " AND start_time >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND start_time <= ?"
                params.append(end_date)
            
            cursor.execute(query, params)
            return dict(zip([column[0] for column in cursor.description], cursor.fetchone()))
        except sqlite3.Error as e:
            print(f"Error getting aggregated stats: {e}")
            return {}
        finally:
            if conn:
                conn.close()


# Timer utility class for measuring execution times
class Timer:
    """Utility class to measure execution time of operations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer"""
        self.end_time = time.time()
        return self
    
    def elapsed_ms(self):
        """Get elapsed time in milliseconds"""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time is not None else time.time()
        return int((end - self.start_time) * 1000) 