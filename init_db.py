import sqlite3

conn = sqlite3.connect("query_logs.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_id TEXT,
    prompt TEXT,
    response TEXT,
    source TEXT
)
""")

conn.commit()
conn.close()
print("✅ query_logs.db initialized.")
