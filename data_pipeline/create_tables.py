import sqlite3
import os

# Database path
db_file = "data_pipeline/data/zepto_books.db"

# Check database exists
if not os.path.exists(db_file):
    print("ERROR: Database file not found.")
    print("Please run create_database.py first.")
    exit()

# Connect to database
connection = sqlite3.connect(db_file)

# Enable foreign keys
connection.execute("PRAGMA foreign_keys = ON")

cursor = connection.cursor()

# ------------------------------
# 1. Create categories table
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
)
""")

# ------------------------------
# 2. Create books table
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")

# Save changes
connection.commit()

# ------------------------------
# Verify tables
# ------------------------------
cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type = 'table'
ORDER BY name
""")

tables = cursor.fetchall()

print("==============================")
print("NORMALIZED TABLES CREATED")
print("==============================")
print()

print("Tables in database:")

for table in tables:
    print("-", table[0])

print()

# Check categories structure
print("categories table:")
cursor.execute("PRAGMA table_info(categories)")
for column in cursor.fetchall():
    print(column)

print()

# Check books structure
print("books table:")
cursor.execute("PRAGMA table_info(books)")
for column in cursor.fetchall():
    print(column)

print()

# Check foreign keys
print("Foreign keys in books table:")
cursor.execute("PRAGMA foreign_key_list(books)")
for foreign_key in cursor.fetchall():
    print(foreign_key)

print()

# Close database
connection.close()

print("==============================")
print("TABLE CREATION COMPLETE")
print("==============================")