import sqlite3
import os

# Database folder
db_folder = "data_pipeline/data"

# Create folder if it does not exist
os.makedirs(db_folder, exist_ok=True)

# Database file path
db_file = os.path.join(db_folder, "zepto_books.db")

# Connect to SQLite database
connection = sqlite3.connect(db_file)

# Create cursor
cursor = connection.cursor()

# Enable foreign key support
cursor.execute("PRAGMA foreign_keys = ON")

# Check foreign key support
foreign_keys = cursor.execute("PRAGMA foreign_keys").fetchone()[0]

print("==============================")
print("SQLITE DATABASE CREATED")
print("==============================")
print()
print("Database file:")
print(db_file)
print()
print("SQLite connection: SUCCESS")
print("Foreign key support:", foreign_keys)
print()

# Close connection
connection.close()

print("Database connection closed")
print("==============================")