import sqlite3
import pandas as pd
import os

# ------------------------------
# File paths
# ------------------------------
csv_file = "data_pipeline/data/cleaned_books.csv"
db_file = "data_pipeline/data/zepto_books.db"
output_file = "data_pipeline/data/sql_results.txt"

# ------------------------------
# Check files
# ------------------------------
if not os.path.exists(csv_file):
    print("ERROR: cleaned_books.csv not found.")
    exit()

if not os.path.exists(db_file):
    print("ERROR: zepto_books.db not found.")
    print("Please complete Step 17 first.")
    exit()

# ------------------------------
# Load cleaned CSV
# ------------------------------
df = pd.read_csv(csv_file)

print("==============================")
print("CLEANED DATA LOADED")
print("==============================")
print("Rows:", len(df))
print()

# ------------------------------
# Connect to SQLite
# ------------------------------
connection = sqlite3.connect(db_file)

# Enable foreign keys
connection.execute("PRAGMA foreign_keys = ON")

cursor = connection.cursor()

# ------------------------------
# Clear old data
# ------------------------------
cursor.execute("DELETE FROM books")
cursor.execute("DELETE FROM categories")

# Reset AUTOINCREMENT counters
cursor.execute(
    "DELETE FROM sqlite_sequence WHERE name IN ('books', 'categories')"
)

# ------------------------------
# Insert categories
# ------------------------------
categories = sorted(df["category"].dropna().unique())

for category in categories:
    cursor.execute(
        """
        INSERT INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )

# ------------------------------
# Create category mapping
# ------------------------------
cursor.execute("""
SELECT category_id, category_name
FROM categories
""")

category_rows = cursor.fetchall()

category_map = {
    category_name: category_id
    for category_id, category_name in category_rows
}

# ------------------------------
# Insert books
# ------------------------------
for _, row in df.iterrows():

    category_id = category_map.get(row["category"])

    cursor.execute(
        """
        INSERT INTO books
        (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            bool(row["in_stock"]),
            category_id
        )
    )

# Save inserted data
connection.commit()

print("==============================")
print("DATA INSERTION COMPLETE")
print("==============================")
print("Books inserted:", len(df))
print("Categories inserted:", len(categories))
print()

# ==========================================================
# SQL QUERIES
# ==========================================================

queries = {

    # 1. SELECT + WHERE
    "QUERY 1 - SELECT WHERE": """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
""",

    # 2. ORDER BY + LIMIT
    "QUERY 2 - ORDER BY LIMIT": """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10
""",

    # 3. DISTINCT
    "QUERY 3 - DISTINCT": """
SELECT DISTINCT rating
FROM books
ORDER BY rating
""",

    # 4. BETWEEN
    "QUERY 4 - BETWEEN": """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp
""",

    # 5. IN
    "QUERY 5 - IN": """
SELECT title, rating
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC
""",

    # 6. JOIN
    "QUERY 6 - JOIN": """
SELECT
    books.title,
    books.price_gbp,
    books.price_inr,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
ORDER BY books.price_gbp DESC
LIMIT 10
"""
}

# ------------------------------
# Run queries
# ------------------------------
all_results = []

for query_name, query in queries.items():

    print()
    print("=" * 60)
    print(query_name)
    print("=" * 60)

    print("SQL:")
    print(query.strip())
    print()

    result = pd.read_sql_query(query, connection)

    print("OUTPUT:")
    print(result.to_string(index=False))

    all_results.append("=" * 60)
    all_results.append(query_name)
    all_results.append("=" * 60)
    all_results.append("SQL:")
    all_results.append(query.strip())
    all_results.append("")
    all_results.append("OUTPUT:")
    all_results.append(result.to_string(index=False))
    all_results.append("")

# ------------------------------
# Save query results
# ------------------------------
with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n".join(all_results))

# ------------------------------
# Final database counts
# ------------------------------
book_count = cursor.execute(
    "SELECT COUNT(*) FROM books"
).fetchone()[0]

category_count = cursor.execute(
    "SELECT COUNT(*) FROM categories"
).fetchone()[0]

print()
print("==============================")
print("DATABASE VERIFICATION")
print("==============================")
print("Books in database:", book_count)
print("Categories in database:", category_count)
print()
print("SQL results saved to:")
print(output_file)

# ------------------------------
# Close database
# ------------------------------
connection.close()

print()
print("==============================")
print("STEP 19 COMPLETE")
print("==============================")