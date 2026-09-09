import sqlite3
import pandas as pd
import os

# ------------------------------
# Database path
# ------------------------------
db_file = "data_pipeline/data/zepto_books.db"

# Check database exists
if not os.path.exists(db_file):
    print("ERROR: Database file not found.")
    exit()

# ------------------------------
# Connect to SQLite
# ------------------------------
connection = sqlite3.connect(db_file)

# Enable foreign keys
connection.execute("PRAGMA foreign_keys = ON")

print("==============================")
print("PANDAS READ_SQL")
print("==============================")
print()

# ==========================================================
# QUERY 1
# Read books with rating >= 4
# ==========================================================

query_1 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
ORDER BY rating DESC
"""

result_1 = pd.read_sql(query_1, connection)

print("QUERY 1: Books with rating >= 4")
print("--------------------------------")
print(result_1.head(10).to_string(index=False))
print()
print("Rows returned:", len(result_1))
print()

# ==========================================================
# QUERY 2
# Read JOIN result
# ==========================================================

query_2 = """
SELECT
    books.title,
    books.price_gbp,
    books.price_inr,
    books.rating,
    books.in_stock,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
ORDER BY books.price_gbp DESC
LIMIT 10
"""

result_2 = pd.read_sql(query_2, connection)

print("QUERY 2: Books + Categories JOIN")
print("---------------------------------")
print(result_2.to_string(index=False))
print()

print("Rows returned:", len(result_2))
print()

# ==========================================================
# Data types verification
# ==========================================================

print("==============================")
print("DATA TYPES")
print("==============================")
print(result_2.dtypes)
print()

# ==========================================================
# Save outputs
# ==========================================================

output_file = "data_pipeline/data/pandas_sql_results.txt"

with open(output_file, "w", encoding="utf-8") as file:

    file.write("==============================\n")
    file.write("PANDAS READ_SQL RESULTS\n")
    file.write("==============================\n\n")

    file.write("QUERY 1: Books with rating >= 4\n")
    file.write("--------------------------------\n")
    file.write(result_1.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 2: Books + Categories JOIN\n")
    file.write("---------------------------------\n")
    file.write(result_2.to_string(index=False))
    file.write("\n\n")

    file.write("==============================\n")
    file.write("DATA TYPES\n")
    file.write("==============================\n")
    file.write(result_2.dtypes.to_string())

# ------------------------------
# Close connection
# ------------------------------
connection.close()

print("==============================")
print("STEP 20 COMPLETE")
print("==============================")
print()
print("Results saved to:")
print(output_file)