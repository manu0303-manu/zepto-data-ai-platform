import sqlite3
import pandas as pd
import os

# ------------------------------
# Database path
# ------------------------------
db_file = "data_pipeline/data/zepto_books.db"

if not os.path.exists(db_file):
    print("ERROR: Database file not found.")
    exit()

# ------------------------------
# Connect to SQLite
# ------------------------------
connection = sqlite3.connect(db_file)

print("==============================")
print("PANDAS MERGE")
print("==============================")
print()

# ==========================================================
# 1. Read books table into Pandas
# ==========================================================

books_query = """
SELECT
    book_id,
    title,
    price_gbp,
    price_inr,
    rating,
    in_stock,
    category_id
FROM books
"""

books_df = pd.read_sql(books_query, connection)

print("BOOKS DATAFRAME")
print("----------------")
print(books_df.head(5).to_string(index=False))
print()

print("Books rows:", len(books_df))
print()

# ==========================================================
# 2. Read categories table into Pandas
# ==========================================================

categories_query = """
SELECT
    category_id,
    category_name
FROM categories
"""

categories_df = pd.read_sql(categories_query, connection)

print("CATEGORIES DATAFRAME")
print("--------------------")
print(categories_df.head(10).to_string(index=False))
print()

print("Category rows:", len(categories_df))
print()

# ==========================================================
# 3. Reproduce SQL JOIN using pd.merge()
# ==========================================================

merge_result = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

# Sort exactly like the SQL JOIN query
merge_result = merge_result.sort_values(
    by="price_gbp",
    ascending=False
)

# Select the same columns as SQL JOIN
merge_result = merge_result[
    [
        "title",
        "price_gbp",
        "price_inr",
        "category_name"
    ]
].head(10)

print("==============================")
print("PANDAS MERGE RESULT")
print("==============================")
print(merge_result.to_string(index=False))
print()

# ==========================================================
# 4. SQL JOIN result for comparison
# ==========================================================

sql_join_query = """
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

sql_result = pd.read_sql(sql_join_query, connection)

print("==============================")
print("SQL JOIN RESULT")
print("==============================")
print(sql_result.to_string(index=False))
print()

# ==========================================================
# 5. Compare SQL JOIN and Pandas merge
# ==========================================================

sql_comparison = sql_result.reset_index(drop=True)
merge_comparison = merge_result.reset_index(drop=True)

# Make sure numeric values have the same precision
sql_comparison["price_gbp"] = sql_comparison["price_gbp"].round(2)
sql_comparison["price_inr"] = sql_comparison["price_inr"].round(2)

merge_comparison["price_gbp"] = merge_comparison["price_gbp"].round(2)
merge_comparison["price_inr"] = merge_comparison["price_inr"].round(2)

results_match = sql_comparison.equals(merge_comparison)

print("==============================")
print("COMPARISON")
print("==============================")
print("SQL JOIN rows:", len(sql_comparison))
print("Pandas merge rows:", len(merge_comparison))
print("Results match:", results_match)
print()

if results_match:
    print("SUCCESS: SQL JOIN and Pandas merge produce the same result.")
else:
    print("WARNING: SQL JOIN and Pandas merge results do not match.")

# ==========================================================
# 6. Save results
# ==========================================================

output_file = "data_pipeline/data/pandas_merge_results.txt"

with open(output_file, "w", encoding="utf-8") as file:

    file.write("==============================\n")
    file.write("PANDAS MERGE RESULT\n")
    file.write("==============================\n")
    file.write(merge_result.to_string(index=False))
    file.write("\n\n")

    file.write("==============================\n")
    file.write("SQL JOIN RESULT\n")
    file.write("==============================\n")
    file.write(sql_result.to_string(index=False))
    file.write("\n\n")

    file.write("==============================\n")
    file.write("COMPARISON\n")
    file.write("==============================\n")
    file.write(f"SQL JOIN rows: {len(sql_comparison)}\n")
    file.write(f"Pandas merge rows: {len(merge_comparison)}\n")
    file.write(f"Results match: {results_match}\n")

# ------------------------------
# Close connection
# ------------------------------
connection.close()

print("==============================")
print("STEP 21 COMPLETE")
print("==============================")
print()
print("Results saved to:")
print(output_file)