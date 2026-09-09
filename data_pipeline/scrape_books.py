import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
import os


all_books = []


for page in range(1, 6):

    if page == 1:
        url = "http://books.toscrape.com/"
    else:
        url = f"http://books.toscrape.com/catalogue/page-{page}.html"

    response = requests.get(url, timeout=10)

    print("Page:", page, "| Status:", response.status_code)

    soup = BeautifulSoup(response.text, "html.parser")

    books = soup.select("article.product_pod")

    print("Books found:", len(books))

    for book in books:

        title = book.h3.a["title"]

        price = book.select_one(
            ".price_color"
        ).get_text(strip=True)

        availability = book.select_one(
            ".availability"
        ).get_text(" ", strip=True)

        rating = book.select_one(
            "p.star-rating"
        )["class"][1]

        book_url = urljoin(
            url,
            book.h3.a["href"]
        )

        book_response = requests.get(
            book_url,
            timeout=10
        )

        book_soup = BeautifulSoup(
            book_response.text,
            "html.parser"
        )

        breadcrumb = book_soup.select(
            "ul.breadcrumb li"
        )

        category = breadcrumb[2].get_text(
            strip=True
        )

        all_books.append({
            "title": title,
            "price": price,
            "availability": availability,
            "rating": rating,
            "category": category
        })


# Create data folder
data_folder = os.path.join(
    "data_pipeline",
    "data"
)

os.makedirs(
    data_folder,
    exist_ok=True
)


# CSV file path
csv_file = os.path.join(
    data_folder,
    "raw_books.csv"
)


# Save books to CSV
with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "title",
        "price",
        "availability",
        "rating",
        "category"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(all_books)


print("\n==============================")
print("TOTAL BOOKS:", len(all_books))
print("==============================")

print("CSV file saved successfully:")
print(csv_file)