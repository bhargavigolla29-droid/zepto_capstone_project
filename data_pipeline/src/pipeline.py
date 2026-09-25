from __future__ import annotations
import re
import sqlite3
from pathlib import Path
from statistics import median
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "http://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "catalogue.db"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

RATING_MAP = {"One":1, "Two":2, "Three":3, "Four":4, "Five":5}


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0 capstone-scraper"})
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def scrape_pages(num_pages: int = 5) -> pd.DataFrame:
    rows = []
    for page_no in range(1, num_pages + 1):
        url = BASE_URL if page_no == 1 else f"{BASE_URL}catalogue/page-{page_no}.html"
        soup = get_soup(url)
        for card in soup.select("article.product_pod"):
            title = card.select_one("h3 a")["title"].strip()
            price = card.select_one(".price_color").get_text(strip=True)
            availability = card.select_one(".availability").get_text(" ", strip=True)
            rating_tag = card.select_one(".star-rating")
            rating_text = next((c for c in rating_tag.get("class", []) if c != "star-rating"), "")
            rows.append({"title": title, "price": price, "star_rating": rating_text.title(), "availability": availability})
    df = pd.DataFrame(rows)
    # All-products pages do not expose category in the listing card. Fetch each detail page and
    # read the breadcrumb category; this keeps the required requests+BeautifulSoup approach.
    categories = []
    for page_no in range(1, num_pages + 1):
        url = BASE_URL if page_no == 1 else f"{BASE_URL}catalogue/page-{page_no}.html"
        soup = get_soup(url)
        for card in soup.select("article.product_pod"):
            href = card.select_one("h3 a")["href"]
            detail_url = requests.compat.urljoin(url, href)
            detail = get_soup(detail_url)
            crumbs = detail.select("ul.breadcrumb li a")
            categories.append(crumbs[-1].get_text(strip=True) if crumbs else "Unknown")
    df["category"] = categories
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_gbp"] = pd.to_numeric(out["price"].str.replace("£", "", regex=False), errors="coerce")
    out["rating"] = out["star_rating"].map(RATING_MAP)
    out["in_stock"] = out["availability"].str.contains("in stock", case=False, na=False)
    # Numeric parse failures use median; categorical parse failures are dropped because there is
    # no defensible category/rating surrogate for a catalogue row.
    for col in ["price_gbp", "rating"]:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    out = out.dropna(subset=["title", "category"])
    out["price_inr"] = (out["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    out["rating"] = out["rating"].astype(int)
    out["in_stock"] = out["in_stock"].astype(bool)
    return out[["title","price_gbp","price_inr","rating","in_stock","category","star_rating","availability"]]


def load_sqlite(df: pd.DataFrame) -> None:
    if DB_PATH.exists(): DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            in_stock INTEGER NOT NULL CHECK(in_stock IN (0,1)),
            category_id INTEGER NOT NULL,
            FOREIGN KEY(category_id) REFERENCES categories(category_id)
        );
        """)
        cats = pd.DataFrame({"category_name": sorted(df["category"].unique())})
        cats.index = range(1, len(cats)+1)
        cats.to_sql("categories", con, if_exists="append", index_label="category_id")
        mapping = dict(zip(cats["category_name"], cats.index))
        books = df.assign(category_id=df["category"].map(mapping), in_stock=df["in_stock"].astype(int))
        books[["title","price_gbp","price_inr","rating","in_stock","category_id"]].to_sql("books", con, if_exists="append", index=False)


def run_queries() -> None:
    queries = {
        "01_select_where": "SELECT title, price_gbp FROM books WHERE price_gbp > 40 ORDER BY price_gbp DESC;",
        "02_order_limit": "SELECT title, rating FROM books ORDER BY rating DESC, title LIMIT 10;",
        "03_distinct": "SELECT DISTINCT category_name FROM categories ORDER BY category_name;",
        "04_between": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 20 ORDER BY price_gbp;",
        "05_join": "SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY c.category_name, b.rating DESC, b.title LIMIT 10;",
    }
    with sqlite3.connect(DB_PATH) as con:
        for name, q in queries.items():
            df = pd.read_sql(q, con)
            (OUT / f"{name}.sql").write_text(q + "\n", encoding="utf-8")
            df.to_csv(OUT / f"{name}.csv", index=False)
            print(f"\n-- {name} --\n{q}\n{df.to_string(index=False)}")
        join_sql = queries["05_join"]
        sql_join = pd.read_sql(join_sql, con)
        books = pd.read_sql("SELECT * FROM books", con)
        cats = pd.read_sql("SELECT * FROM categories", con)
        merge_join = books.merge(cats, on="category_id", how="inner")[["category_name","title","rating","price_inr"]].sort_values(["category_name","rating","title"], ascending=[True,False,True]).head(10)
        print("\nSQL JOIN vs pandas.merge equivalent:", sql_join.reset_index(drop=True).equals(merge_join.reset_index(drop=True)))
        pd.concat({"pd.read_sql":sql_join.reset_index(drop=True), "pd.merge":merge_join.reset_index(drop=True)}, axis=1).to_csv(OUT / "join_comparison.csv", index=False)


def main():
    raw = scrape_pages(5)
    if len(raw) < 60:
        raise RuntimeError(f"Scrape returned {len(raw)} rows; requirement is at least 60.")
    cleaned = clean(raw)
    cleaned.to_csv(OUT / "cleaned_books.csv", index=False)
    load_sqlite(cleaned)
    run_queries()
    print(f"Completed data pipeline: {len(cleaned)} books, {cleaned['category'].nunique()} categories, {DB_PATH}")

if __name__ == "__main__": main()
