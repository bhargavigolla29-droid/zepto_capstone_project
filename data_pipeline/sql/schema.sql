PRAGMA foreign_keys = ON;
CREATE TABLE categories (category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE NOT NULL);
CREATE TABLE books (
 book_id INTEGER PRIMARY KEY,
 title TEXT NOT NULL,
 price_gbp REAL NOT NULL,
 price_inr REAL NOT NULL,
 rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
 in_stock INTEGER NOT NULL CHECK(in_stock IN (0,1)),
 category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
