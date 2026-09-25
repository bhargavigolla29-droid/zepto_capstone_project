# Data Pipeline

Implements the required scrape → clean → convert → relational store → SQL/pandas workflow. The pipeline uses the first five all-products pages (100 books), satisfying the ≥60-book acceptance threshold. Category is read from each book's detail-page breadcrumb because the catalogue listing card does not itself contain the category.

Cleaning decisions:
- `price_gbp`: strip `£`, parse as float; numeric parse failures use median.
- `rating`: map One–Five to 1–5; numeric parse failures use median.
- `in_stock`: boolean from the availability text.
- missing title/category: row dropped because those fields identify the entity and cannot be safely imputed.
- `price_inr = price_gbp * 105.50` exactly; this is the assignment's artificial fixed rate, not a live FX lookup.

The SQLite schema has `categories(category_id)` and `books(category_id)` with a foreign key. Five saved queries cover SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN and JOIN. The join is independently reproduced with `pd.merge`.

Implementation note: generated outputs are supporting artifacts; rerun the module scripts to refresh them.
