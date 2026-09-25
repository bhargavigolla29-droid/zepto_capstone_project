SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY c.category_name, b.rating DESC, b.title LIMIT 10;
