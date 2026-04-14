# Preprocessing Notes

Steps applied:
1. Lowercase all text
2. Strip HTML tags (`<br/>`, `<b>`, etc.) — IMDB has these
3. Remove non-alphabetic characters
4. Normalize whitespace

Decision: kept stopwords in since TF-IDF down-weights common words anyway.
