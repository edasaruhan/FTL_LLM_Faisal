
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Load dataset and prepare embeddings
data = fetch_20newsgroups(subset='all', categories=['rec.autos', 'sci.space'])
documents = data.data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(documents).toarray()

# Build FAISS index
dimension = X.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(X.astype(np.float32))

# Search interface
query = input("Enter your search query: ")
query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
_, indices = index.search(query_vector, k=5)
print("Top 5 documents for your query:")
for idx in indices[0]:
    print(documents[idx])
