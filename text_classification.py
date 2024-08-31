
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups

# Load dataset
data = fetch_20newsgroups(subset='train', categories=['rec.autos', 'sci.space'])
X, y = data.data, data.target

# Convert text data to embeddings
vectorizer = CountVectorizer(max_features=1000)
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
