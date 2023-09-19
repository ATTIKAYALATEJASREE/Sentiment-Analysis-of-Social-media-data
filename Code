# Sample code for data collection
import pandas as pd

# Assuming you have a CSV file with social media data
data = pd.read_csv('social_media_data.csv')
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords and tokenizer data
nltk.download('stopwords')
nltk.download('punkt')

# Remove special characters and lowercase text
data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).lower())

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data['text'])
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, data['sentiment'], test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report, accuracy_score

y_pred = nb_classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
