import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.tokenize import word_tokenize
import nltk

# Load the CSV file
csv_file_path = r'C:\Users\91846\Downloads\archive (5)\spam.csv'  # Update the path to the CSV file
data = pd.read_csv(csv_file_path, encoding='latin-1')

# Data Preprocessing
data.drop_duplicates(inplace=True)
data.dropna(subset=['v1'], inplace=True)
data.reset_index(drop=True, inplace=True)

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['v1'])
y = data['v2']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Model Evaluation
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Testing with New Emails
def test_new_email(email_text):
    # Preprocess the new email
    preprocessed_email = preprocess_email(email_text)

    
    new_email_features = tfidf_vectorizer.transform([preprocessed_email])

    # Prediction
    prediction = classifier.predict(new_email_features)

    if prediction[0] == 'spam':
        print("This is a spam email.")
    else:
        print("This is not a spam email.")

# Preprocess New Emails
def preprocess_email(email_text):
    # Remove special characters and numbers
    email_text = re.sub(r'[^a-zA-Z\s]', '', email_text)

    # Convert to lowercase
    email_text = email_text.lower()

    words = word_tokenize(email_text)

    preprocessed_email = ' '.join(words)

    return preprocessed_email

nltk.download('punkt')

# Example usage:
new_email_text = "Congratulations! You've won a million dollars. Click here to claim your prize!"
test_new_email(new_email_text)


import joblib

model_filename = 'spam_classifier_model.pkl'
joblib.dump(classifier, model_filename)


loaded_classifier = joblib.load(model_filename)


print(data['v2'].value_counts())
