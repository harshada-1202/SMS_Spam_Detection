#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('sms-spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Preprocess the data
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
predictions = model.predict(X_test_tfidf)

# Evaluation
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


# In[2]:


import pandas as pd
import numpy as np
import re
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
df = pd.read_csv('sms-spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Keep only relevant columns
df.columns = ['label', 'message']  # Rename columns

# Clean the text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Apply text cleaning
df['cleaned_message'] = df['message'].apply(clean_text)

# Prepare the model
X = df['cleaned_message']
y = df['label'].map({'ham': 0, 'spam': 1})  # Encode labels

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_tfidf, y)

# Function to predict spam or ham
def predict_spam():
    user_input = entry.get()
    cleaned_input = clean_text(user_input)
    input_tfidf = tfidf_vectorizer.transform([cleaned_input])
    prediction = model.predict(input_tfidf)
    
    if prediction[0] == 1:
        messagebox.showinfo("Result", "This message is SPAM!")
    else:
        messagebox.showinfo("Result", "This message is NOT SPAM.")

# Create the GUI
root = tk.Tk()
root.title("SMS Spam Detection")

# Create a label
label = tk.Label(root, text="Enter your SMS message:")
label.pack(pady=10)

# Create a text entry box
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Create a button to predict
button = tk.Button(root, text="Check Spam", command=predict_spam)
button.pack(pady=20)

# Run the GUI
root.mainloop()

