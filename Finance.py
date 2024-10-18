import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import re

import nltk

nltk.download('stopwords')

df = pd.read_csv("E:/YBI Project/Finance.csv", encoding='ISO-8859-1')

df.dropna(subset=['Label'], inplace=True)

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  
    text = text.lower()   
    text = re.sub(r'\s+', ' ', text) 
    return text

news = [' '.join(str(x) for x in df.iloc[row, 2:27]) for row in range(len(df.index))]
cleaned_news = [clean_text(article) for article in news]

X = cleaned_news
y = df['Label']

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Bigrams for more context
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2529)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=2529)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))
