import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

# 1. Load data
df = pd.read_csv('balanced_drugs_data.csv')

# 2. Preprocessing (Task 2)
X = df['review']
y = df['condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 3. MODEL TUNING (Task 5 - 10 Marks)
# Instead of just training once, we test different 'C' values
# 'C' controls how 'Aggressive' the model is when it makes a mistake.
param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0],
    'max_iter': [50, 100],
    'loss': ['hinge', 'squared_hinge']
}

print("Tuning Model... Testing combinations...")
grid_search = GridSearchCV(PassiveAggressiveClassifier(), param_grid, cv=5, verbose=1)
grid_search.fit(X_train_tfidf, y_train)

# 4. Results for your Report
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Improved Accuracy: {grid_search.best_score_:.2%}")

# 5. Save the tuned version (Task 4)
best_pac = grid_search.best_estimator_
joblib.dump(best_pac, 'medical_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Success! Tuned 'medical_model.pkl' created.")