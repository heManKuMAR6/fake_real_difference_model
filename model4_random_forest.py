import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. LOAD DATA
# =========================
print("Loading data...")
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Assign labels
true_df["label"] = 1   # Real news
fake_df["label"] = 0   # Fake news

# Combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42)  # Shuffle

# Prepare features and labels
X = df["text"].astype(str)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# =========================
# 2. TF-IDF VECTORIZATION
# =========================
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 3. RANDOM FOREST MODEL
# =========================
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train_vec, y_train)

# =========================
# 4. EVALUATE
# =========================
print("Evaluating model...")
y_pred = rf_model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== RANDOM FOREST RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =========================
# 5. SAVE MODEL
# =========================
print("\nSaving model and vectorizer...")
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Random Forest model and vectorizer saved successfully!")
