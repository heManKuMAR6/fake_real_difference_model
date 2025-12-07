import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. LOAD AND PREPARE DATA
# =========================
print("Loading data...")
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Assign labels
true_df["label"] = 1
fake_df["label"] = 0

# Combine and shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42)

texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# =========================
# 2. TOKENIZATION & PADDING
# =========================
print("Tokenizing texts...")
vocab_size = 5000
max_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

y_train = np.array(y_train)
y_test = np.array(y_test)

# =========================
# 3. SAVE TOKENIZER FOR INFERENCE
# =========================
print("Saving tokenizer for inference...")
import pickle
with open('cnn_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# =========================
# 4. SIMPLE CNN MODEL (FAST)
# =========================
print("Building simple CNN model...")
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    Conv1D(64, 5, activation='relu', padding='same'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# =========================
# 4. TRAIN MODEL (FAST)
# =========================
print("Training CNN model...")
history = model.fit(
    X_train_pad, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# =========================
# 5. EVALUATE
# =========================
print("Evaluating model...")
y_pred_prob = model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== SIMPLE CNN RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =========================
# 6. SAVE MODEL
# =========================
print("Saving model...")
model.save('simple_cnn_model.h5')
print("Simple CNN model saved successfully!")
