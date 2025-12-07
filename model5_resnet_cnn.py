import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Input, Add, Flatten, Concatenate
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
vocab_size = 10000
max_length = 300

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

y_train = np.array(y_train)
y_test = np.array(y_test)

# =========================
# SAVE TOKENIZER FOR INFERENCE
# =========================
print("Saving tokenizer for inference...")
import pickle
with open('resnet_cnn_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# =========================
# 3. BUILD RESIDUAL CNN MODEL
# =========================
print("Building Residual CNN model...")

def residual_cnn():
    """
    Residual CNN with skip connections for text classification
    """
    input_layer = Input(shape=(max_length,))
    
    # Embedding layer
    embedding = Embedding(vocab_size, 128, input_length=max_length)(input_layer)
    
    # First CNN block with residual connection
    conv1 = Conv1D(64, 3, activation='relu', padding='same')(embedding)
    conv1 = Conv1D(64, 3, activation='relu', padding='same')(conv1)
    
    # Project embedding to match conv1 dimensions if needed
    embedding_proj = Dense(64)(embedding)
    residual1 = Add()([conv1, embedding_proj])
    pool1 = MaxPooling1D(3)(residual1)
    
    # Second CNN block with residual connection
    conv2 = Conv1D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(128, 3, activation='relu', padding='same')(conv2)
    
    # Project pool1 to match conv2 dimensions
    pool1_proj = Dense(128)(pool1)
    residual2 = Add()([conv2, pool1_proj])
    pool2 = MaxPooling1D(3)(residual2)
    
    # Third CNN block with residual connection
    conv3 = Conv1D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(256, 3, activation='relu', padding='same')(conv3)
    
    # Project pool2 to match conv3 dimensions
    pool2_proj = Dense(256)(pool2)
    residual3 = Add()([conv3, pool2_proj])
    pool3 = MaxPooling1D(3)(residual3)
    
    # Flatten and dense layers
    flatten = Flatten()(pool3)
    dense1 = Dense(128, activation='relu')(flatten)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    output = Dense(1, activation='sigmoid')(dropout2)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

model = residual_cnn()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# =========================
# 4. TRAIN MODEL
# =========================
print("Training Residual CNN model...")
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

print("\n=== RESIDUAL CNN RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =========================
# 6. SAVE MODEL
# =========================
print("Saving model...")
model.save('resnet_cnn_model.h5')
print("Residual CNN model saved successfully!")
