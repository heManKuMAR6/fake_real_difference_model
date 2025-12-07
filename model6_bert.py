import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===========================================================
# 1. LOAD DATA
# ===========================================================
print("Loading data...")
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42)

texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# ===========================================================
# 2. TOKENIZER
# ===========================================================
print("Loading BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ===========================================================
# 3. CUSTOM DATASET
# ===========================================================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx])
        }
        return item

train_dataset = NewsDataset(X_train, y_train, tokenizer)
test_dataset = NewsDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# ===========================================================
# 4. MODEL - FULL BERT (more powerful than DistilBERT)
# ===========================================================
print("Loading BERT model...")
device = torch.device("cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ===========================================================
# 5. TRAIN (light training)
# ===========================================================
print("Training BERT model...")
model.train()
steps = 60  # More steps than DistilBERT for better learning
count = 0

for batch in train_loader:
    if count >= steps:
        break
    
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"Step {count+1}/{steps} | Loss: {loss.item():.4f}")
    count += 1

# ===========================================================
# 6. EVALUATE
# ===========================================================
print("Evaluating BERT model...")
model.eval()
preds, true_vals = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1)

        preds.extend(pred.cpu().numpy())
        true_vals.extend(batch["labels"].numpy())

accuracy = accuracy_score(true_vals, preds)
precision = precision_score(true_vals, preds)
recall = recall_score(true_vals, preds)
f1 = f1_score(true_vals, preds)

print("\n=== FULL BERT RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ===========================================================
# 7. SAVE MODEL
# ===========================================================
print("Saving BERT model...")
model.save_pretrained("./bert_model")
tokenizer.save_pretrained("./bert_tokenizer")

print("BERT model + Tokenizer saved successfully!")
