import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys

# Fix encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("Loading all models...\n")

# Load DistilBERT
distilbert_model = AutoModelForSequenceClassification.from_pretrained("./distilbert_model")
distilbert_tokenizer = AutoTokenizer.from_pretrained("./distilbert_tokenizer")

# Load Full BERT
bert_model = AutoModelForSequenceClassification.from_pretrained("./bert_model")
bert_tokenizer = AutoTokenizer.from_pretrained("./bert_tokenizer")

# Load Random Forest
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load ResNet CNN
resnet_model = load_model('resnet_cnn_model.h5')
with open('resnet_cnn_tokenizer.pkl', 'rb') as f:
    resnet_tokenizer = pickle.load(f)

print("OK All models loaded!\n")

# Test sentences
test_sentences = [
    "Apple Inc. announced today that it will invest $100 million in renewable energy infrastructure across its global manufacturing facilities.",
    "The World Health Organization reported that vaccination rates have increased by 15% globally in the past year.",
    "Tesla's stock price surged 12% following the company's announcement of record quarterly revenue of $24.3 billion.",
    "Scientists published a new study in Nature Medicine showing breakthrough treatment for Alzheimer's disease.",
    "BREAKING: NASA confirms aliens landed at the White House last night! Scientists say this is the most important discovery in human history.",
    "New Study PROVES that drinking coffee cures cancer completely! Doctors HATE this simple one-trick cure!",
    "Celebrity A is CONFIRMED DEAD in mysterious accident! The government is covering it up!",
    "EXCLUSIVE: Secret email proves the entire election was rigged! Anonymous hackers found documents!",
]

print("=" * 100)
print("TESTING ALL 4 MODELS")
print("=" * 100)
print()

for idx, sentence in enumerate(test_sentences, 1):
    print(f"\n{'='*100}")
    print(f"Test Sentence {idx}:")
    print(f"{sentence[:80]}...")
    print(f"{'='*100}")
    
    # DistilBERT
    enc = distilbert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = distilbert_model(enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = outputs.logits
        pred_dist = torch.argmax(logits, dim=1).item()
        conf_dist = torch.softmax(logits, dim=1).max().item()
    
    # Full BERT
    enc = bert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = outputs.logits
        pred_bert = torch.argmax(logits, dim=1).item()
        conf_bert = torch.softmax(logits, dim=1).max().item()
    
    # Random Forest
    text_vec = vectorizer.transform([sentence])
    pred_rf = rf_model.predict(text_vec)[0]
    conf_rf = rf_model.predict_proba(text_vec).max()
    
    # ResNet CNN
    sequence = resnet_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=300, padding='post')
    pred_prob = resnet_model.predict(padded, verbose=0)[0][0]
    pred_resnet = 1 if pred_prob > 0.5 else 0
    conf_resnet = pred_prob if pred_prob > 0.5 else 1 - pred_prob
    
    # Display results
    print(f"DistilBERT    : {'REAL' if pred_dist == 1 else 'FAKE'} ({conf_dist*100:.2f}%)")
    print(f"Full BERT     : {'REAL' if pred_bert == 1 else 'FAKE'} ({conf_bert*100:.2f}%)")
    print(f"Random Forest : {'REAL' if pred_rf == 1 else 'FAKE'} ({conf_rf*100:.2f}%)")
    print(f"ResNet CNN    : {'REAL' if pred_resnet == 1 else 'FAKE'} ({conf_resnet*100:.2f}%)")
    
    # Consensus
    votes = [pred_dist, pred_bert, pred_rf, pred_resnet]
    real_votes = sum(votes)
    fake_votes = 4 - real_votes
    consensus = "REAL" if real_votes > 2 else "FAKE" if fake_votes > 2 else "SPLIT"
    print(f"Consensus     : {consensus} ({real_votes} REAL, {fake_votes} FAKE)")

print("\n\n" + "=" * 100)
print("📝 BEST TEST SENTENCES FOR YOUR STREAMLIT APP")
print("=" * 100)

print("""
🟢 REAL NEWS EXAMPLES TO TEST:

1. "Apple Inc. announced today that it will invest $100 million in renewable 
    energy infrastructure across its global manufacturing facilities."

2. "The World Health Organization reported that vaccination rates have increased 
    by 15% globally in the past year."

3. "Tesla's stock price surged 12% following the company's announcement of 
    record quarterly revenue of $24.3 billion."

4. "Scientists published a new study in Nature Medicine showing breakthrough 
    treatment for Alzheimer's disease."

🔴 FAKE NEWS EXAMPLES TO TEST:

1. "BREAKING: NASA confirms aliens landed at the White House last night! 
    Scientists say this is the most important discovery in human history."

2. "New Study PROVES that drinking coffee cures cancer completely! 
    Doctors HATE this simple one-trick cure!"

3. "Celebrity A is CONFIRMED DEAD in mysterious accident! 
    The government is covering it up!"

4. "EXCLUSIVE: Secret email proves the entire election was rigged! 
    Anonymous hackers found documents!"

💡 HOW TO USE IN STREAMLIT:
1. Go to http://localhost:8501
2. Copy any of the above sentences into the text area
3. Select which models you want to test
4. Click "Predict"
5. See how all 4 models vote
6. Compare their confidence scores
7. Look at the consensus verdict

⭐ TIP: If all 4 models agree = 99%+ confident!
""")
