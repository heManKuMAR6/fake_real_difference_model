import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
print("Loading ResNet CNN model and tokenizer...")
model = load_model('resnet_cnn_model.h5')
with open('resnet_cnn_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("✅ Model and tokenizer loaded!\n")

# Test sentences
test_sentences = [
    # Real news examples
    ("Apple Inc. announced today that it will invest $100 million in renewable energy infrastructure across its global manufacturing facilities.", "REAL"),
    ("The World Health Organization reported that vaccination rates have increased by 15% globally in the past year.", "REAL"),
    ("Tesla's stock price surged 12% following the company's announcement of record quarterly revenue of $24.3 billion.", "REAL"),
    ("Scientists published a new study in Nature Medicine showing breakthrough treatment for Alzheimer's disease.", "REAL"),
    
    # Fake news examples
    ("BREAKING: NASA confirms aliens landed at the White House last night! Scientists say this is the most important discovery in human history.", "FAKE"),
    ("New Study PROVES that drinking coffee cures cancer completely! Doctors HATE this simple one-trick cure!", "FAKE"),
    ("Celebrity A is CONFIRMED DEAD in mysterious accident! The government is covering it up!", "FAKE"),
    ("EXCLUSIVE: Secret email proves the entire election was rigged! Anonymous hackers found documents!", "FAKE"),
]

print("=" * 80)
print("TESTING RESNET CNN MODEL")
print("=" * 80)
print()

for idx, (sentence, label) in enumerate(test_sentences, 1):
    # Tokenize
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=300, padding='post')
    
    # Predict
    pred_prob = model.predict(padded, verbose=0)[0][0]
    prediction = 1 if pred_prob > 0.5 else 0
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
    
    predicted_label = "🟢 REAL" if prediction == 1 else "🔴 FAKE"
    actual_label = "REAL" if label == "REAL" else "FAKE"
    match = "✅" if (prediction == 1 and label == "REAL") or (prediction == 0 and label == "FAKE") else "❌"
    
    print(f"Test {idx}: {match}")
    print(f"Sentence: {sentence[:70]}...")
    print(f"Prediction: {predicted_label} (Confidence: {confidence*100:.2f}%)")
    print(f"Actual: {actual_label}")
    print()

print("=" * 80)
print("\n📝 EXAMPLE TEST SENTENCES FOR YOUR STREAMLIT APP:\n")

print("✅ REAL NEWS EXAMPLES (should predict 🟢 REAL):")
print("-" * 80)
print("""
1. "Apple Inc. announced today that it will invest $100 million in renewable 
   energy infrastructure across its global manufacturing facilities."

2. "The World Health Organization reported that vaccination rates have increased 
   by 15% globally in the past year."

3. "Tesla's stock price surged 12% following the company's announcement of record 
   quarterly revenue of $24.3 billion."

4. "Scientists published a new study in Nature Medicine showing breakthrough 
   treatment for Alzheimer's disease."
""")

print("\n❌ FAKE NEWS EXAMPLES (should predict 🔴 FAKE):")
print("-" * 80)
print("""
1. "BREAKING: NASA confirms aliens landed at the White House last night! 
   Scientists say this is the most important discovery in human history."

2. "New Study PROVES that drinking coffee cures cancer completely! 
   Doctors HATE this simple one-trick cure!"

3. "Celebrity A is CONFIRMED DEAD in mysterious accident! The government 
   is covering it up!"

4. "EXCLUSIVE: Secret email proves the entire election was rigged! 
   Anonymous hackers found documents!"
""")

print("\n" + "=" * 80)
print("💡 TIPS FOR TESTING:")
print("=" * 80)
print("""
✓ Copy any of the above sentences into the Streamlit app
✓ Make sure to check "ResNet CNN" checkbox
✓ Click "Predict" to see results
✓ Try mixing real and fake news
✓ Try your own news articles
✓ See how the model compares with other models
""")
