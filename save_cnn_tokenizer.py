import pickle
import json
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the tokenizer that was used during training
try:
    # Try to load from JSON (more efficient)
    with open('cnn_tokenizer_config.json', 'r') as f:
        tokenizer_json = json.load(f)
    
    tokenizer = Tokenizer()
    tokenizer.word_index = tokenizer_json['word_index']
    tokenizer.word_counts = tokenizer_json['word_counts']
    tokenizer.word_docs = tokenizer_json['word_docs']
    tokenizer.filters = tokenizer_json['filters']
    tokenizer.split = tokenizer_json['split']
    tokenizer.lower = tokenizer_json['lower']
    tokenizer.num_words = tokenizer_json['num_words']
    tokenizer.document_count = tokenizer_json['document_count']
    tokenizer.oov_token = tokenizer_json.get('oov_token', None)
    
    with open('cnn_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("Tokenizer converted and saved!")
except Exception as e:
    print(f"Error: {e}")
