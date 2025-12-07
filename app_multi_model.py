import streamlit as st
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model as keras_load_model
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fake News Detector", layout="wide")

# ==========================================
# LOAD ALL MODELS (Cached)
# ==========================================

@st.cache_resource
def load_distilbert_model():
    """Load DistilBERT model (Production Model)"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained("./distilbert_model")
        tokenizer = AutoTokenizer.from_pretrained("./distilbert_tokenizer")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading DistilBERT: {e}")
        return None, None

@st.cache_resource
def load_bert_model():
    """Load Full BERT model"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained("./bert_model")
        tokenizer = AutoTokenizer.from_pretrained("./bert_tokenizer")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading BERT: {e}")
        return None, None

@st.cache_resource
def load_random_forest_model():
    """Load Random Forest + TF-IDF"""
    try:
        with open('random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return rf_model, vectorizer
    except Exception as e:
        st.error(f"Error loading Random Forest: {e}")
        return None, None

@st.cache_resource
def load_resnet_cnn_model():
    """Load ResNet CNN model"""
    try:
        model = keras_load_model('resnet_cnn_model.h5')
        with open('resnet_cnn_tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        return None, None

# ==========================================
# HELPER FUNCTIONS FOR PREDICTION
# ==========================================

# ==========================================
# HELPER FUNCTIONS FOR PREDICTION
# ==========================================

def predict_distilbert(text, model, tokenizer):
    """Predict using DistilBERT"""
    if model is None or tokenizer is None:
        return None, None
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(
                enc["input_ids"],
                attention_mask=enc["attention_mask"]
            )
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        return prediction, confidence
    except Exception as e:
        return None, None

def predict_bert(text, model, tokenizer):
    """Predict using Full BERT"""
    if model is None or tokenizer is None:
        return None, None
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(
                enc["input_ids"],
                attention_mask=enc["attention_mask"]
            )
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        return prediction, confidence
    except Exception as e:
        return None, None

def predict_random_forest(text, rf_model, vectorizer):
    """Predict using Random Forest"""
    if rf_model is None or vectorizer is None:
        return None, None
    try:
        text_vec = vectorizer.transform([text])
        prediction = rf_model.predict(text_vec)[0]
        confidence = rf_model.predict_proba(text_vec).max()
        return prediction, confidence
    except Exception as e:
        return None, None

def predict_resnet_cnn(text, cnn_model, tokenizer, max_len=300):
    """Predict using ResNet CNN"""
    if cnn_model is None or tokenizer is None:
        return None, None
    try:
        # Tokenize and pad the text
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post')
        
        # Predict
        pred_prob = cnn_model.predict(padded, verbose=0)[0][0]
        prediction = 1 if pred_prob > 0.5 else 0
        confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
        
        return prediction, confidence
    except Exception as e:
        return None, None

# ==========================================
# STREAMLIT UI
# ==========================================

st.title("📰 Fake News Detection - Multi-Model Comparison")
st.markdown("---")
st.write("""
This app tests **4 different models** on the same news text to compare their predictions.
- **DistilBERT** (Current Production) - Fast & Accurate
- **Full BERT** - Highest Accuracy
- **Random Forest + TF-IDF** - Lightweight Baseline
- **ResNet CNN** - Deep Learning with Residual Connections
""")

st.markdown("---")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("📝 Enter news text here:", height=200, placeholder="Paste your news article here...")

with col2:
    st.write("**Text Length:**")
    text_length = len(text.split())
    st.metric("Words", text_length)

st.markdown("---")

# Model selection
st.write("**Select Models to Compare:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    test_distilbert = st.checkbox("DistilBERT", value=True)
with col2:
    test_bert = st.checkbox("Full BERT", value=True)
with col3:
    test_rf = st.checkbox("Random Forest", value=True)
with col4:
    test_resnet = st.checkbox("ResNet CNN", value=False)

st.markdown("---")

# Predict button
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    if len(text.strip()) == 0:
        st.warning("⚠️ Please enter some news text first!")
    else:
        # Load models
        st.info("🔄 Loading models...")
        
        distilbert_model, distilbert_tokenizer = load_distilbert_model()
        bert_model, bert_tokenizer = load_bert_model()
        rf_model, vectorizer = load_random_forest_model()
        resnet_cnn_model, resnet_cnn_tokenizer = load_resnet_cnn_model()
        
        st.success("✅ Models loaded!")
        
        # Get predictions
        results = {}
        
        if test_distilbert and distilbert_model is not None:
            pred, conf = predict_distilbert(text, distilbert_model, distilbert_tokenizer)
            if pred is not None:
                results['DistilBERT'] = {
                    'prediction': pred,
                    'confidence': conf,
                    'label': "🟢 REAL" if pred == 1 else "🔴 FAKE"
                }
        
        if test_bert and bert_model is not None:
            pred, conf = predict_bert(text, bert_model, bert_tokenizer)
            if pred is not None:
                results['Full BERT'] = {
                    'prediction': pred,
                    'confidence': conf,
                    'label': "🟢 REAL" if pred == 1 else "🔴 FAKE"
                }
        
        if test_rf and rf_model is not None:
            pred, conf = predict_random_forest(text, rf_model, vectorizer)
            if pred is not None:
                results['Random Forest'] = {
                    'prediction': pred,
                    'confidence': conf,
                    'label': "🟢 REAL" if pred == 1 else "🔴 FAKE"
                }
        
        if test_resnet and resnet_cnn_model is not None and resnet_cnn_tokenizer is not None:
            pred, conf = predict_resnet_cnn(text, resnet_cnn_model, resnet_cnn_tokenizer)
            if pred is not None:
                results['ResNet CNN'] = {
                    'prediction': pred,
                    'confidence': conf,
                    'label': "🟢 REAL" if pred == 1 else "🔴 FAKE"
                }
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        if not results:
            st.error("❌ No models available for prediction!")
        else:
            # Display results in columns
            cols = st.columns(len(results))
            for idx, (model_name, result) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"### {model_name}")
                    
                    # Display prediction
                    if result['prediction'] == 1:
                        st.success(f"**{result['label']}**")
                        st.metric("Confidence", f"{result['confidence']*100:.2f}%", delta=f"Real News")
                    else:
                        st.error(f"**{result['label']}**")
                        st.metric("Confidence", f"{result['confidence']*100:.2f}%", delta=f"Fake News")
            
            st.markdown("---")
            
            # Summary
            st.subheader("📈 Summary")
            real_count = sum(1 for r in results.values() if r['prediction'] == 1)
            fake_count = len(results) - real_count
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Predicting REAL", real_count)
            with col2:
                st.metric("Models Predicting FAKE", fake_count)
            with col3:
                consensus = "REAL" if real_count > fake_count else "FAKE" if fake_count > real_count else "SPLIT"
                st.metric("Consensus", consensus)

st.markdown("---")
st.sidebar.info("""
### 📌 About the Models

**DistilBERT**: Fast transformer, 99%+ accuracy. Your current production model.

**Full BERT**: More powerful transformer, highest accuracy (99.76%).

**Random Forest**: Lightweight ML baseline, instant predictions (99.4%).

**ResNet CNN**: Neural network with residual connections, excellent accuracy (99.35%).

Test different models to see how they compare!
""")
