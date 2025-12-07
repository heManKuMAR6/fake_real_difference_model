# 🎯 HOW TO TEST MODELS FROM STREAMLIT

## Option 1: Enhanced Multi-Model Comparison App (Recommended)

### Start the App:

```powershell
cd "C:\Users\Siva Sai Anangi\Desktop\trust"
& ".\venv\Scripts\Activate.ps1"
python -m streamlit run app_multi_model.py
```

### What You Can Do:

1. **Enter any news text** in the text area
2. **Select which models to compare** (checkboxes on the left):
   - ✅ DistilBERT (Default - Production Model)
   - ✅ Full BERT (Default - Best Accuracy)
   - ✅ Random Forest (Default - Fast Baseline)
   - ☐ Simple CNN (Optional)
3. **Click "🔍 Predict"** button
4. **Compare results** side-by-side:
   - Each model shows its prediction (REAL/FAKE)
   - Confidence score displayed
   - Summary shows consensus across models

### Key Features:

- **Multi-Model Comparison**: See how different models vote on the same text
- **Confidence Scores**: Know how certain each model is
- **Consensus View**: See if models agree or disagree
- **Word Count**: Shows length of your text
- **Sidebar Info**: Details about each model

---

## Option 2: Original Single Model App

### Start the Original App:

```powershell
cd "C:\Users\Siva Sai Anangi\Desktop\trust"
& ".\venv\Scripts\Activate.ps1"
python -m streamlit run app.py
```

### What You Can Do:

- Simple, fast testing with **DistilBERT only**
- Enter news text
- Get instant REAL/FAKE prediction
- Good for quick checks

---

## 🧪 Testing Workflow

### Step 1: Start the App

Use the enhanced app for better testing:

```bash
streamlit run app_multi_model.py
```

### Step 2: Open in Browser

The app will show:

```
Streamlit app started at http://localhost:8501
```

### Step 3: Test with Sample News

**Example 1 - REAL NEWS:**

```
Apple Inc. announced today that it will invest $100 million in renewable energy
infrastructure across its global manufacturing facilities. The company aims to
reduce its carbon footprint by 50% by 2025, according to CEO Tim Cook.
```

**Example 2 - FAKE NEWS:**

```
Breaking: NASA confirms aliens landed at the White House last night! Scientists
say this is the most important discovery in human history. The government has
been hiding this for 70 years according to anonymous sources.
```

### Step 4: Compare Predictions

- See which models agree/disagree
- Check confidence scores
- Look at the consensus verdict

---

## 📊 Model Performance Reminder

| Model             | Speed        | Accuracy | Best For                  |
| ----------------- | ------------ | -------- | ------------------------- |
| **DistilBERT**    | ⚡ Fast      | 99%+     | Production (current)      |
| **Full BERT**     | 🐢 Medium    | 99.76%   | Maximum accuracy          |
| **Random Forest** | ⚡⚡ Fastest | 99.4%    | Baseline comparison       |
| **Simple CNN**    | ⚡ Fast      | 99.35%   | Deep learning alternative |

---

## 🛠️ Troubleshooting

### Model Files Not Found

**Error:** `FileNotFoundError: random_forest_model.pkl`
**Solution:** Make sure you ran all training scripts first:

```bash
python model4_random_forest.py
python model5_simple_cnn.py
python model6_bert.py
```

### Out of Memory Error

**Error:** `RuntimeError: CUDA out of memory`
**Solution:** The app uses CPU by default, which is fine. Models use less than 1GB RAM.

### Port Already in Use

**Error:** `Address already in use`
**Solution:** Stop the previous streamlit process:

```bash
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 💡 Pro Tips

1. **Test with various text lengths** - See how models perform on short vs long articles

2. **Test controversial topics** - See if models correctly identify misinformation

3. **Compare model confidence** - High confidence on all models = strong signal

4. **Use consensus voting** - If 3/4 models say "FAKE", it's likely correct

5. **Save interesting results** - Screenshot disagreements to investigate why models differ

---

## 🎓 What Each Model Looks For

- **DistilBERT/BERT**: Context, semantics, language patterns from pre-training
- **Random Forest**: Keyword frequency, TF-IDF patterns from training data
- **Simple CNN**: Sequential word patterns and local context windows

Different models → Different strengths → Better combined predictions

---

## 📝 Quick Start

```bash
# Activate environment
& ".\venv\Scripts\Activate.ps1"

# Run multi-model app
streamlit run app_multi_model.py

# Visit: http://localhost:8501
# Paste news text → Click Predict → Compare results!
```
