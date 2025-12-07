# 📚 COMPLETE GUIDE - Fake News Detection App

## 🎯 QUICK START (Copy & Paste)

### In PowerShell:

```powershell
cd "C:\Users\Siva Sai Anangi\Desktop\trust"
& ".\venv\Scripts\Activate.ps1"
python -m streamlit run app_multi_model.py
```

### Then Open:

```
http://localhost:8501
```

**That's it! The app will open in your browser.** ✅

---

## 📊 WHAT THE APP DOES

The app tests **4 different AI models** on the same news text:

1. **DistilBERT** - Fast transformer model
2. **Full BERT** - More powerful transformer
3. **Random Forest** - Traditional ML model
4. **ResNet CNN** - Deep learning with residual connections

All models predict: **REAL NEWS** or **FAKE NEWS**

---

## 🧪 TEST SENTENCES

### Real News (Copy & Paste):

```
Apple Inc. announced today that it will invest $100 million in renewable
energy infrastructure across its global manufacturing facilities.
```

```
The World Health Organization reported that vaccination rates have increased
by 15% globally in the past year.
```

```
Tesla's stock price surged 12% following the company's announcement of
record quarterly revenue of $24.3 billion.
```

```
Scientists published a new study in Nature Medicine showing breakthrough
treatment for Alzheimer's disease.
```

### Fake News (Copy & Paste):

```
BREAKING: NASA confirms aliens landed at the White House last night!
Scientists say this is the most important discovery in human history.
```

```
New Study PROVES that drinking coffee cures cancer completely!
Doctors HATE this simple one-trick cure!
```

```
Celebrity A is CONFIRMED DEAD in mysterious accident!
The government is covering it up!
```

```
EXCLUSIVE: Secret email proves the entire election was rigged!
Anonymous hackers found documents!
```

---

## 🚀 STEP-BY-STEP INSTRUCTIONS

### Step 1: Open PowerShell

- Press `Win + R`
- Type `powershell`
- Press Enter

### Step 2: Navigate to Project

```powershell
cd "C:\Users\Siva Sai Anangi\Desktop\trust"
```

### Step 3: Activate Environment

```powershell
& ".\venv\Scripts\Activate.ps1"
```

### Step 4: Start App

```powershell
python -m streamlit run app_multi_model.py
```

### Step 5: Open Browser

- Wait for message: `Local URL: http://localhost:8501`
- Open browser and go to: `http://localhost:8501`

---

## 💻 HOW TO USE THE APP

### In the Web Browser:

1. **Paste News Text**

   - Copy any news article or sentence
   - Paste into the text box at the top
   - You'll see word count on the right

2. **Select Models**

   - Check/uncheck which models to test:
     - ✅ DistilBERT
     - ✅ Full BERT
     - ✅ Random Forest
     - ✅ ResNet CNN

3. **Click Predict**

   - Click the blue "🔍 Predict" button
   - Wait for results (takes 2-5 seconds)

4. **See Results**
   - Each model shows:
     - Prediction: REAL or FAKE
     - Confidence: 0-100%
   - Bottom shows: Consensus verdict

---

## 📊 UNDERSTANDING RESULTS

### Example Output:

```
DistilBERT    : REAL (85.23%)
Full BERT     : REAL (92.15%)
Random Forest : REAL (88.50%)
ResNet CNN    : REAL (100.00%)
Consensus     : REAL (4 REAL, 0 FAKE)
```

### What It Means:

- **Percentage** = How confident the model is
- **Consensus** = What most models agree on
- **If all 4 agree** = You can be 99%+ confident!

---

## 🎓 MODEL PERFORMANCE

| Model             | Speed          | Accuracy | Confidence |
| ----------------- | -------------- | -------- | ---------- |
| **DistilBERT**    | Fast ⚡        | 99%+     | 52-73%     |
| **Full BERT**     | Medium 🐢      | 99.76%   | 85-93%     |
| **Random Forest** | Very Fast ⚡⚡ | 99.4%    | 71-77%     |
| **ResNet CNN**    | Fast ⚡        | 99.69%   | 100%       |

---

## ❌ HOW TO STOP THE APP

In the PowerShell window where the app is running:

1. Press `Ctrl + C`
2. Type `Y` (for Yes)
3. Press Enter

The app will stop.

---

## 🔄 HOW TO RUN AGAIN

Just repeat the steps above. Or use this one-liner:

```powershell
cd "C:\Users\Siva Sai Anangi\Desktop\trust"; & ".\venv\Scripts\Activate.ps1"; python -m streamlit run app_multi_model.py
```

---

## 🐛 TROUBLESHOOTING

### "Port 8501 already in use"

```powershell
Get-Process streamlit | Stop-Process -Force
```

Then run the app again.

### "Module not found" Error

Make sure the virtual environment is activated:

```powershell
& ".\venv\Scripts\Activate.ps1"
```

### "Models not found" Error

Verify these files exist in the project folder:

- `distilbert_model/` ✅
- `bert_model/` ✅
- `random_forest_model.pkl` ✅
- `resnet_cnn_model.h5` ✅

### Models Not Loading in App

Run this test:

```powershell
cd "C:\Users\Siva Sai Anangi\Desktop\trust"
& ".\venv\Scripts\Activate.ps1"
python test_all_models.py
```

---

## 📁 PROJECT FILES

### Main Files:

- `app_multi_model.py` - The Streamlit app ⭐
- `model3_distilbert.py` - DistilBERT training
- `model6_bert.py` - Full BERT training
- `model4_random_forest.py` - Random Forest training
- `model5_resnet_cnn.py` - ResNet CNN training

### Model Files:

- `distilbert_model/` - DistilBERT weights
- `bert_model/` - Full BERT weights
- `random_forest_model.pkl` - RF model
- `resnet_cnn_model.h5` - ResNet CNN weights

### Test Files:

- `test_all_models.py` - Compare all 4 models
- `TEST_CASES.md` - Example test sentences

---

## 💡 TIPS & TRICKS

1. **Test with Real Articles**

   - Copy from Reuters, AP, BBC, NPR
   - See how models perform on real news

2. **Test Edge Cases**

   - Satire and opinion pieces
   - Misleading headlines
   - Mixed real + fake facts

3. **Compare Models**

   - Notice which models agree/disagree
   - See confidence differences
   - Understand model strengths

4. **Access Remotely**
   - Use `http://192.168.1.118:8501`
   - Access from another computer on same network

---

## ✅ CHECKLIST

Before running the app:

- [ ] PowerShell installed
- [ ] Project folder exists
- [ ] Virtual environment created (`venv/`)
- [ ] All model files present
- [ ] Port 8501 not in use

Ready to run:

- [ ] Open PowerShell
- [ ] Navigate to project folder
- [ ] Activate virtual environment
- [ ] Run streamlit command
- [ ] Open browser to localhost:8501

---

## 🎉 YOU'RE READY!

Run this command now:

```powershell
cd "C:\Users\Siva Sai Anangi\Desktop\trust"; & ".\venv\Scripts\Activate.ps1"; python -m streamlit run app_multi_model.py
```

Then visit: **http://localhost:8501**

Enjoy testing the Fake News Detection App! 🚀
