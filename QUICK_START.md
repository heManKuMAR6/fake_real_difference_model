# ⚡ QUICK REFERENCE - Testing Models in Streamlit

## 🚀 Start Testing (30 seconds)

```powershell
cd "C:\Users\Siva Sai Anangi\Desktop\trust"
& ".\venv\Scripts\Activate.ps1"
streamlit run app_multi_model.py
```

Browser opens → `http://localhost:8501`

## 📝 What to Do

1. **Paste news text** in the box
2. **Check models** you want to compare:
   - ✅ DistilBERT (fast, production)
   - ✅ Full BERT (most accurate)
   - ✅ Random Forest (baseline)
   - ☐ Simple CNN (optional)
3. **Click "🔍 Predict"**
4. **See results** side-by-side

## 📊 Results Show

```
╔─ DistilBERT ─╗
│ 🟢 REAL NEWS  │
│ 95.23% conf   │
└───────────────┘

╔─ Full BERT ──╗
│ 🟢 REAL NEWS  │
│ 98.45% conf   │
└───────────────┘

╔─ Random Forest ─╗
│ 🟢 REAL NEWS    │
│ 92.10% conf     │
└─────────────────┘

Consensus: REAL ✅
```

## 🎯 Test These Types

- ✅ Real news articles (from Reuters, AP, BBC, etc.)
- ✅ Obvious fake news (aliens, conspiracy theories)
- ✅ Borderline articles (opinion pieces, satire)
- ✅ Short snippets (headlines)
- ✅ Long articles (full news stories)

## 🔧 Two App Options

**Enhanced (Multi-Model):**

```bash
streamlit run app_multi_model.py
```

Compare 4 models, see confidence, consensus voting

**Original (Single Model):**

```bash
streamlit run app.py
```

Fast single model (DistilBERT only)

## ❌ Stop the App

Press `Ctrl+C` in terminal

## 🐛 Model Files Missing?

Train them first:

```bash
python model4_random_forest.py   # 10 sec
python model5_simple_cnn.py      # 21 sec
python model6_bert.py             # 3 min
```

## 📌 Model Quick Facts

| Model         | Time    | Accuracy | Confidence |
| ------------- | ------- | -------- | ---------- |
| DistilBERT    | Fast    | 99%+     | ⭐⭐⭐⭐   |
| Full BERT     | Medium  | 99.76%   | ⭐⭐⭐⭐⭐ |
| Random Forest | Instant | 99.4%    | ⭐⭐⭐     |
| Simple CNN    | Fast    | 99.35%   | ⭐⭐⭐⭐   |

## 💡 Pro Tip

**If all 4 models agree = You can be VERY confident in the result!**

## 📞 Common Issues

| Problem               | Solution                             |
| --------------------- | ------------------------------------ |
| "Model not found"     | Run training scripts first           |
| Port in use           | `Get-Process streamlit` then kill it |
| Slow first prediction | Model loading cached, next is faster |
| Memory error          | Close other apps, models use <1GB    |

---

**Total time to test:** 30 seconds ⏱️
