# 🚀 HOW TO RUN THE FAKE NEWS DETECTION APP LOCALLY

## Quick Start (3 Steps)

### Step 1: Open PowerShell

Press `Win + R` and type:

```
powershell
```

Then press Enter.

### Step 2: Navigate to the Project

Copy and paste this command:

```powershell
cd "C:\Users\Siva Sai Anangi\Desktop\trust"
```

Then press Enter.

### Step 3: Start the App

Copy and paste this command:

```powershell
& ".\venv\Scripts\Activate.ps1"; python -m streamlit run app_multi_model.py
```

Then press Enter.

---

## What Happens Next?

You'll see output like:

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.118:8501
```

---

## Step 4: Open in Browser

### Option A: Click the Link

- The terminal will show `http://localhost:8501`
- Copy it and paste in your browser

### Option B: Manual

- Open your web browser (Chrome, Firefox, Edge, etc.)
- Go to: **http://localhost:8501**

---

## 🎯 You're Ready!

Once the browser opens, you'll see:

1. **Text area** - Paste news text here
2. **Model checkboxes** - Select which models to test
3. **Predict button** - Click to get results
4. **Results** - See predictions from all 4 models

---

## 📝 Example Usage

### 1. Paste a sentence:

```
Apple Inc. announced today that it will invest $100 million in renewable
energy infrastructure across its global manufacturing facilities.
```

### 2. Check models:

- ✅ DistilBERT
- ✅ Full BERT
- ✅ Random Forest
- ✅ ResNet CNN

### 3. Click "🔍 Predict"

### 4. See results:

```
DistilBERT   : FAKE (56.21%)
Full BERT    : FAKE (74.78%)
Random Forest: FAKE (77.25%)
ResNet CNN   : FAKE (100.00%)
Consensus    : FAKE
```

---

## ❌ To Stop the App

In PowerShell, press:

```
Ctrl + C
```

Then type `Y` and press Enter.

---

## 🔄 To Run Again

Just run this command again:

```powershell
& ".\venv\Scripts\Activate.ps1"; python -m streamlit run app_multi_model.py
```

---

## 🐛 Troubleshooting

### Port Already in Use?

```powershell
Get-Process streamlit | Stop-Process -Force
```

Then run the app again.

### Models Not Loading?

Make sure these files exist:

- `distilbert_model/`
- `bert_model/`
- `random_forest_model.pkl`
- `resnet_cnn_model.h5`

### Still Having Issues?

Run this to verify the setup:

```powershell
python test_all_models.py
```

---

## 📱 Access from Another Computer

If you want to access the app from another PC on your network, use:

```
http://192.168.1.118:8501
```

---

## ✅ All Set!

You now have a fully functional Fake News Detection app with 4 models!

**Start it now with:**

```powershell
& ".\venv\Scripts\Activate.ps1"; python -m streamlit run app_multi_model.py
```

Then visit: **http://localhost:8501**
