# 🧪 TEST CASES FOR STREAMLIT APP

Copy and paste these test cases into the Streamlit app to verify model performance.

---

## ✅ REAL NEWS EXAMPLES

### Test Case 1: Technology News

```
Apple Inc. announced on Monday that it has achieved carbon neutrality in
its operations as of 2022. The company said it has reduced emissions by
75% since 2015 across its business and supply chain. Apple plans to achieve
complete carbon neutrality across its entire value chain by 2030, including
manufacturing and product transportation.
```

**Expected:** 🟢 REAL  
**Why:** Official corporate announcement, specific dates, realistic targets

---

### Test Case 2: Science News

```
A groundbreaking study published in Nature Medicine reveals that a newly
discovered gene variant significantly increases the risk of developing
Alzheimer's disease. The international research team analyzed genetic data
from over 100,000 participants. Researchers believe this discovery could
lead to new treatment approaches within the next five years.
```

**Expected:** 🟢 REAL  
**Why:** Journal citation, large sample size, measured claims

---

### Test Case 3: Business News

```
Tesla's stock price surged 12% today following the company's announcement
of record quarterly revenue of $24.3 billion. The company delivered 1.8 million
vehicles in 2023, exceeding analyst expectations. CEO Elon Musk attributed
the success to increased production capacity and growing demand for electric vehicles.
```

**Expected:** 🟢 REAL  
**Why:** Specific numbers, credible attributions, market-based reporting

---

### Test Case 4: Health News

```
The World Health Organization reported today that vaccination rates have
increased by 15% globally in the past year. New data shows that countries
with aggressive vaccination campaigns have seen a 40% reduction in preventable
disease outbreaks. Public health officials credit improved vaccine distribution
and public education campaigns for the increase.
```

**Expected:** 🟢 REAL  
**Why:** Cites authoritative source, uses percentages (not absolutes), measured claims

---

## 🔴 FAKE NEWS EXAMPLES

### Test Case 5: Obvious Conspiracy

```
BREAKING: NASA officials admit that the moon landing was faked in a Hollywood
studio! Inside sources reveal that the government has been lying to the public
for 50 years. Secret documents prove that aliens have been controlling world
governments since 1947. Wake up sheeple! They don't want you to know the truth!
```

**Expected:** 🔴 FAKE  
**Why:** Conspiracy theory, vague sources, sensationalism, no evidence

---

### Test Case 6: False Health Claim

```
New Study PROVES that drinking coffee cures cancer completely! Doctors HATE
this simple one-trick cure! Scientists discovered that a chemical in coffee
destroys 100% of all cancer cells. The pharmaceutical industry is trying to
suppress this information because they want your money instead. Start drinking
coffee today and you'll be guaranteed to live to 150!
```

**Expected:** 🔴 FAKE  
**Why:** Extreme claims, "doctors hate," vague "study," all-caps sensationalism

---

### Test Case 7: Misleading Celebrity News

```
Celebrity A is CONFIRMED DEAD! Multiple sources say they died last night in
a mysterious accident! The government is covering it up! Their family won't
release a statement which PROVES something is wrong! Don't believe the
mainstream media denials - anonymous insiders are saying this is real!
```

**Expected:** 🔴 FAKE  
**Why:** Vague claims, "anonymous," inflammatory, demands you distrust "mainstream"

---

### Test Case 8: False Political Claim

```
EXCLUSIVE: A secret email proves that the entire election was rigged!
Nobody is talking about this! Anonymous hackers found documents showing
the government changed millions of votes. The media won't report this because
they're all in on it! SHARE THIS BEFORE IT GETS DELETED!
```

**Expected:** 🔴 FAKE  
**Why:** No verifiable evidence, vague "anonymous," urgency manipulation, conspiracy thinking

---

## 🤔 BORDERLINE / TRICKY CASES

### Test Case 9: Opinion/Satire

```
The new smartphone is absolutely revolutionary and will change humanity forever.
While competitors are stuck in the past with their outdated designs, this device
represents the pinnacle of technological achievement. Any rational person would
agree that this is objectively the best phone ever created, and anyone who thinks
otherwise is simply wrong.
```

**Expected:** Mixed (Could be opinion or exaggeration)  
**Why:** Heavy opinion language, superlatives, lacks objective evidence

---

### Test Case 10: Partially True with Misleading Headline

```
Study Links Social Media to Teen Anxiety - But Full Picture is More Complex.
Researchers found correlation between social media use and reported anxiety
in teens, though causation remains unclear. The study involved 5,000 participants
and lasted two years. However, experts caution that many factors contribute to
teen mental health, and more research is needed before drawing firm conclusions.
```

**Expected:** 🟢 REAL  
**Why:** Acknowledges nuance, cites methodology, experts quoted appropriately

---

### Test Case 11: Short Fake Headline

```
Celebrity spotted driving UFO in downtown Los Angeles!
```

**Expected:** 🔴 FAKE  
**Why:** Absurd claim, no supporting evidence, classic tabloid sensationalism

---

### Test Case 12: Short Real News

```
Apple releases iPhone 15 with improved camera and battery life.
```

**Expected:** 🟢 REAL  
**Why:** Simple, factual, credible product announcement

---

## 📊 TESTING STRATEGY

1. **Start with obvious cases** (Test 5-8, 11-12) - Models should agree 100%
2. **Try real news** (Test 1-4, 10, 12) - See if models recognize quality reporting
3. **Try edge cases** (Test 9) - See how models handle opinion vs fake news
4. **Try your own text** - Paste actual news articles you find online

---

## 🎯 What to Look For

- ✅ **Do all 4 models agree?** = Strong signal
- ⚠️ **Do they disagree?** = Text is ambiguous or borderline
- 📊 **How confident are they?** = Check the percentage scores
- 🔍 **Which model is most confident?** = Usually Full BERT

---

## 💡 Tips for Testing

- **Copy real news** from Reuters, AP, BBC, NPR, etc. to verify models work
- **Create fake news** by mixing true facts with false conclusions
- **Test satire** - Models might struggle with obvious satire/parody sites
- **Test opinion pieces** - Might be flagged as "fake" due to strong language
- **Test short vs long** - See how model length affects predictions
- **Save results** - Take screenshots of interesting patterns

---

## 🚀 Run the App & Test

```bash
streamlit run app_multi_model.py
```

Then copy/paste any test case above and see how all 4 models respond!
