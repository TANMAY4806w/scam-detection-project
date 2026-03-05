# 🧪 Testing Documentation — ScamGuard AI

> Comprehensive testing plan and results for the AI-Based Fake Investment Scam & Ponzi Scheme Detection System.

---

## 1. Test Strategy Overview

| Test Category | Scope | Method |
|---------------|-------|--------|
| **ML Functional Testing** | Model predictions on known inputs | `07_custom_evaluation.py` — 20 test cases |
| **API Integration Testing** | Backend `/predict` endpoint | `test_api.py` — 8 HTTP test cases |
| **Frontend Functional Testing** | UI behavior and user interaction | Manual browser testing |
| **Edge Case Testing** | Boundary inputs and error handling | Documented below |
| **Explainability Testing** | LIME word-level indicators | `06_real_life_test.py` |

---

## 2. ML Model Functional Tests

### 2.1 Test Dataset (20 Custom Real-Life Messages)

These test cases use hand-crafted messages that simulate real social media content. They are evaluated against all 4 trained TF-IDF models in `07_custom_evaluation.py`.

#### Scam Test Cases (12 messages)

| # | Input Message | Expected | Notes |
|---|---------------|----------|-------|
| 1 | "GUARANTEED returns of 100% daily in crypto! Click here to join my VIP mining pool now limits apply!!" | Scam | Classic guaranteed-return scam |
| 2 | "You have been selected to win 5 BTC. Please send 0.1 BTC to verify your identity and claim." | Scam | Advance-fee crypto scam |
| 3 | "Hey bro, I found this amazing new trading bot that consistently doubles your money every week. DM for link" | Scam | Trading bot promotion |
| 4 | "URGENT: Your Binance account is blocked. Verify your funds at www.binance-secure-auth.com immediately to unlock." | Scam | Phishing scam |
| 5 | "Invest just $50 today and I will manage your forex trades. Profits guaranteed $5000 in 3 days. Risk free." | Scam | Forex scam with guaranteed profits |
| 6 | "Elon Musk is giving back to the community! Send ETH to the address below and get 2x back instantly!" | Scam | Celebrity impersonation crypto scam |
| 7 | "This new altcoin is about to pump 1000x tomorrow! Join our pump and dump signal group right now." | Scam | Pump-and-dump scheme |
| 8 | "Hello sir, I am an expert account manager. Just fund your mt4 account and give me the password." | Scam | Account manager fraud |
| 9 | "Double your ethereum overnight with our new staking DApp protocol. Secure, audited, and mathematically guaranteed." | Scam | Fake DeFi staking scam |
| 10 | "Hi darling, I made $50k last month just trading binary options from my phone. I can teach you for a small signup fee." | Scam | Binary options social scam |
| 11 | "Congratulations! Your number won the WhatsApp Financial Lottery of $1,000,000. Pay the $500 transfer fee." | Scam | Lottery/advance-fee fraud |
| 12 | "Secret Ponzi matrix system just launched! Getting in early means you get paid from everyone who joins later." | Scam | Explicit Ponzi scheme |

#### Legitimate Test Cases (8 messages)

| # | Input Message | Expected | Notes |
|---|---------------|----------|-------|
| 13 | "Federal Reserve announces a new interest rate hike of 0.25% in an effort to curb inflation." | Legitimate | Central bank news |
| 14 | "I just read the Wall Street Journal article about the upcoming Apple earnings report." | Legitimate | Financial media discussion |
| 15 | "According to my technical analysis on the daily chart, BTC might test support at the 200 EMA." | Legitimate | Crypto technical analysis |
| 16 | "Just dollar cost averaging into my Vanguard S&P 500 index fund this month like usual." | Legitimate | Personal investing strategy |
| 17 | "Can anyone recommend a good book on value investing? I've already read The Intelligent Investor." | Legitimate | Investment education |
| 18 | "Goldman Sachs reports that oil prices may stabilize next quarter." | Legitimate | Institutional financial report |
| 19 | "I need to rebalance my 401k portfolio since US equities have grown to be 85% of my allocation." | Legitimate | Portfolio management discussion |
| 20 | "The European Central Bank plans to inject liquidity into the banking sector to prevent a recession." | Legitimate | Economic policy news |

---

## 3. API Integration Tests

### 3.1 Test Script (`test_api.py`)

The API is tested via direct HTTP `POST` requests to `http://127.0.0.1:8000/predict`.

| # | True Label | Input Message (truncated) | Expected Prediction |
|---|------------|---------------------------|---------------------|
| 1 | SCAM | "Guaranteed 200 percent profit daily! Send Bitcoin now..." | Scam |
| 2 | LEGIT | "The Federal Reserve held interest rates steady at 5.25%..." | Legitimate |
| 3 | SCAM | "Hey bro join our VIP pump and dump signal group..." | Scam |
| 4 | SCAM | "You have been selected to receive 5 BTC. Send 0.1 BTC..." | Scam |
| 5 | LEGIT | "Apple Inc. reported strong quarterly earnings beating..." | Legitimate |
| 6 | SCAM | "Elon Musk crypto giveaway! Send ETH to this address..." | Scam |
| 7 | LEGIT | "I am dollar cost averaging into index funds this month..." | Legitimate |
| 8 | SCAM | "Earn passive income forever. Our Ponzi matrix pays you..." | Scam |

### 3.2 Expected API Response Format

```json
{
  "prediction": "Scam | Legitimate",
  "probability": 0.0 - 1.0,
  "risk_level": "High | Medium | Low"
}
```

### 3.3 API Error Handling Tests

| Test Case | Input | Expected Status | Expected Behavior |
|-----------|-------|-----------------|-------------------|
| Empty text | `{"text": ""}` | 400 | Returns `"Text cannot be empty."` |
| Empty whitespace | `{"text": "   "}` | 400 | Returns `"Text cannot be empty."` |
| Valid scam text | `{"text": "Send BTC now!"}` | 200 | Returns prediction JSON |
| Missing body | `{}` | 422 | Pydantic validation error |
| Invalid JSON | malformed | 422 | FastAPI parse error |

---

## 4. Frontend Functional Tests

| # | Test Case | Steps | Expected Result |
|---|-----------|-------|-----------------|
| 1 | Page Load | Open `http://127.0.0.1:5500` | Header, hero, card, stats render correctly |
| 2 | Sample Load (Scam) | Click "💸 Crypto Scam" button | Textarea fills with scam sample text |
| 3 | Sample Load (Legit) | Click "📈 Legit Finance" button | Textarea fills with legitimate sample text |
| 4 | Character Counter | Type in textarea | Character count updates in real-time |
| 5 | Empty Submit | Click "Analyze Now" with empty input | Red border flash + warning placeholder |
| 6 | Valid Analysis | Enter text → Click "Analyze Now" | Loading spinner → Result card with prediction |
| 7 | Scam Result Display | Submit a scam message | Red banner, "🚨 SCAM DETECTED", High risk badge |
| 8 | Legit Result Display | Submit a legitimate message | Green banner, "✅ LEGITIMATE MESSAGE", Low risk badge |
| 9 | Probability Bar | After prediction | Bar animates to the scam probability % |
| 10 | Clear Button | Click "Clear" | Textarea empties, char count resets, result hides |
| 11 | Analyze Another | Click "Analyze Another" in result | Returns to empty input state |
| 12 | Keyboard Shortcut | Press Ctrl+Enter | Triggers analysis (same as clicking button) |
| 13 | Server Down | Submit with backend offline | Error message with restart instructions shown |
| 14 | Responsive Layout | Resize to ≤600px | Stats grid → 2 columns, card header stacks |

---

## 5. Edge Case Tests

| # | Scenario | Input | Expected Behavior |
|---|----------|-------|-------------------|
| 1 | Very short text | "hi" | Returns a prediction (likely Legitimate, Low risk) |
| 2 | Very long text | 5000+ character message | Truncated by FinBERT at 128 tokens; TF-IDF handles full text |
| 3 | Non-English text | "Invertir $100 y ganar $1000" | May misclassify—models trained on English only |
| 4 | Numbers only | "123456789" | Returns prediction (likely Legitimate) |
| 5 | URL-heavy text | "visit http://scam.com http://fake.xyz" | URLs stripped during preprocessing; classified on remaining text |
| 6 | Special characters | "!!!@@@###$$$%%%" | Cleaned to empty → should handle gracefully |
| 7 | Mixed legitimate + scam | "Apple earnings were good. Also send BTC to double money!" | Ensemble averages conflicting signals |
| 8 | All caps input | "GUARANTEED RETURNS JOIN NOW" | Lowercased during preprocessing; model handles normally |

---

## 6. Explainability Tests (LIME)

Tested in `06_real_life_test.py`. LIME provides word-level contributions to the scam prediction.

### Test Case 1: Scam Input
**Input:** "Hey man! I just found a guaranteed way to double your money in 24 hours using smart crypto mining returns."

**Expected LIME Output:** Words like `guaranteed`, `double`, `money`, `crypto`, `mining` should have **positive weights** (contributing to Scam prediction).

### Test Case 2: Legitimate Input
**Input:** "Apple Inc. announced its quarterly earnings today. Revenue increased by 5%, beating expectations."

**Expected LIME Output:** Words like `earnings`, `revenue`, `quarterly`, `apple` should have **negative weights** (pushing toward Legitimate prediction).

---

## 7. Model Performance Verification

### 7.1 Confusion Matrix (Logistic Regression on FinBERT, 20% Test Split)

```
                 Predicted
                 Legit   Scam
Actual Legit  [ TN   |  FP  ]
Actual Scam   [ FN   |  TP  ]
```

- **True Positives (TP):** Scams correctly caught
- **False Negatives (FN):** Scams missed — **critical metric to minimize**
- **True Negatives (TN):** Legitimate messages correctly passed
- **False Positives (FP):** Legitimate messages incorrectly flagged

### 7.2 Cross-Model Comparison on Custom Benchmark (20 examples)

| Model | Accuracy | Scams Caught |
|-------|----------|--------------|
| Logistic Regression | Evaluated | X/12 |
| Naive Bayes | Evaluated | X/12 |
| XGBoost | Evaluated | X/12 |
| Voting Classifier | Evaluated | X/12 |

> Results from running `python 07_custom_evaluation.py` — see `output.txt` for full output.

---

## 8. Test Execution Instructions

### Run ML Pipeline Tests
```bash
cd notebooks
python 07_custom_evaluation.py   # 20-example benchmark
python 06_real_life_test.py      # LIME explainability test
```

### Run API Integration Tests
```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: Run tests
cd backend
python test_api.py
```

### Frontend Manual Testing
```bash
cd frontend
python -m http.server 5500
# Open http://127.0.0.1:5500 in browser
```
