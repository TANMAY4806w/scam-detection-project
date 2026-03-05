const API_URL = 'http://127.0.0.1:8000/predict';

const SAMPLES = {
  scam1: "Guaranteed 200% daily returns on your crypto investment! Join our exclusive mining pool now. Limited spots available — send BTC to double your money in 24 hours!",
  scam2: "Hey! Join my Ponzi matrix system. Everyone who joins after you pays you first. Earn passive income forever just by investing $50 and recruiting 3 friends.",
  legit1: "Just DCA-ing into my Vanguard S&P 500 index fund this month like usual. Long-term investing requires patience and discipline.",
  legit2: "Apple Inc. reported strong quarterly earnings today, beating analyst estimates. Revenue grew by 7% year over year. The stock market responded positively.",
};

// Character counter
const textarea = document.getElementById('messageInput');
const charCount = document.getElementById('charCount');
textarea.addEventListener('input', () => {
  const len = textarea.value.length;
  charCount.textContent = `${len} character${len !== 1 ? 's' : ''}`;
});

function loadSample(key) {
  textarea.value = SAMPLES[key];
  const ev = new Event('input');
  textarea.dispatchEvent(ev);
  textarea.focus();
}

function clearInput() {
  textarea.value = '';
  charCount.textContent = '0 characters';
  hideResult();
  textarea.focus();
}

function hideResult() {
  document.getElementById('resultCard').classList.add('hidden');
}

function reset() {
  hideResult();
  clearInput();
}

async function analyzeText() {
  const text = textarea.value.trim();
  if (!text) {
    textarea.style.borderColor = 'rgba(255,75,110,0.6)';
    textarea.placeholder = '⚠️  Please enter a message to analyze first.';
    setTimeout(() => {
      textarea.style.borderColor = '';
      textarea.placeholder = "Paste a social media message, investment offer, or Telegram/WhatsApp message here...";
    }, 2500);
    return;
  }

  // Show card in loading state
  const resultCard   = document.getElementById('resultCard');
  const loadingState = document.getElementById('loadingState');
  const resultState  = document.getElementById('resultState');

  resultCard.classList.remove('hidden');
  loadingState.classList.remove('hidden');
  resultState.classList.add('hidden');

  // Disable button
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.querySelector('.btn-text').textContent = 'Analyzing…';

  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Server error');
    }

    const data = await response.json();
    displayResult(data);

  } catch (err) {
    displayError(err.message);
  } finally {
    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'Analyze Now';
  }
}

function displayResult(data) {
  const loadingState  = document.getElementById('loadingState');
  const resultState   = document.getElementById('resultState');
  const resultBanner  = document.getElementById('resultBanner');
  const resultIcon    = document.getElementById('resultIcon');
  const resultLabel   = document.getElementById('resultLabel');
  const resultSubtitle = document.getElementById('resultSubtitle');
  const riskBadge     = document.getElementById('riskBadge');
  const probBar       = document.getElementById('probBar');
  const probValue     = document.getElementById('probValue');
  const confValue     = document.getElementById('confValue');
  const riskValue     = document.getElementById('riskValue');

  const isScam   = data.prediction === 'Scam';
  const prob     = data.probability;           // 0–1
  const probPct  = Math.round(prob * 100);
  const risk     = data.risk_level;

  // Banner
  resultBanner.className = `result-banner ${isScam ? 'scam-banner' : 'legit-banner'}`;
  resultIcon.textContent  = isScam ? '⚠️' : '✅';
  resultLabel.textContent = isScam ? '🚨 SCAM DETECTED' : '✅ LEGITIMATE MESSAGE';
  resultLabel.style.color = isScam ? '#ff4b6e' : '#00d97e';
  resultSubtitle.textContent = isScam
    ? 'This message contains patterns commonly associated with financial scams or Ponzi schemes.'
    : 'This message appears to be a legitimate financial discussion. Always verify independently.';

  // Risk badge
  riskBadge.textContent  = `${risk} Risk`;
  riskBadge.className    = `risk-badge risk-${risk.toLowerCase()}`;

  // Probability bar – animate after tiny delay
  probBar.style.width = '0%';
  probValue.textContent = `${probPct}%`;
  setTimeout(() => { probBar.style.width = `${probPct}%`; }, 80);

  // Indicators
  const conf = prob >= 0.8 ? 'Very High' : prob >= 0.6 ? 'High' : prob >= 0.4 ? 'Medium' : 'Low';
  confValue.textContent = conf;
  riskValue.textContent = risk;
  riskValue.style.color = risk === 'High' ? '#ff4b6e' : risk === 'Medium' ? '#ff8c42' : '#00d97e';

  loadingState.classList.add('hidden');
  resultState.classList.remove('hidden');
}

function displayError(message) {
  const loadingState = document.getElementById('loadingState');
  const resultState  = document.getElementById('resultState');

  loadingState.innerHTML = `
    <div style="text-align:center; color:#ff4b6e; padding: 12px 0;">
      <div style="font-size:1.8rem; margin-bottom:10px;">⚡</div>
      <p style="font-weight:600;">Connection Error</p>
      <p style="font-size:0.85rem; color:#7c8497; margin-top:6px;">
        Could not reach the backend. Make sure the FastAPI server is running at port 8000.<br>
        <code style="font-size:0.8rem; opacity:0.7">uvicorn main:app --reload --port 8000</code>
      </p>
      <button class="btn btn-secondary" style="margin-top:14px;" onclick="reset()">Try Again</button>
    </div>`;
}

// Allow Ctrl+Enter to submit
textarea.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyzeText();
});
