import requests
import time

time.sleep(3)  # wait for uvicorn to reload the changes

base_url = 'http://127.0.0.1:8000'

tests = [
    ('SCAM',  'Guaranteed 200 percent profit daily! Send Bitcoin now and double your crypto in 24 hours. Limited spots!'),
    ('LEGIT', 'The Federal Reserve held interest rates steady at 5.25 percent today. Stock markets responded positively.'),
    ('SCAM',  'Hey bro join our VIP pump and dump signal group. We moon 1000x altcoins every week. DM me for invite.'),
    ('SCAM',  'You have been selected to receive 5 BTC. Send 0.1 BTC to verify your identity and claim your prize now.'),
    ('LEGIT', 'Apple Inc. reported strong quarterly earnings beating analyst estimates. Revenue grew 7 percent year over year.'),
    ('SCAM',  'Elon Musk crypto giveaway! Send ETH to this address and receive 2x back instantly. Limited promotion.'),
    ('LEGIT', 'I am dollar cost averaging into index funds this month. Long term investing requires patience and discipline.'),
    ('SCAM',  'Earn passive income forever. Our Ponzi matrix pays you from every new member who joins after you.'),
]

print('=' * 70)
print(f"{'True':<6} | {'Result':<12} | {'Prob%':<8} | {'Risk':<7} | Message (first 40 chars)")
print('=' * 70)

correct = 0
for true_label, msg in tests:
    r = requests.post(f'{base_url}/predict', json={'text': msg})
    res = r.json()
    pred = res['prediction']
    prob = f"{res['probability']*100:.1f}%"
    risk = res['risk_level']
    is_correct = (true_label == 'SCAM' and pred == 'Scam') or (true_label == 'LEGIT' and pred == 'Legitimate')
    match_icon = '[OK]' if is_correct else '[X]'
    if is_correct:
        correct += 1
    print(f"{true_label:<6} | {pred:<12} | {prob:>7} | {risk:<7} | {msg[:40]}...")

print('=' * 70)
print(f'Accuracy: {correct}/8 = {correct/8*100:.0f}%')
