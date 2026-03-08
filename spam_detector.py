# ============================================================
# PROJECT 1: Spam Email Detector
# Skills: Python, NLP, Scikit-learn, Naive Bayes
# Dataset: UCI SMS Spam Collection (loaded via sklearn)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── 1. Sample Dataset (replace with full CSV if available) ──
data = {
    'label': [
        'spam','spam','spam','spam','spam','spam','spam','spam','spam','spam',
        'ham','ham','ham','ham','ham','ham','ham','ham','ham','ham',
        'spam','spam','spam','ham','ham','ham','ham','spam','ham','ham'
    ],
    'message': [
        'Free entry in 2 a weekly competition to win FA Cup final tickets!',
        'You have won a $1000 Walmart gift card. Click here to claim now!',
        'URGENT! You have won a 1 week FREE membership in our prize draw!',
        'Congratulations! You have been selected for a cash prize of $500.',
        'Get cheap meds online! No prescription needed. Click here.',
        'You are a winner! Call 09061702893 to claim your prize NOW!',
        'SMS: Your free ringtone is waiting to be collected!',
        'Win a brand new car! Text WIN to 87239 now!',
        'PRIZE NOTIFICATION: You have won 2000 pounds. Reply YES to claim.',
        'Your mobile number has been awarded a $2000 prize. Call now!',
        'Hey, are you coming to the party tonight?',
        'Can you pick up some milk on your way home?',
        'The meeting has been rescheduled to 3pm tomorrow.',
        'Happy birthday! Hope you have a wonderful day.',
        'I will be late by 10 minutes, stuck in traffic.',
        'Did you watch the game last night? It was amazing!',
        'Thanks for the help yesterday, really appreciated it.',
        'What time does the movie start? I forgot.',
        'Lunch was great today, we should do it again sometime.',
        'Let me know when you are free to catch up.',
        'Claim your free iPhone now! Limited time offer!',
        'You have been pre-selected for a low interest credit card!',
        'Buy 1 get 1 FREE! Text OFFER to 12345 to redeem now.',
        'Hey, just checking in. How are you doing?',
        'The report is ready. I have sent it to your email.',
        'See you at the gym tomorrow morning at 7am.',
        'Can we reschedule our call to Friday instead?',
        'WINNER! As a valued customer you have been selected for a prize.',
        'I am on my way, be there in 5 minutes.',
        'Good morning! Have a great day ahead.',
    ]
}

df = pd.DataFrame(data)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("=" * 55)
print("       PROJECT 1: SPAM EMAIL DETECTOR")
print("=" * 55)
print(f"\n📊 Dataset Overview:")
print(f"   Total samples : {len(df)}")
print(f"   Ham (legit)   : {(df['label']=='ham').sum()}")
print(f"   Spam          : {(df['label']=='spam').sum()}")

# ── 2. Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# ── 3. Build Pipeline (Vectorizer + TF-IDF + Naive Bayes) ──
pipeline = Pipeline([
    ('bow',   CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf',   MultinomialNB()),
])

pipeline.fit(X_train, y_train)

# ── 4. Evaluate ──
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Trained Successfully!")
print(f"   Accuracy : {acc * 100:.1f}%")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham','Spam']))

# ── 5. Predict on new emails ──
print("🔍 Predictions on New Emails:")
test_emails = [
    "You have won a FREE holiday! Claim now by clicking this link.",
    "Hey! Are we still on for dinner tonight?",
    "Congratulations, your account has been selected for a cash reward!",
    "Can you send me the notes from today's lecture please?",
]

for email in test_emails:
    pred = pipeline.predict([email])[0]
    label = "🚨 SPAM" if pred == 1 else "✅ HAM"
    print(f"   {label} → \"{email[:55]}...\"" if len(email) > 55 else f"   {label} → \"{email}\"")
