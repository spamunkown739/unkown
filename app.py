from flask import Flask, request, render_template
import pickle
import re
import string
import nltk


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')  # <-- Yeh line add karein
app = Flask(__name__)


# Load trained model and vectorizer
model = pickle.load(open('best_spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

scam_keywords = {  # Existing scam keywords
    "jeeto": 0.15, "prize": 0.20, "mubarak": 0.10, "car": 0.15, "rupay": 0.12, "lakh": 0.10,
    "benazir": 0.18, "bisp": 0.20, "support": 0.08, "reward": 0.15, "number": 0.10, "lottery": 0.25,
    "win": 0.18, "gift": 0.15, "cash": 0.12, "income": 0.10, "offer": 0.20, "register": 0.10,
    "shipment": 0.12, "tracking": 0.15, "package": 0.10, "delivery": 0.10, "pakpost": 0.25,
    "courier": 0.10, "fake": 0.30,"sale": 0.15,"contact": 0.08,

    # Phishing-related keywords
    "phishing": 0.30, "login": 0.25, "account": 0.20, "password": 0.25, "secure": 0.20, "verify": 0.18,
    "bank": 0.15, "payment": 0.20, "urgent": 0.25, "confirm": 0.18, "update": 0.15, "suspicious": 0.25,

    # Fake lottery/scam-related keywords
    "lottery": 0.25, "winners": 0.20, "grand": 0.30, "jackpot": 0.25, "claim": 0.20, "reward": 0.18,
    "claim prize": 0.20, "exclusive": 0.15, "free gift": 0.20, "immediate": 0.18, "cash prize": 0.25,

    # Investment & Ponzi Scheme Keywords
    "investment": 0.30, "opportunity": 0.25, "guaranteed": 0.25, "high return": 0.30, "risk-free": 0.20,
    "share": 0.15, "trade": 0.18, "profits": 0.25, "stock": 0.18, "bonus": 0.20,

    # Fake job or offer-related keywords
    "job offer": 0.20, "work from home": 0.15, "apply now": 0.20, "hurry": 0.18, "limited time": 0.15,
    "career": 0.12, "full-time": 0.10, "part-time": 0.12, "training": 0.18, "recruitment": 0.15,

    # Romance scam keywords
    "love": 0.25, "relationship": 0.20, "beautiful": 0.18, "heart": 0.20, "trust": 0.15, "gift": 0.18,
    "send money": 0.30, "wedding": 0.25, "special offer": 0.20, "partner": 0.18,

    # Malware/virus-related scam keywords
    "malware": 0.30, "virus": 0.25, "trojan": 0.20, "spyware": 0.25, "infected": 0.15, "scan": 0.18,
    "security alert": 0.20, "update now": 0.15, "download": 0.18, "fix error": 0.20,
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|bit.ly/\S+", ' [URL] ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]
    return " ".join(tokens)

def predict_spam(text):
    text_cleaned = clean_text(text)
    text_vectorized = vectorizer.transform([text_cleaned]).toarray()
    model_probability = model.predict_proba(text_vectorized)[0][1]

    link_weight = 0.20 if re.search(r"http\S+|www\S+|bit.ly/\S+", text, re.IGNORECASE) else 0.0
    if model_probability > 0.50:
        keyword_weight = sum(scam_keywords[word] for word in scam_keywords if word in text_cleaned)
        final_score = model_probability + keyword_weight + link_weight
    elif link_weight > 0.0:
        final_score = model_probability + link_weight
    else:
        final_score = model_probability

    return "Spam" if final_score > 0.65 else "Not Spam"


@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        sms = request.form['sms']
        result = predict_spam(sms)
    return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
