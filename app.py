from flask import Flask, render_template, request
import pickle
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

# Download stopwords (run once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Text cleaning function (same as notebook)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["review"]

    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]
    confidence = round(max(probs) * 100, 2)

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        positive=round(probs[1]*100, 2),
        negative=round(probs[0]*100, 2),
        review=text
    )

if __name__ == "__main__":
    app.run(debug=True)
