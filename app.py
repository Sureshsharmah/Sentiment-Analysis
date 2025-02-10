from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Load trained sentiment model and vectorizer
model_path = r"D:\Projects\NLP Project\Sentiment Analysis\Models\model_xgb.pkl"
vectorizer_path = r"D:\Projects\NLP Project\Sentiment Analysis\Models\countVectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file not found. Please check the paths.")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)
        sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"

        return jsonify({"sentiment": sentiment})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
