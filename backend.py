from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)
model = joblib.load("email spam.joblib")
vectorizer = joblib.load("vectorizer.joblib")


@app.route("/", methods=["POST"])
def predict_spam():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        if not data or "emailContent" not in data:
            return jsonify(error="Invalid input. 'emailContent' is required."), 400

        # Extract email content
        email_content = data["emailContent"]

        # Transform input string using the vectorizer
        vec = vectorizer.transform([email_content])

        # Predict
        res = model.predict(vec)[0]
        result = "Spam" if res == 1 else "Not Spam"

        return jsonify(message=f"Result is: {result}")
    
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True)