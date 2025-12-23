# =====================================================
# Support Ticket Classification
# ML Training + Model Persistence + REST API
# =====================================================

import pandas as pd
import re
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from flask import Flask, request, jsonify


# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------
# Dataset columns:
# Document -> Subject + Description (combined)
# Topic_group -> Ticket Category (target)

DATA_PATH = "IT_Service_Ticket_Classification.csv"

df = pd.read_csv(DATA_PATH)


# -----------------------------------------------------
# 2. Text Preprocessing
# -----------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["Document"].astype(str).apply(clean_text)

X = df["text"]
y = df["Topic_group"]


# -----------------------------------------------------
# 3. Train-Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------------------------
# 4. Build ML Pipeline
# -----------------------------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])


# -----------------------------------------------------
# 5. Train Model
# -----------------------------------------------------
model.fit(X_train, y_train)


# -----------------------------------------------------
# 6. Evaluate Model
# -----------------------------------------------------
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

latency = (end_time - start_time) / len(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro"
)

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy)
print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)
print("Average inference latency (seconds):", latency)

print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred))


# -----------------------------------------------------
# 7. Save Trained Model (Persistence)
# -----------------------------------------------------
MODEL_PATH = "ticket_classifier.pkl"
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")


# =====================================================
# 8. REST API (Flask)
# =====================================================
app = Flask(__name__)

# Load model once at startup
loaded_model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def health_check():
    return "Ticket Classification API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    subject = data.get("subject", "")
    description = data.get("description", "")

    if not subject and not description:
        return jsonify({"error": "Subject or description required"}), 400

    text = clean_text(subject + " " + description)

    start = time.time()
    prediction = loaded_model.predict([text])[0]
    inference_latency = time.time() - start

    return jsonify({
        "predicted_category": prediction,
        "latency_seconds": inference_latency
    })


if __name__ == "__main__":
    app.run(debug=True)
