from pathlib import Path
import re
import joblib

MODEL_FILES = {
    "SVM": "svm_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "AdaBoost": "adaboost_model.pkl",
}
MODEL_DIR = Path(__file__).resolve().parent / "models"

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()

def load_artifacts(name):
    return (joblib.load(MODEL_DIR / MODEL_FILES[name]),
            joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl"))

def classify(text, model, vectorizer):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Please enter text containing English letters.")
    features = vectorizer.transform([cleaned])
    if features.nnz == 0:
        raise ValueError("No recognized vocabulary was found. Try a longer passage.")
    prediction = int(model.predict(features)[0])
    label = "AI Generated" if prediction == 1 else "Human Written"
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        index = list(model.classes_).index(prediction)
        probability = float(probabilities[index])
    return label, probability
