# api/app.py
from flask import Flask, request, jsonify
from joblib import load
from pathlib import Path
from flask_cors import CORS  # pip install flask-cors

app = Flask(__name__)
CORS(app)  # enable CORS for dev; restrict origins in prod

# Use either relative paths under the project (recommended) OR direct absolute paths, not both at once.
# Option A: relative to project root (assuming this file is at <project>/api/app.py and model/ is at <project>/model)
#VEC_PATH = Path(__file__).parent.parent / "model" / "vectorizer.joblib"
#MODEL_PATH = Path(__file__).parent.parent / "model" / "fake_news_model.joblib"

# Option B (if you truly want absolute paths, uncomment and use only these):
VEC_PATH = Path(r"C:\Users\Aryan\Documents\Programs\Fake News Detector\model\vectorizer.joblib")
MODEL_PATH = Path(r"C:\Users\Aryan\Documents\Programs\Fake News Detector\model\fake_news_model.joblib")

vectorizer = load(VEC_PATH)
model = load(MODEL_PATH)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    X = vectorizer.transform([text])
    probas = model.predict_proba(X)  # shape: (1, n_classes)

    # Safe selection of the "fake" class probability:
    # If labels are {0,1}, column for class 1 is the "fake" probability.
    # Otherwise, find the column index by inspecting model.classes_.
    if getattr(model, "classes_", None) is not None:
        import numpy as np
        idx_candidates = np.where(model.classes_ == 1)[0]
        if len(idx_candidates) == 0:
            # Fallback: if class 1 doesn't exist, assume binary and take the second column
            idx_fake = 1
        else:
            idx_fake = int(idx_candidates[0])
    else:
        idx_fake = 1  # typical for binary classifiers with classes_ == [0, 1]

    proba_fake = float(probas[0, idx_fake])
    label = "FAKE" if proba_fake >= 0.5 else "REAL"
    return jsonify({"label": label, "score": proba_fake})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
