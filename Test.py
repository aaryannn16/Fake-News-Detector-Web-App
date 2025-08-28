import argparse
import sys
from joblib import load
from pathlib import Path

VEC_PATH = Path("C:/Users/Aryan/Documents/Programs/Fake News Detector/model/vectorizer.joblib")
MODEL_PATH = Path("C:/Users/Aryan/Documents/Programs/Fake News Detector/model/fake_news_model.joblib")

def load_artifacts():
    if not VEC_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model artifacts. Expected:\n  {VEC_PATH}\n  {MODEL_PATH}\n"
            "Train and save the model first."
        )
    vectorizer = load(VEC_PATH)
    model = load(MODEL_PATH)
    return vectorizer, model

def predict_text(text: str, vectorizer, model):
    X = vectorizer.transform([text])
    proba_fake = float(model.predict_proba(X)[0][1])  # class 1 = FAKE
    label = "FAKE" if proba_fake >= 0.5 else "REAL"
    return {"label": label, "score": proba_fake}

def main():
    parser = argparse.ArgumentParser(description="Fake news classifier (inference)")
    parser.add_argument("--text", type=str, help="Text to classify (headline/article)")
    parser.add_argument("--file", type=str, help="Path to a .txt file with the text to classify")
    args = parser.parse_args()

    vectorizer, model = load_artifacts()

    if args.text:
        text = args.text.strip()
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Failed to read file: {e}")
            sys.exit(1)
    else:
        print("Enter/paste the news text, then press Enter:")
        try:
            text = sys.stdin.readline().strip()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(1)

    if not text:
        print("No text provided.")
        sys.exit(1)

    result = predict_text(text, vectorizer, model)
    print(f"Label: {result['label']}")
    print(f"Score (probability of FAKE): {result['score']:.4f}")

if __name__ == "__main__":
    main()
