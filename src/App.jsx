import { useState } from "react";
import { classify } from "./api";
import "./App.css"; // custom styles

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const r = await classify(text);
      setResult(r); // { label, score }
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-container">
      <div className="card">
        <h1 className="title">📰 Fake News Detector</h1>

        <form onSubmit={onSubmit} className="form">
          <textarea
            rows={7}
            className="textarea"
            placeholder="Paste a headline or article text…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <button
            disabled={loading || !text.trim()}
            className={`button ${loading || !text.trim() ? "disabled" : ""}`}
          >
            {loading ? "🔎 Classifying…" : "🚀 Classify"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <section className="result">
            <p>
              Label:{" "}
              <span
                className={`label ${
                  result.label.toLowerCase() === "fake"
                    ? "label-fake"
                    : "label-real"
                }`}
              >
                {result.label}
              </span>
            </p>
            <p>
              Score (probability of FAKE):{" "}
              <span className="score">{result.score.toFixed(4)}</span>
            </p>
          </section>
        )}
      </div>
    </main>
  );
}
