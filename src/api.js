export async function classify(text) {
    const res = await fetch("http://192.168.1.12:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return await res.json(); // { label, score }
  }
  