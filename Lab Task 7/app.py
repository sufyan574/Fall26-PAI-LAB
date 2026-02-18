from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

THEMEALDB_BASE = "https://www.themealdb.com/api/json/v1/1"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/random")
def get_random():
    try:
        r = requests.get(f"{THEMEALDB_BASE}/random.php", timeout=8)
        r.raise_for_status()
        return jsonify(r.json())
    except:
        return jsonify({"error": "Cannot reach recipe service"}), 503

@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Missing search query (?q=...)"}), 400
    
    try:
        r = requests.get(f"{THEMEALDB_BASE}/search.php", params={"s": q}, timeout=8)
        r.raise_for_status()
        return jsonify(r.json())
    except:
        return jsonify({"error": "Recipe service unavailable"}), 503

if __name__ == "__main__":
    print("Starting Recipe App → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)