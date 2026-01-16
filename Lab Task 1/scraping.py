from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import re
import csv
import os

# 🔴 CHANGE 1: Base directory define
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔴 CHANGE 2: CSV same folder me force
CSV_FILE = os.path.join(BASE_DIR, "emails.csv")

app = Flask(__name__)

# CSV file create agar pehle se na ho
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Email"])

@app.route("/", methods=["GET", "POST"])
def index():
    emails = []
    url = ""

    if request.method == "POST":
        url = request.form.get("url")

        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()

            # Email regex
            emails = set(re.findall(
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                text
            ))

            # Save to CSV (same folder)
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for email in emails:
                    writer.writerow([url, email])

        except Exception as e:
            emails = [f"Error: {str(e)}"]

    return render_template("index.html", emails=emails, url=url)

if __name__ == "__main__":
    app.run(debug=True)
