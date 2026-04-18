import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from dotenv import load_dotenv
import requests
from fpdf import FPDF

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

print("=== GROQ DEBUG START ===")
print(f"API Key Loaded: {'YES' if GROQ_API_KEY else 'NO'}")
if GROQ_API_KEY:
    print(f"Key starts with: {GROQ_API_KEY[:30]}...")
print("=== GROQ DEBUG END ===\n")

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY missing in .env file!")

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS history')
    cursor.execute('''CREATE TABLE history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT, email TEXT, phone TEXT, linkedin TEXT, location TEXT,
                        education TEXT, experience TEXT, skills TEXT,
                        target_job_title TEXT, job_description TEXT, company_name TEXT,
                        resume TEXT, cover_letter TEXT, template_type TEXT,
                        created_at TIMESTAMP
                    )''')
    conn.commit()
    conn.close()

init_db()

# Real Groq API Call (No Silent Fail)
def generate_resume_and_cover_letter(data, template_type):
    if not GROQ_API_KEY:
        print("ERROR: No Groq API key found")
        return None, None

    try:
        print("DEBUG: Groq API call STARTED")

        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }

        prompt = f"""Create a professional resume and cover letter.

Name: {data.get('name', 'Candidate')}
Email: {data.get('email', '')}
Phone: {data.get('phone', '')}
LinkedIn: {data.get('linkedin', '')}
Location: {data.get('location', '')}
Education: {data.get('education', '')}
Experience: {data.get('experience', '')}
Skills: {data.get('skills', '')}
Target Job: {data.get('target_job_title', '')}
Company: {data.get('company_name', 'the company')}

Template: {template_type}

Return exactly:

RESUME:
[resume here]

COVER LETTER:
[cover letter here]
"""

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.7
        }

        print("DEBUG: Sending request to https://api.groq.com...")

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=50
        )

        print(f"DEBUG: Response Status Code: {response.status_code}")

        response.raise_for_status()
        result = response.json()

        content = result['choices'][0]['message']['content']

        print("DEBUG: Groq API call SUCCESSFUL")

        if "COVER LETTER:" in content.upper():
            parts = content.split("COVER LETTER:", 1)
            resume = parts[0].replace("RESUME:", "").strip()
            cover_letter = parts[1].strip()
        else:
            resume = content
            cover_letter = "Cover letter generated."

        return resume, cover_letter

    except Exception as e:
        print(f"Groq API Error: {type(e).__name__} - {e}")
        flash(f"Groq API Error: {str(e)[:100]}", "error")
        return None, None

# PDF Generation
def generate_pdf_from_markdown(text, filename, title):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(15)
    pdf.set_font('Helvetica', '', 11)
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            pdf.ln(6)
            continue
        line = line.replace('•', '-').replace('✦', '*')
        pdf.multi_cell(0, 8, line)
        pdf.ln(3)
    
    pdf.output(filename)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        data = request.form.to_dict()
        template_type = data.get('template_type', 'professional')

        resume_md, cover_letter_md = generate_resume_and_cover_letter(data, template_type)

        if not resume_md or not cover_letter_md:
            flash('Failed to generate resume using AI. Please try again.', 'error')
            return redirect(url_for('generate'))

        os.makedirs('static/resumes', exist_ok=True)
        os.makedirs('static/cover_letters', exist_ok=True)

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO history 
            (name, email, phone, linkedin, location, education, experience, skills, 
             target_job_title, job_description, company_name, resume, cover_letter, template_type, created_at) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (data.get('name'), data.get('email'), data.get('phone'), data.get('linkedin'),
             data.get('location'), data.get('education'), data.get('experience'), 
             data.get('skills'), data.get('target_job_title'), data.get('job_description'),
             data.get('company_name'), resume_md, cover_letter_md, template_type, datetime.now()))

        gen_id = cursor.lastrowid
        conn.commit()
        conn.close()

        resume_pdf_path = f"static/resumes/resume_{gen_id}.pdf"
        cover_letter_pdf_path = f"static/cover_letters/cover_letter_{gen_id}.pdf"

        generate_pdf_from_markdown(resume_md, resume_pdf_path, "Resume")
        generate_pdf_from_markdown(cover_letter_md, cover_letter_pdf_path, "Cover Letter")

        return render_template('success.html',
                               resume=resume_md,
                               cover_letter=cover_letter_md,
                               resume_pdf=resume_pdf_path,
                               cover_letter_pdf=cover_letter_pdf_path)

    return render_template('generate.html')

# History, Delete, Download
@app.route('/history')
def history():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, created_at, target_job_title, company_name, template_type FROM history ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()

    history_data = [
        {
            'id': row[0],
            'created_at': row[1],
            'target_job_title': row[2],
            'company_name': row[3],
            'template_type': row[4] or 'professional'
        }
        for row in rows
    ]

    return render_template('history.html', history=history_data)

@app.route('/history/delete/<int:gen_id>')
def delete_history(gen_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history WHERE id = ?', (gen_id,))
    conn.commit()
    conn.close()
    flash('Record deleted successfully!', 'success')
    return redirect(url_for('history'))

@app.route('/download/<int:gen_id>/<string:file_type>')
def download(gen_id, file_type):
    if file_type == 'resume':
        filename = f"resumes/resume_{gen_id}.pdf"
    elif file_type == 'cover_letter':
        filename = f"cover_letters/cover_letter_{gen_id}.pdf"
    else:
        return "Invalid file type", 404
    return send_from_directory('static', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)