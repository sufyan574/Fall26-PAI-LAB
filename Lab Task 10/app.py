from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def get_bot_response(message):
    msg = message.lower().strip()
    
    if any(greet in msg for greet in ['hi', 'hello', 'hey', 'assalamualaikum', 'salam']):
        return "Hello! 👋 Welcome to Lahore University of Excellence (LUE) Admissions Chatbot. How can I help you today?"
    
    elif any(word in msg for word in ['requirement', 'eligible', 'criteria', 'qualification', 'admission criteria']):
        return "Undergraduate Admission Requirements at LUE:\n• Minimum 60% marks in HSSC / Intermediate (FSc/FA) or equivalent\n• Valid LUE Entry Test / NTS score\n• For Engineering & CS programs: higher marks in Math & Physics required\n• Age limit: 17–25 years"
    
    elif any(word in msg for word in ['deadline', 'last date', 'closing', 'when', 'schedule']):
        return "Important Deadlines – Fall 2026 Intake:\n• Online Applications Open: April 1, 2026\n• Last Date to Apply: June 30, 2026\n• LUE Entry Test: July 15, 2026\n• Merit List Display: July 25, 2026\nAlways double-check on the official website."
    
    elif any(word in msg for word in ['program', 'course', 'degree', 'offered', 'bs', 'bachelor']):
        return "Popular Programs Offered at LUE:\n• BS Computer Science\n• BS Software Engineering\n• BS Artificial Intelligence\n• BBA (Business Administration)\n• BS Electrical Engineering\n• BS Data Science\nWhich program are you interested in?"
    
    elif any(word in msg for word in ['fee', 'tuition', 'cost', 'scholarship', 'fees']):
        return "Approximate Tuition Fees (per semester):\n• BS Programs: PKR 130,000 – 190,000\n• Merit-based scholarships up to 75%\n• Need-based financial aid available\nContact admissions office for exact fee structure."
    
    elif any(phrase in msg for phrase in ['how to apply', 'application process', 'apply', 'admission form', 'online apply']):
        return "Step-by-Step Application Process:\n1. Visit admissions.lue.edu.pk\n2. Create an account\n3. Fill online application form\n4. Upload CNIC, transcripts, and photos\n5. Pay application fee (PKR 2,500)\n6. Submit & download acknowledgement"
    
    elif any(word in msg for word in ['contact', 'email', 'phone', 'help', 'office']):
        return "Admissions Office Contact:\n📧 admissions@lue.edu.pk\n📞 +92-42-111-123-456\n📍 Lahore, Punjab\nOffice Hours: 9 AM – 5 PM (Mon–Fri)"
    
    elif any(word in msg for word in ['thank', 'thanks', 'shukriya', 'jazakallah']):
        return "You're most welcome! 🎓 Best of luck with your admission to LUE. Feel free to ask anything else!"
    
    else:
        return "I'm not sure about that. You can ask me about:\n• Admission requirements\n• Deadlines\n• Programs offered\n• Fees & scholarships\n• How to apply\n\nHow else can I assist you?"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_message = data.get('message', '')
    response = get_bot_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)