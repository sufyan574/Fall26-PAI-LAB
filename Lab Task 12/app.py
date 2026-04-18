from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import csv

app = Flask(__name__)

DATA_PATH = 'data/admission_qna.csv'
INDEX_PATH = 'faiss_index/admission_index.faiss'

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

model = SentenceTransformer(MODEL_NAME)

df_global = None
faiss_index = None

if os.path.exists(INDEX_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
    df_global = pd.read_csv(DATA_PATH, quoting=csv.QUOTE_ALL)
    print("Loaded existing FAISS index.")
else:
    df_global = pd.read_csv(DATA_PATH, quoting=csv.QUOTE_ALL)
    df_global['combined'] = df_global['question'].astype(str) + " " + df_global['answer'].astype(str)
    
    embeddings = model.encode(df_global['combined'].tolist(), convert_to_numpy=True).astype('float32')
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    
    os.makedirs('faiss_index', exist_ok=True)
    faiss.write_index(faiss_index, INDEX_PATH)
    print(f"FAISS index created with {len(df_global)} entries.")

def retrieve_answer(query, k=3):
    global faiss_index, df_global
    query_vec = model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = faiss_index.search(query_vec, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(df_global):
            results.append(str(df_global.iloc[idx]['answer']))
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        data = request.get_json(silent=True)
        user_query = data.get('message', '').strip() if data and isinstance(data, dict) else ''
        
        if not user_query:
            return jsonify({'response': "Please ask a question about university admissions."})
        
        retrieved = retrieve_answer(user_query, k=3)
        
        if not retrieved:
            response = "Sorry, I could not find any relevant information. Please try rephrasing your question."
        else:
            # Best answer as main response
            response = "**Answer:** " + retrieved[0]
            
            # Show other relevant only if they are quite different
            if len(retrieved) > 1:
                response += "\n\n**More information:**\n"
                for ans in retrieved[1:]:
                    if ans[:100] not in retrieved[0]:   # avoid repetition
                        response += "• " + ans[:180] + "...\n"
        
        return jsonify({'response': response})
    
    except Exception as e:
        print("Backend Error:", str(e))
        return jsonify({'response': "Sorry, something went wrong. Please try again."})

if __name__ == '__main__':
    app.run(debug=True)