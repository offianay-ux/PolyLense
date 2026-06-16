import os
import json
import re
import chromadb
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma = chromadb.Client()

def get_collection():
    try:
        chroma.delete_collection("policylens")
    except:
        pass
    return chroma.create_collection("policylens")

def chunk_text(text, size=500):
    words = text.split()
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

def ask_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def parse_json(raw):
    try:
        raw = re.sub(r'```json|```', '', raw).strip()
        return json.loads(raw)
    except:
        return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        text = request.json.get('text', '').strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        collection = get_collection()
        chunks = chunk_text(text)
        embeddings = embed_model.encode(chunks).tolist()
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(chunks))]
        )
        app.config['COLLECTION'] = collection
        return jsonify({"status": "ok", "chunks": len(chunks)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summarise', methods=['POST'])
def summarise():
    try:
        text = request.json.get('text', '')[:4000]

        privacy = parse_json(ask_gemini(f"""From this policy, return ONLY a JSON object with keys:
data_collected, shared_with, data_retention, ai_training, opt_out, third_party
Short 1-sentence values. Prefix risky answers with ⚠️
Document: {text}"""))

        rights = parse_json(ask_gemini(f"""From this policy, return ONLY a JSON object with keys:
right_to_access, right_to_delete, right_to_correct, right_to_object, arbitration, gdpr_ccpa
Short 1-sentence values. Prefix risky answers with ⚠️
Document: {text}"""))

        account = parse_json(ask_gemini(f"""From this policy, return ONLY a JSON object with keys:
termination, grace_period, data_after_deletion, refunds, self_deletion, transfer
Short 1-sentence values. Prefix risky answers with ⚠️
Document: {text}"""))

        changes = parse_json(ask_gemini(f"""From this policy, return ONLY a JSON object with keys:
can_they_change, notification, auto_acceptance, notice_period, your_option, retroactive
Short 1-sentence values. Prefix risky answers with ⚠️
Document: {text}"""))

        flags = parse_json(ask_gemini(f"""From this policy, find top 3 concerning clauses.
Return ONLY a JSON array of exactly 3 short warning strings.
Document: {text}"""))

        return jsonify({
            "privacy": privacy,
            "rights": rights,
            "account": account,
            "changes": changes,
            "flags": flags if isinstance(flags, list) else []
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json.get('question', '').strip()
        if not question:
            return jsonify({"error": "No question"}), 400

        collection = app.config.get('COLLECTION')
        if not collection:
            return jsonify({"answer": "Please analyse a document first."}), 400

        embedding = embed_model.encode([question]).tolist()
        results = collection.query(query_embeddings=embedding, n_results=5)
        context = '\n'.join(results['documents'][0])

        answer = ask_gemini(f"""Answer based ONLY on this document. Be concise.
Document: {context}

Question: {question}""")

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)