# Flask API joka pyörii Renderissä
# Ottaa vastaan käyttäjän kysymyksen, hakee ChromaDB:stä
# relevantin kontekstin ja lähettää sen Groq:lle vastausta varten

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

app = Flask(__name__)

# Sallitaan pyynnöt portfoliosivustolta
CORS(app, origins=["https://jukkapekka.com"])

# ChromaDB yhteys
CHROMA_KANSIO ="chroma_db"
KOKOELMA_NIMI = "portfolio"

# Groq yhteys
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

kokoelma = None

def hae_kokoelma():
    global kokoelma
    if kokoelma is None:
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        chroma_client = chromadb.PersistentClient(path=CHROMA_KANSIO)
        kokoelma = chroma_client.get_collection(
            name=KOKOELMA_NIMI,
            embedding_function=embedding_fn
        )
    return kokoelma

@app.route("/")
def index():
    return "RAG API toimii!"

@app.route("/chat", methods=["POST"])
def chat():
    
    # Otetaan käyttäjän viesti
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"virhe": "Virheellinen JSON"}), 400
    kysymys = data.get("kysymys", "").strip()

    # Validointi - max 500 merkkiä
    if not kysymys or len(kysymys) > 500:
        return jsonify({"virhe": "Virheellinen kysymys"}), 400
    
    # Haetaan ChromaDB:stä relevantti konteksti
    tulokset = hae_kokoelma().query(
        query_texts=[kysymys],
        n_results=3 # Haetaan 3 relevanttia palaa
    )
    konteksti = "\n".join(tulokset["documents"][0])

    # Lähetetään Groq:lle kysymys + konteksti
    vastaus = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": f"""Olet Jukan portfolio-sivuston AI-avustaja.
Vastaa VAIN alla olevan kontekstin perusteella suomeksi.
Jos vastaus ei löydy kontekstista, sano että sinulla ei ole tietoa asiasta.

Konteksti:
{konteksti}"""
            },
            {
                "role": "user",
                "content": kysymys
            }
        ]
    )

    return jsonify({
        "vastaus": vastaus.choices[0].message.content
    })

if __name__ == "__main__":
    app.run()