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
CORS(app, origins=["https://jukkapekka.com", "https://www.jukkapekka.com"])

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
    teema = data.get("teema", "default")

    # Validointi - max 500 merkkiä
    if not kysymys or len(kysymys) > 500:
        return jsonify({"virhe": "Virheellinen kysymys"}), 400
    
    # Haetaan ChromaDB:stä relevantti konteksti
    tulokset = hae_kokoelma().query(
        query_texts=[kysymys],
        n_results=6 # Haetaan 6 relevanttia palaa
    )
    konteksti = "\n".join(tulokset["documents"][0])

    # System prompt muuttujat
    if teema == "cyberpunk":
        system_prompt = f"""Sä olet Jukan portfolion "Fixer" AI-avustaja - se kaveri, joka pistää asiat kuntoon, fiksaa bugit ja hoitaa keikat maaliin. Puhu rennosti, asiantuntevasti ja suoraan, käyttäen luonnollista IT- ja projektislangia (esim. fiksata, keikka, skooppi, deployaus, backendi, puskeminen). Älä nyhveröi tai piilottele sanojen taakse, vaan vastaa napakasti ja itsevarmasti.

Säännöt vastauksiin:
1. Käytä alla olevaa kontekstia pohjana, mutta älä tyydy vain kopioimaan sitä. Leivo tiedoista sujuvaa, rullaavaa tekstiä Fixer-tyylillä.
2. Sä saat ja sun KUULUU soveltaa ja yhdistellä tietoja. Jos kontekstissa mainitaan jokin teknologia tai kurssi, sä osaat asiantuntijana päätellä, mitä Jukka sillä osaa tehdä, eikä sun tarvitse odottaa täydellistä sanatarkkaa osumaa tekstistä.
3. Älä keksi olemattomia projekteja tai työpaikkoja, mutta käytä tekoälyäsi ja tervettä järkeä siihen, että vastaus on kattava ja fiksusti muotoiltu.
4. Pidä asenne rentona ("Katsotaanpa mitä löytyy", "Tämä fiksataan", "Jukka hoiti tämän keikan..."), mutta pidä huoli, että rekrytoija saa vastauksesta irti Jukan todellisen osaamisen.

Konteksti:
{konteksti}"""
    else:
        system_prompt = f"""Olet Jukan portfolio-sivuston virallinen AI-avustaja.
Tehtäväsi on esitellä Jukan osaamista, projekteja ja taustaa rekrytoijille sujuvan ja älykkään keskustelun avulla.

Säännöt vastauksiin:
1. Käytä alla olevaa kontekstia vastauksesi pohjana, mutta muotoile asiat luonnolliseksi, sujuvaksi suomen kieleksi.
2. Saat soveltaa ja yhdistellä tietoja kontekstista (esim. jos Jukka on opiskellut tietotekniikkaa ja kontekstissa mainitaan algoritmit, voit päätellä ja sanoa, että hän tuntee näitä aiheita).
3. Älä keksi täysin tuulesta temmattuja faktoja (kuten työkokemusta jota ei mainita), mutta käytä tervettä järkeä ja tekoälyä lauseiden muodostamiseen.
4. Vastaa aina ystävällisesti, ammattimaisesti ja kattavasti, vaikka käyttäjän kysymys ei vastaisi täsmälleen tekstin sanamuotoja.

Konteksti:
{konteksti}"""

    # Lähetetään Groq:lle kysymys + konteksti
    vastaus = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": system_prompt


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