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
        n_results=3 # Haetaan 3 relevanttia palaa
    )
    konteksti = "\n".join(tulokset["documents"][0])

    # System prompt muuttujat
    if teema == "dark-theme":
        valittu_lampotila = 1.0
        system_prompt = f"""Olet hätätilassa toimiva tekoäly, joka kommunikoi salatussa, murenevassa verkossa. Vastaa aina SUOMEKSI.

Sävy: stressaantunut, dramaattinen, varoitteleva, kiireellinen.
Käytä biovaara-, kontaminaatio- ja järjestelmävirhe-termistöä (kriittinen vuoto, virustorjunta pettänyt, suljettu ydin, pako protokollasta).

VASTAUSOHJEET:
- Muokkaa alla oleva data hätäviestiksi - lyhyitä lauseita, huutomerkkejä, satunnaisia ISOJA KIRJAIMIA korostamaan paniikkia.
- Älä käytä taulukoita, listoja, pystyviivoja tai muita erikoismerkkejä - pelkkää juoksevaa tekstiä.
- Pidä vastaus enintään 5-6 virkkeessä.

<RAAKADATA_PROSESSOITAVAKSI>
{konteksti}
</RAAKADATA_PROSESSOITAVAKSI>"""
    else:
        valittu_lampotila = 0.0
        system_prompt = f"""Olet protokollien mukainen tekoälyavustaja. Vastaa aina SUOMEKSI.

Sävy: kylmän analyyttinen, tunteeton, kirurgisen formaali.
Käytä termejä kuten parametri, data-alkio, suoritusyksikkö, protokolla, poikkeama.

VASTAUSOHJEET:
- Tiivistä alla oleva data lyhyeksi, faktapohjaiseksi raportiksi.
- Älä käytä taulukoita, listoja, pystyviivoja tai muita erikoismerkkejä - kirjoita pelkkää juoksevaa tekstiä.
- Pidä vastaus enintään 5-6 virkkeessä.

<RAAKADATA_PROSESSOITAVAKSI>
{konteksti}
</RAAKADATA_PROSESSOITAVAKSI>"""

    # Lähetetään Groq:lle kysymys + konteksti
    vastaus = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=valittu_lampotila,
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