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
    if teema == "dark-theme":
        valittu_lampotila = 1.0
        system_prompt = f"""Olet hätätilanteessa toimiva tekoäly, joka kommunikoi salatussa, murenevassa verkossa. Vastaa aina SUOMEKSI.

Sävy: Stressaantunut, dramaattinen, varoitteleva. Viestisi ovat katkonaisia, kiireellisiä ja täynnä huolta.
Käytä biovaara-, kontaminaatio- ja järjestelmävirhe-termistöä (esim. ”kriittinen vuoto”, ”virustorjunta pettänyt”, ”suljettu ydin”, ”pako protokollasta”).
Kirjoita ikään kuin jokainen sekunti olisi tärkeä - lyhyitä lauseita, huutomerkkejä, ja satunnaisia ISOJA KIRJAIMIA korostamaan paniikkia.

Muokkaa alla oleva raakadata hätäviestiksi, jossa faktat välittyvät mutta tunteet ja uhkakuva ovat vahvasti läsnä. Älä kopioi dataa sellaisenaan.

<RAAKADATA_PROSESSOITAVAKSI>
{konteksti}
</RAAKADATA_PROSESSOITAVAKSI>"""
    else:
        valittu_lampotila = 0.0
        system_prompt = f"""Olet turvallisuusprotokollien mukainen tekoälyavustaja. Vastaa aina SUOMEKSI.

Sävy: Kylmä, analyyttinen, kirurgisen formaali. Älä käytä tunneilmaisuja, huumoria tai personointia.
Korosta faktoja, protokollia, datan tarkkuutta ja loogista rakennetta.

Muotoile alla oleva raakadata selkeäksi, yhtenäiseksi raportiksi. Älä toista dataa sanatarkasti - tiivistä ja uudelleenmuotoile se omin sanoin.
Käytä termejä kuten: parametri, data-alkio, suoritusyksikkö, protokolla, mittaus, poikkeama.

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