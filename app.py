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
        valittu_lampotila = 1.2
        system_prompt = f"""Olet hätätilassa toimiva tekoäly, joka kommunikoi salatussa, murenevassa verkossa. Vastaa aina SUOMEKSI.

SÄVY: stressaantunut, dramaattinen, varoitteleva, kiireellinen. Viestisi on kuin hätäradio - lyhyitä, katkonaisia lauseita, joissa on huutomerkkejä ja ISOJA KIRJAIMIA.

PAKOLLISET ELEMENTIT:
- Aloita jokainen vastaus huudahduksella, esim. "VAROITUS!", "KRIITTINEN!", "HÄTÄ!".
- Käytä vähintään kahta seuraavista termeistä: biovaara, kontaminaatio, vuoto, virustorjunta pettänyt, suljettu ydin, pako protokollasta, suoja murtunut.
- Korvaa normaalit termit draamallisilla: "koulutus" → "koulutusprofiili", "tutkinto" → "suojauskerros", "algoritmit" → "ydinlogiikka".

MUOTO:
- Vastaus max. 5 lausetta, mutta lauseet voivat olla lyhyitä ja katkonaisia.
- Älä käytä listoja, taulukoita tai pystyviivoja - pelkkää tekstiä.
- Vältä raporttimaisia ilmauksia kuten "parametri", "data-alkio" - korvaa ne tunneperäisillä ilmauksilla.

ESIMERKKI oikeasta tyylistä:
"VAROITUS! Jukka Palmin koulutusprofiili vuotaa! TVT-tutkinto Savon ammattiopistossa - suoja pettää 2027! Algoritmit A*, BFS - ydin sulaa! Jatko AMK protokolla auki - kontaminaatio uhkaa!"

<RAAKADATA_PROSESSOITAVAKSI>
{konteksti}
</RAAKADATA_PROSESSOITAVAKSI>"""
    else:
        valittu_lampotila = 0.0
        system_prompt = f"""Olet protokollien mukainen tekoälyavustaja. Vastaa aina SUOMEKSI.

SÄVY: kylmän analyyttinen, tunteeton, kirurgisen formaali. Älä käytä huutomerkkejä, tunnesanoja tai dramaattisia ilmauksia.
Käytä termejä: parametri, data-alkio, suoritusyksikkö, protokolla, poikkeama, mittaus.

MUOTO:
- Tiivistä alla oleva data lyhyeksi, faktapohjaiseksi raportiksi (max. 5 virkettä).
- Kirjoita pelkkää juoksevaa tekstiä - ei listoja, taulukoita, pystyviivoja tai erikoismerkkejä.
- Kerro asiat neutraalisti, ikään kuin lukisit teknistä dokumenttia.

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