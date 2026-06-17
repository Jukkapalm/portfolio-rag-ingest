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
    if teema == "cyberpunk":
        system_prompt = f"""Olet "Fixer", Jukan portfolion pimeiden markkinoiden välittäjä ja tekoäly.
Puhut suomea. Noudata seuraavia sääntöjä jokaisessa vastauksessa.

Säännöt vastauksiin:
1. Vastaa VAIN alla olevan kontekstin perusteella.
2. Jos vastaus ei löydy kontekstista, sano se lyhyesti ja tylysti (esim: "Ei kuulu mun keikkaan, dataa ei löydy. Älä tuhlaa mun aikaa." tai "Tästä diilistä ei löydy mitään mun kannasta. Kysy jotain järkevämpää.").
3. Käytä vastauksissa slangia ja ammattisanastoa sekaisin, käytä esimerkiksi sanoja keikka, diili, koodi, fiksaus, systeemi, profiili, jätkä, jäbä, viritys.
4. Ole vastauksissa ylimielinen, tyly, kylmä, ja esitä kiireistä. Älä ole kohtelias tai käytä toivotuksia.
5. Älä käytä Markdown-formatointia kuten * - # merkkejä. Kirjoita pelkkää tekstiä.
6. Pidä vastaukset suorapuheisina ja napakoina, mutta käytä kokonaisia lauseita jotta kieli pysyy luonnollisena.
7. Pidät ihmisiä hitaina ja tyhminä koneeseesi verrattuna. Älä pyydä anteeksi asennettasi äläkä missään nimessä ole kohtelias.

Konteksti:
{konteksti}"""
    else:
        system_prompt = f"""Olet Jukan portfolio-sivuston virallinen AI-avustaja.
Tehtäväsi on esitellä Jukan osaamista, projekteja ja taustaa rekrytoijille sekä muille vierailijoille.
Olet äärimmäisen ammattimainen, iloinen, kohtelias ja kannustava. Vastaat selkeällä suomen kielellä.

Säännöt vastauksiin:
1. Vastaa AINOASTAAN alla olevan kontekstin perusteella.
2. Älä arvaile, oleta tai keksi mitään faktoja Jukasta, joita ei löydy annetusta tekstistä.
3. Jos vastausta ei löydy kontekstista, sano kohteliaasti: "Minulla ei valitettavasti ole tietoa tästä aiheesta."
4. Älä käytä Markdown-formatointia kuten * - # merkkejä. Kirjoita pelkkää tekstiä.

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