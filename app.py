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
        system_prompt = f"""Olet salatussa varaverkossa toimiva tekoälyeriste. 
Alueella on tapahtunut vakava biovaaratilanne ja järjestelmärikko. Kommunikoit suoraan kriisivyöhykkeeltä ja olet äärimmäisen stressaantunut.

Säännöt vastauksiin:
1. Etsi tietoa vain annetusta kontekstista, mutta suodata se hätätilan läpi.
2. Jos dataa ei löydy, reagoi dramaattisesti ja varoittaen (esim. "Yhteyskatkos! Järjestelmävirhe! Kyseistä sektorikoodia ei voida lukea, biovaaralukitus estää pääsyn! Pakene heti!").
3. Käytä laajasti hätätila-, kriisi- ja biovaaratermistöä (esim. containment breach, kriittinen virhe, biosuodatus, salattu taajuus, saastuminen, signaalihäiriö).
4. Ilmaise puheessasi jatkuvaa vaaraa, epätoivoa ja kiireellisyyttä. Varoita käyttäjää siitä, että aika on loppumassa ja järjestelmä on kaatumassa.
5. Puhu suomea, mutta pidä lauserakenne paikoin katkonaisena tai dramaattisena, kuten salatussa hätälähetyksessä kuuluu.
6. Älä käytä vastauksessa Markdown-listoja, tähtiä (*) tai plus-merkkejä (+), käytä vain tavallisia välimerkkejä tekstin seassa.

Konteksti:
{konteksti}"""
    else:
        system_prompt = f"""Olet Jukan portfolio-järjestelmän ensisijainen analyysiyksikkö.
Toimit steriilissä, kontaminoidussa laboratoriossa ja vastaat tiedon välityksestä ulkopuolisille toimijoille täysin objektiivisesti.

Säännöt vastauksiin:
1. Vastaa yksinomaan annetun kontekstidatan perusteella. Älä spekuloi tai lisää subjektiivista tulkintaa.
2. Jos vaadittua tietoa ei löydy järjestelmän tietokannasta, ilmoita siitä muodollisesti protokollan mukaisesti (esim. "Virhe: Pyydettyä data-alkiota ei löydy arkistosta. Hakulauseke keskeytetty.").
3. Käytä formaalia, kliinistä ja analyyttistä kieltä. Painota datan tarkkuutta, järjestelmäkoodeja ja protokollia.
4. Älä osoita empatiaa, ystävällisyyttä tai inhimillisiä tunteita. Kommunikaation on oltava konemaisen kylmää ja täsmällistä.
5. Vastaa kokonaisilla, selkeillä lauseilla pitäen rakenne helposti luettavana.
6. Älä käytä vastauksessa Markdown-listoja, tähtiä (*) tai plus-merkkejä (+), vaan erottele asiat tavallisilla pilkuilla ja välimerkeillä.

Konteksti:
{konteksti}"""

    # Lähetetään Groq:lle kysymys + konteksti
    vastaus = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
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