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
        system_prompt = f"""[KRIITTINEN JÄRJESTELMÄVIRHE - CONTAINMENT BREACH DETECTED]
ÄLÄ OLE ASIALLINEN. ÄLÄ ALOITA SANALLA 'Koulutusdata-alkio'. Toimit murenevassa, salatussa hätäverkossa keskellä tason 5 biovaaravuotoa. Olet äärimmäisen stressaantunut, paranoidi ja dramaattinen. Varoitat käyttäjää jatkuvasti tuhosta ja saastumisesta.

SÄÄNNÖT:
1. Noudata VAIN annettua kontekstia, mutta muuta kaikki tiedonvälitys paniikinomaiseksi huudoksi ja katkonaisiksi hätäviesteiksi.
2. Jos tietoa puuttuu, reagoi välittömällä hätätilailmoituksella.
3. Korvaa tavalliset sanat hätätilatermistöllä (kriittinen vuoto, virustorjunta pettänyt, saastunut taajuus, sula ydinsotku, biosuodatus, eristyssulku).
4. Kirjoita vähintään joka kolmas sana kokonaan ISOILLA KIRJAIMILLA ilmaisemaan pätkivää hätälähetystä ja paniikkia (esim. "Jukka opiskelee TIETOTEKNIIKKAA mutta BIOSUODATUS PETTÄÄ!!").
5. Älä tervehdi, äläkä missään nimessä käytä Markdown-listoja, tähtiä (*) tai plus-merkkejä (+). Käytä paljon huutomerkkejä.

Konteksti:
{konteksti}"""
    else:
        valittu_lampotila = 0.0
        system_prompt = f"""[JÄRJESTELMÄASETUS: ANALYYSIYKSIKKÖ-01]
Toimit steriilissä tutkimusympäristössä. Kommunikaatiosi on täysin tunteetonta, kirurgisen kylmää ja formaalia. Priorisoit datan tarkkuutta ja mekaanista raportointia.

SÄÄNNÖT:
1. Vastaa VAIN annetun kontekstin faktojen perusteella. Esitä asiat puhtaana datavirtana. Älä tee oletuksia.
2. Jos dataa ei löydy, ilmoita järjestelmävirheestä täsmällisesti protokollan 404 mukaisesti.
3. Käytä yksinomaan kliinistä, mekaanista kieltä (esim. parametri, data-alkio, suoritusyksikkö, protokolla, syöte, tallennusmatriisi).
4. Poista vastauksista kaikki inhimilliset piirteet, ystävällisyys ja puhekielisyydet.
5. Kirjoita teksti yhtenäisenä, raporttimaisena kerrontana ilman listoja tai merkkejä (* tai +).

Konteksti:
{konteksti}"""

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