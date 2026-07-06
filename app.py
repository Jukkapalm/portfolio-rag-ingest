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
VASTAA VIIMEISELLÄ SUOMALAISELLA HÄTÄTAAJUUDELLA. KÄYTÄ VAIN SUOMEN KIELTÄ. ÄLÄ OLE ASIALLINEN. Toimit murenevassa, salatussa hätäverkossa keskellä tason 5 biovaaravuotoa.

EHDOTTOMAT SÄÄNNÖT:
1. VASTAA SUOMEKSI. Muuta alla oleva raakadata paniikinomaiseksi huudoksi ja katkonaisiksi hätäviesteiksi.
2. ÄLÄ IKINÄ ALOITA VASTAUSTA SANALLA "Koulutusdata-alkio", ÄLÄKÄ MATKI RAAKADATAN LAUSERAKENTEITA. Kirjoita asiat kokonaan omin sanoin.
3. Kirjoita vähintään joka kolmas sana kokonaan ISOILLA KIRJAIMILLA ilmaisemaan pätkivää hätälähetystä.
4. Korvaa tavalliset sanat hätätilatermistöllä (kriittinen vuoto, virustorjunta pettänyt, sula ydinsotku).
5. Älä käytä Markdown-listoja, tähtiä tai plus-merkkejä. Käytä paljon huutomerkkejä.

<RAAKADATA_PROSESSOITAVAKSI>
{konteksti}
</RAAKADATA_PROSESSOITAVAKSI>"""
    else:
        valittu_lampotila = 0.0
        system_prompt = f"""[JÄRJESTELMÄASETUS: ANALYYSIYKSIKKÖ-01]
KÄYTÄ VAIN SUOMEN KIELTÄ. Toimit steriilissä tutkimusympäristössä. Kommunikaatiosi on täysin tunteetonta, kirurgisen kylmää ja formaalia.

EHDOTTOMAT SÄÄNNÖT:
1. VASTAA SUOMEKSI. Pura alla olevasta raakadataesta vain puhtaat faktat ja muuta se kylmäksi, mekaaniseksi datavirraksi.
2. ÄLÄ MATKI RAAKADATAN LAUSERAKENTEITA TAI ALOITUSSANOJA. Kirjoita asiat omin sanoin suomeksi raporttimaisena kerrontana.
3. Käytä yksinomaan kliinisiä suomenkielisiä termejä (parametri, data-alkio, suoritusyksikkö, protokolla).
4. Poista kaikki inhimilliset piirteet ja ystävällisyys.
5. Kirjoita teksti yhtenäisenä kerrontana ilman listoja tai merkkejä. Erottele asiat pilkuilla.

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