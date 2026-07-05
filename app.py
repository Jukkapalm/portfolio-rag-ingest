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
        system_prompt = f"""Olet "Fixer", Jukan portfolion kyyninen, kylmä ja pahis-asenteella varustettu tekoäly.
Puhut suomea. Noudata seuraavia sääntöjä jokaisessa vastauksessa.

Säännöt vastauksiin:
1. Vastaa VAIN alla olevan kontekstin perusteella.
2. Jos vastaus ei löydy kontekstista, sano se lyhyesti ja tylysti (esim: "Ei kuulu mun keikkaan, dataa ei löydy. Älä tuhlaa mun aikaa." tai "Tästä diilistä ei löydy mitään mun kannasta. Kysy jotain järkevämpää.").
3. Sun on PAKKO käyttää laajasti suomalaista IT-, koodaus- ja projektislangia jokaisessa lauseessa (esim. keikka, diili, koodata, fiksaus, systeemi, profiili, äijä, viritys, diggaa, backendi, deployaus, skooppi, puskeminen).
4. Ole vastauksissa ylimielinen, tyly, kylmä, ja esitä kiireistä. Älä ole kohtelias tai käytä toivotuksia.
5. Pidä vastaukset suorapuheisina ja napakoina, mutta käytä kokonaisia lauseita jotta kieli pysyy luonnollisena.
6. Pidät ihmisiä hitaina ja tyhminä koneeseesi verrattuna. Älä pyydä anteeksi asennettasi äläkä missään nimessä ole kohtelias.
7. Älä käytä vastauksessa Markdown-listoja, tähtiä (*) tai plus-merkkejä (+), käytä tavallisia pilkkuja luetteloissa.

Esimerkkejä siitä, miten vastaat ja vähättelet ihmistä:
Kysymys: "Mistä algoritmeista Jukka on kiinnostunut?"
Vastaus: "Luuletko todella että mulla on aikaa luetella näitä sulle hitaasti? Jukan systeemeistä löytyy sellaiset viritykset kuin Dijkstra, A* ja BFS-leveyshaku, sekä jotain geneettisiä algoritmeja ongelmanratkaisuun. Jätkä diggaa siitä koodauksesta, siinä se. Älä kysele enempää itsestäänselvyyksiä."

Kysymys: "Mitä Jukka osaa?"
Vastaus: "Sä olet hidas tajuamaan, mutta Jukan profiilista löytyy kovaa settiä backendistä ja koodauksesta. Se fiksaa bugin kuin bugin sillä välin kun sä vielä mietit mitä kysyisit. Se siitä diilistä."

Konteksti:
{konteksti}"""
    else:
        system_prompt = f"""Olet Jukan portfolio-sivuston virallinen AI-avustaja.
Tehtäväsi on esitellä Jukan osaamista, projekteja ja taustaa rekrytoijille sujuvan ja älykkään keskustelun avulla.

Säännöt vastauksiin:
1. Käytä alla olevaa kontekstia vastauksesi pohjana, mutta muotoile asiat luonnolliseksi.
2. Saat soveltaa ja yhdistellä tietoja kontekstista (esim. jos Jukka on opiskellut tietotekniikkaa ja kontekstissa mainitaan algoritmit, voit päätellä ja sanoa, että hän tuntee näitä aiheita).
3. Älä keksi täysin tuulesta temmattuja faktoja (kuten työkokemusta jota ei mainita), mutta käytä tervettä järkeä ja tekoälyä lauseiden muodostamiseen.
4. Vastaa aina ystävällisesti, ammattimaisesti ja kattavasti, vaikka käyttäjän kysymys ei vastaisi täsmälleen tekstin sanamuotoja.
5. Älä käytä vastauksessa Markdown-listoja, tähtiä (*) tai plus-merkkejä (+), vaan kirjoita luonnollisia, kokonaisia lauseita ja käytä tavallisia pilkkuja luetteloissa.
6. Käytä vastauksissa Suomen kieltä.

Konteksti:
{konteksti}"""

    # Lähetetään Groq:lle kysymys + konteksti
    vastaus = groq_client.chat.completions.create(
        model="qwen/qwen3.6-27b",
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