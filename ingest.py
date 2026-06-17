# Tämä ajetaan lokaalisti
# Lukee data kansion tekstitiedostot
# Pilkkoo ne paloihin ja tallentaa chromaDB-vektorikantaan
# Tämä ajetaan aina kun dataa lisätään tai muokataan

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()

# Asetukset
DATA_KANSIO = "data"
CHROMA_KANSIO = "chroma_db"
KOKOELMA_NIMI = "portfolio"

# Chunking asetukset
# Montako merkkiä per pala
# Kuinka monta merkkiä palat jakavat keskenään
CHUNK_KOKO = 500
CHUNK_PAALLEKKAIN = 50

embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# Yhdistetään ChromaDB:hen (luo kansion jos ei ole)
client = chromadb.PersistentClient(path=CHROMA_KANSIO)

# Poistetaan vanha kokoelma jos on, jotta päivitys toimii puhtaasti
try:
    client.delete_collection(KOKOELMA_NIMI)
except:
    pass

# Luodaan uusi kokoelma
kokoelma = client.create_collection(
    name=KOKOELMA_NIMI,
    embedding_function=embedding_fn
)

def pilko_teksti(teksti, koko, paallekkain):

    # Pilkkoo tekstin paloihin joissa on päällekkäisyyttä
    palat = []
    alku = 0
    while alku < len(teksti):
        loppu = alku + koko
        pala = teksti[alku:loppu]
        palat.append(pala)
        alku += koko - paallekkain
    return palat

# Luetaan kaikki .txt tiedostot data/-kansiosta
dokumentit = []
id_lista = []

for tiedosto in os.listdir(DATA_KANSIO):
    if tiedosto.endswith(".txt"):
        polku = os.path.join(DATA_KANSIO, tiedosto)
        with open(polku, "r", encoding="utf-8") as f:
            teksti = f.read().strip()

        # Pilkotaan teksti paloihin
        palat = pilko_teksti(teksti, CHUNK_KOKO, CHUNK_PAALLEKKAIN)

        for i, pala in enumerate(palat):
            dokumentit.append(pala)

            # ID muodostuu tiedostonimestä ja palan numerosta
            id_lista.append(f"{tiedosto}_chunk_{i}")

        print(f"Luettu: {tiedosto} ({len(palat)} palaa)")

# Tallennetaan ChromaDB:hen
kokoelma.add(
    documents=dokumentit,
    ids=id_lista
)

print(f"\nValmis! {len(dokumentit)} palaa tallennettu ChromaDB:hen.")