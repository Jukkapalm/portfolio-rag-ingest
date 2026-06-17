# portfolio-rag-ingest

RAG-pohjainen (Retrieval-Augmented Generation) AI-botti Jukan portfolio-sivustolle. Botti vastaa kysymyksiin Jukan osaamisesta, projekteista ja koulutuksesta tekstitiedostoihin tallennetun datan perusteella.

🤖 **Käytössä: [jukkapekka.com](https://jukkapekka.com)**

---

## Arkkitehtuuri

```
Lokaali kone                    Render.com (ilmainen)
────────────────                ─────────────────────
data/*.txt                      Flask API (app.py)
    ↓                               ↓
ingest.py                       ChromaDB (vektorikanta)
    ↓                               ↓
chroma_db/  ──── git push ────→  Groq API (Llama 3.3 70B)
                                    ↓
                              Vastaus käyttäjälle
```

### Miten toimii

1. Tekstitiedostot kirjoitetaan lokaaliasti `data/` kansioon
2. `ingest.py` lukee tiedostot, pilkkoo ne 500 merkin paloihin ja tallentaa vektorit ChromaDB-kantaan
3. Vektorikanta pushataan GitHubiin → päivittyy automaattisesti Renderiin
4. Käyttäjä kirjoittaa kysymyksen portfoliossa
5. JavaScript lähettää kysymyksen ja sivun teematiedon Flask API:lle
6. ChromaDB hakee semanttisesti lähimmät tekstipalat
7. Groq (Llama 3.3 70B) muodostaa vastauksen kontekstin perusteella
8. Vastaus palautuu käyttäjälle

---

## Teknologiat
 
| Teknologia | Käyttötarkoitus |
|---|---|
| Python / Flask | REST API Renderillä |
| ChromaDB | Vektoritietokanta semanttiseen hakuun |
| all-MiniLM-L6-v2 | Embedding-malli (ajetaan lokaaliasti) |
| Groq API | LLM-palvelu (Llama 3.3 70B) |
| Render.com | Ilmainen pilvihosting Flask API:lle |
| flask-cors | Cross-origin pyyntöjen hallinta |
| gunicorn | Tuotantopalvelin |

---

## Rakenne

```
portfolio-rag-ingest/
├── data/                   # Tekstitiedostot (ei githubissa)
│   ├── minusta.txt
│   ├── koulutus.txt
│   ├── taidot.txt
│   └── projekti_*.txt
├── chroma_db/              # ChromaDB vektorikanta (githubissa)
├── app.py                  # Flask API
├── ingest.py               # Lokaali ingest-skripti
├── requirements.txt        # Python riippuvuudet
├── render.yaml             # Render konfiguraatio
└── .gitignore
```

---

## Asennus ja käyttö

### Vaatimukset

- Python 3.10+
- Groq API-avain (ilmainen: [console.groq.com](https://console.groq.com))

### 1. Kloonaa repo

```bash
git clone https://github.com/Jukkapalm/portfolio-rag-ingest
cd portfolio-rag-ingest
```

### 2. Asenna riippuvuudet

```bash
pip install chromadb python-dotenv
```

### 3. Luo .env tiedosto

```
GROQ_API_KEY=sinun_groq_avaimesi
```

### 4. Lisää tekstitiedostoja

Kirjoita tekstitiedostoja `data/` kansioon:

```
data/minusta.txt
data/koulutus.txt
data/projektit.txt
```

### 5. Aja ingest

```bash
python ingest.py
```

### 6. Pushaa GitHubiin

```bash
git add .
git commit -m "Päivitetty data"
git push
```

Render deployaa automaattisesti.

---

## AI-botin kaksi persoonaa

Botti vaihtaa persoonaa portfolion teeman mukaan:

**AI-avustaja** (vaalea teema) — selkeä ja kannustava

**Fixer** (tumma-teema) — ylimielinen, slangia käyttävä pimeiden markkinoiden välittäjä

---

## Ympäristömuuttujat Renderissä

| Muuttuja | Kuvaus |
|---|---|
| `GROQ_API_KEY` | Groq API-avain |

---

## Tekijä

**Jukka Palm**
- GitHub: [github.com/Jukkapalm](https://github.com/Jukkapalm)
- LinkedIn: [linkedin.com/in/jukkapalm](https://linkedin.com/in/jukkapalm)