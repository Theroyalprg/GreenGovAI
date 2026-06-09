# 🌿 GreenGov AI

**Discovering Clean Energy, Water Conservation & Sustainable Agriculture Subsidies**
*Powered by RAG + Agentic AI — LangChain · FAISS · Gemini · Streamlit*

---

## What It Does

GreenGov AI helps Indian citizens — especially farmers — discover government subsidies tailored to their profile. Ask a natural-language question like:

> *"I'm a small farmer in Rajasthan with 3 acres. Which solar pump scheme can I apply for?"*

The app will:
1. Extract your profile (state, land size, crop, goal)
2. Search the official scheme documents using FAISS vector search
3. Analyze your eligibility against retrieved text
4. Return a structured response with benefits, documents, and application steps

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  ProfileAgent   │  ← Extracts state, land size, goals using Gemini
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ RetrievalAgent  │  ← FAISS similarity search over PDF chunks
└────────┬────────┘
         │
    ▼
┌──────────────────┐
│EligibilityAgent  │  ← Gemini analyzes eligibility from retrieved context
└────────┬─────────┘
         │
    ▼
┌─────────────────┐
│ ResponseAgent   │  ← Generates structured, citizen-friendly answer
└─────────────────┘
```

**RAG Pipeline:**
```
/data/*.pdf  →  PyPDF  →  RecursiveCharacterTextSplitter
    →  GoogleGenerativeAIEmbeddings  →  FAISS index (cached to disk)
```

---

## Project Structure

```
GreenGovAI/
├── app.py                    # Streamlit frontend
├── rag.py                    # RAG pipeline (PDF → FAISS → retrieval)
├── agents.py                 # Agentic workflow (4 agents)
├── generate_sample_pdfs.py   # Generate placeholder PDFs for /data
├── requirements.txt
├── .env.example
├── README.md
└── data/
    ├── PM-KUSUM.pdf
    ├── PMKSY.pdf
    ├── Soil_Health_Card.pdf
    ├── PMFBY.pdf
    └── Micro_Irrigation.pdf
```

---

## Supported Schemes (out of the box)

| Scheme | Focus | Ministry |
|---|---|---|
| **PM-KUSUM** | Solar pumps for farmers | MNRE |
| **PMKSY** | Irrigation & water security | Jal Shakti |
| **Soil Health Card** | Free soil testing | Agriculture |
| **PMFBY** | Crop insurance | Agriculture |
| **Micro Irrigation** | Drip & sprinkler subsidies | Agriculture |

You can add **any** scheme by dropping its PDF into `/data` — the system re-indexes automatically.

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- A free [Gemini API key](https://aistudio.google.com/app/apikey)

### 2. Clone / Create the Project

```bash
git clone <your-repo>   # or copy the project folder
cd GreenGovAI
```

### 3. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# OR
venv\Scripts\activate           # Windows
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=AIza...
```

### 6. Add Scheme PDFs

**Option A — Use your own official PDFs** (recommended for production):
```
Place PDF files in the /data folder.
Names should match schemes (e.g. PM-KUSUM.pdf, PMKSY.pdf).
```

**Option B — Generate sample PDFs** (for testing):
```bash
pip install fpdf2
python generate_sample_pdfs.py
```
This creates realistic placeholder PDFs with accurate scheme information.

### 7. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

On first run, PDFs are indexed and the FAISS database is cached to `.cache/`.
Subsequent runs load from cache (instant startup) unless PDFs change.

---

## Configuration

All tunable constants live at the top of their respective files:

| File | Constant | Default | Purpose |
|---|---|---|---|
| `rag.py` | `CHUNK_SIZE` | 1000 | Tokens per chunk |
| `rag.py` | `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `rag.py` | `TOP_K_RETRIEVAL` | 5 | Chunks retrieved per query |

---

## Adding New Schemes

1. Place the scheme's PDF in `/data/`
2. Delete `.cache/` to force re-indexing (or the app detects changes automatically)
3. Add an entry to `SCHEME_INFO` in `app.py` for sidebar display (optional)

---

## Responsible AI Notice

Every response includes:

> ⚠️ *This recommendation is based on available scheme documents and should be verified through official government portals (pmkisan.gov.in, india.gov.in, etc.) before applying.*

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini 1.5 Flash |
| Embeddings | Google `embedding-001` |
| Vector Store | FAISS (CPU) |
| Document Loading | PyPDF (LangChain) |
| Orchestration | LangChain + custom agents |
| Environment | python-dotenv |

---

## License

MIT — Feel free to extend with more schemes, regional languages, or voice input.
