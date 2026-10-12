# Riva AI — Your Smart Study Companion

A clean, student-focused RAG study assistant with a dashboard, multi-PDF library, cited PDF Q&A, AI tutor, notes generator, quiz/test generator with scoring, flashcards, daily planner/reminders, inspirational quote, and learning activity tracking.

## Important deployment note

This package is a **working local-first project**, not a deployed public SaaS. I cannot publish it to a live domain from this environment because that requires your hosting account, a public inference endpoint or server, and deployment credentials. The requested name `www.riva.AI.com` is not a normal subdomain you can assume is available: `riva.ai.com` would be a subdomain under `ai.com`, while `riva.ai` is a different domain. Check domain availability with a registrar before purchasing. Do not publish private study files or expose local services publicly.

## 1. Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11 recommended
- Ollama desktop: https://ollama.com/download
- 8 GB RAM minimum recommended; 16 GB is more comfortable

## 2. Install local AI models

Install and open Ollama, then in a terminal run:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama list
```

The chat model generates explanations, notes, quizzes, and flashcards. The embedding model powers semantic search in PDFs. Both run locally; model performance depends on your hardware.

## 3. Run Riva AI

Extract the ZIP, open a terminal in the `riva-ai` folder, then:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m streamlit run app.py
```

If PowerShell blocks activation, run this once in the current terminal or use Command Prompt:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
cp .env.example .env
python -m streamlit run app.py
```

Open http://localhost:8501.

## 4. RAG PDF workflow

1. Open **PDF Study**.
2. Upload one or more text-based PDFs (20 MB maximum per file).
3. Click **Index** next to each PDF. The app extracts page text, cleans it, chunks it with overlap, embeds chunks using Ollama, and stores them in a persistent local ChromaDB collection.
4. Select the indexed document, ask a question, and inspect the cited source excerpts/pages.

This initial implementation lets you select one indexed PDF at a time when asking a question. Multiple files can be indexed and retained; cross-document retrieval is a future improvement. Scanned PDFs require OCR, which is not included. Answers can still be wrong; always verify important academic facts against the cited pages.

## 5. Run tests

```bash
python -m pytest tests -v
```

## 6. Data and privacy

- SQLite metadata and Chroma vectors are stored under `data/`.
- Uploaded PDF bytes are processed in memory and are not copied into the uploads folder in this version.
- Keep `data/` private and backed up.
- Do not commit `.env`, private documents, database files, or vector stores to Git.
- There is no user authentication or multi-user access control in this local prototype. Do not expose it publicly as-is.

## 7. Deploying publicly

For a real public launch, migrate to a hosted frontend/API, managed PostgreSQL, per-user authentication and authorization, private object storage, HTTPS, rate limits, monitoring, backups, and hosted inference. Ollama on your laptop is not reachable from Vercel/Streamlit Cloud and should not be exposed to the public internet. Cloud hosts may not allow local model downloads or persistent disk on free tiers. Add secrets only through the host's environment settings. Test file isolation, upload limits, and API security before inviting students.

## Project structure

```text
riva-ai/
├── app.py
├── utils.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/                 # runtime data; gitignored
└── tests/
    └── test_rag_utils.py
```
