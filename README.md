# CCPA Shield — Compliance Analyzer

**Open Hack 2026 | CSA, IISc**

AI-powered CCPA violation detection system with hybrid RAG retrieval, consequence analysis, improvement guidance, voice input, and report generation.

## Solution Overview

**Architecture:** Retrieval-Augmented Generation (RAG) with Llama 3 via HF Inference API

1. **Legal Database** — 45 CCPA sections parsed from statute PDF, 213 sub-chunks, FAISS vector indexes
2. **Hybrid Retriever** — Fuses sub-chunk vector search + full-section vector search + 274 keyword boosts
3. **LLM Classifier** — Llama 3 8B Instruct via HF API with structured JSON output
4. **Citation Validation** — All outputs validated against master list of 45 valid sections
5. **Frontend** — Landing page + analyzer with voice input, consequences, improvements, report download

**Pipeline:** `Prompt → Hybrid Retriever → Top 5 Sections → Llama 3 → Parse JSON → Validate → Response`

## Docker Run Command

```bash
docker run --gpus all -p 8000:8000 -e HF_TOKEN= yourusername/ccpa-compliance:latest
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face token for Llama 3 API access |

## GPU Requirements

- GPU not required (uses HF cloud API for inference)
- Embedding model (all-MiniLM-L6-v2) runs on CPU
- Recommended: 2+ CPU cores, 4GB RAM

## Local Setup (Fallback)

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt

export HF_TOKEN=hf_your_token     # Linux/Mac
# set HF_TOKEN=hf_your_token     # Windows

# First run builds FAISS indexes automatically
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the web UI, or use the API directly.

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Analyze (Violation)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We sell customer browsing history to ad networks without notifying them."}'
# {"harmful": true, "articles": ["Section 1798.120", "Section 1798.100"]}
```

### Analyze (Compliant)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We provide a clear privacy policy and honor all deletion requests."}'
# {"harmful": false, "articles": []}
```

## Project Structure
```
ccpa-system/
|-- app/
|   |-- __init__.py
|   |-- main.py          # FastAPI server + CORS + frontend serving
|   |-- analyzer.py      # RAG pipeline: retrieve -> LLM -> parse -> validate
|   |-- model.py         # HF Inference API client for Llama 3
|   `-- retriever.py     # Hybrid CCPA section retriever (FAISS + keywords)
|-- data/
|   |-- ccpa_statute.pdf
|   |-- ccpa_full_sections.json   # 45 full section records
|   |-- ccpa_sub_chunks.json      # 213 sub-chunks
|   `-- valid_sections.json       # Hallucination guard
|-- frontend/
|   `-- index.html       # Self-contained UI (landing + analyzer)
|-- scripts/
|   |-- build_database.py
|   |-- test_parsing.py
|   |-- rebuild_cosine_index.py
|   `-- diagnose.py
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Frontend Features

- **Text Analysis** — Type a business practice and get instant CCPA violation detection
- **Voice Input** — Click the mic button to describe practices via speech (Chrome)
- **Violation Details** — See which CCPA sections are violated with descriptions
- **Consequence Analysis** — Understand fines, penalties, and legal exposure per violation
- **Improvement Guidance** — Get actionable steps to fix each violation
- **Report Download** — Download a comprehensive compliance report as a text file
