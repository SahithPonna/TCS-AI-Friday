# Capital Markets Regulatory Compliance Query Agent

## Problem Statement
Regulatory requirements in capital markets are complex and frequently updated, making compliance verification challenging. Staff must interpret dense legal texts and map them to internal processes. A generative AI agent that answers compliance-related questions by referencing regulatory documents can reduce reliance on legal experts, speed decision-making, foster adherence to regulations, and minimize risk.

## Data Requirements
Provide folders of regulatory documents (PDF or TXT) such as SEC filings, Basel accords, market rules. Supply document metadata: effective dates, jurisdiction, hierarchy. Optionally add compliance FAQs & historical inquiry logs.

Place documents in the `data/` folder. Optionally create a `metadata.json` alongside each file or a global `metadata.csv`.

## Expected Output
A chat UI (Streamlit) that:
- Accepts natural language compliance questions.
- Retrieves relevant regulatory passages.
- Returns answers with cited excerpts + summarized obligations.
- Allows export of Q&A transcript and reference list to PDF.

## Quick Start
1. (Recommended) Set execution policy if needed for venv activation (PowerShell):
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force`
2. Activate environment: `./myenv/Scripts/Activate.ps1`
3. Install deps: `pip install -r requirements.txt`
4. Ensure OpenAI key is stored in `api_key.txt` as: `API-key = "sk-..."` (or set `OPENAI_API_KEY` env var).
5. (Optional simple gate) The file in `token/` holds an access token. Provide that value in the UI when prompted.
6. Add PDF/TXT files under `data/`.
7. Run ingestion: `python ingestion.py` (creates / updates local Chroma vector store in `vectorstore/`).
8. Start app: `streamlit run app.py`

## Export PDF
After a chat session, click "Export Session" to generate a PDF under `exports/`.

## Security Notes
- Do not commit real API keys.
- Consider using Azure OpenAI or self-hosted models if sensitive data.
- Add role-based auth beyond simple token for production.

## Future Enhancements
- Add structured metadata filtering (jurisdiction, effective date).
- Support multi-model fallback & local LLMs.
- Add monitoring of unanswered query types.

## License
Internal hackathon prototype. Improve before production use.
