# The Economic Surveyer (2025–26)

A grounded React + FastAPI + Capacitor Android research assistant for **Chapter 12: Employment and Skill Development: Getting Skilling Right** and Statistical Appendix **Table 8.10 (Labour Force Participation Rate)** and **Table 8.12 (Unemployment Rate)** of India’s Economic Survey 2025–26.

## Architecture

- **Ingestion (`backend/ingest.py`)**: Extracts paragraph-sized chapter chunks from `data/echap12.pdf` (45 pages) and reconstructs structured state/All-India rows and table metadata from `data/tab8.10.pdf` and `data/tab8.12.pdf` into `data/processed/corpus.json`.
- **Hybrid Retrieval (`backend/retrieval.py`)**: Combines in-memory `BM25Okapi` lexical search with cached local `FastEmbed` (`BAAI/bge-small-en-v1.5`) dense embeddings using Reciprocal Rank Fusion (RRF).
- **Grounded Answers (`backend/answer.py`)**: Resolves state/quarter numeric lookups and two-quarter comparisons deterministically in Python, and uses the Google Gen AI SDK (`google-genai`) with strict citation ID validation for explanatory questions.
- **Web & Android Frontend (`frontend/`)**: React + TypeScript + Vite reader bundled as an installable Android APK via Capacitor (`com.economicsurveyer.app`).

---

## Local Two-Terminal Development Setup (Windows PowerShell)

### Prerequisites
1. Install Python dependencies and build the processed corpus:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m backend.ingest
   ```
2. Configure `.env` in the repository root (copy from `.env.example`):
   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-3.8-flash
   ```

### Terminal 1 — Start the FastAPI Backend (Port 8000)
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Terminal 2 — Start the React Frontend (Port 5173)
```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Deploying the HTTPS FastAPI Backend

For the Android APK to work on any phone without your laptop running, deploy the FastAPI backend to an HTTPS host (such as Render, Railway, Fly.io, or Google Cloud Run).

1. **Deployment files included**:
   - [`render.yaml`](file:///c:/Users/adhit/Downloads/chatbot/render.yaml): Render Blueprint config.
   - [`Dockerfile`](file:///c:/Users/adhit/Downloads/chatbot/Dockerfile): Container build that runs `python -m backend.ingest` and pre-warms `FastEmbed` during image build.
   - [`Procfile`](file:///c:/Users/adhit/Downloads/chatbot/Procfile): Standard PaaS process definition.
2. **Host Build & Start Commands**:
   - **Build command**:
     ```bash
     pip install -r requirements.txt && python -m backend.ingest
     ```
   - **Start command**:
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
3. **Host Environment Variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API key (set in the host's secret settings; never commit to Git).
   - `GEMINI_MODEL`: `gemini-3.8-flash` (or your preferred Gemini text model).
4. **Verify the Deployed Backend**:
   - Check `GET https://<your-backend-host>/api/health` returns HTTP 200 with `{"status": "ok", "chapter_chunks": 245, "table_rows": 46}`.

---

## Building the Android APK (Capacitor)

### 1. Configure the Production Backend Origin
Create `frontend/.env.production.local` (copied from [`frontend/.env.example`](file:///c:/Users/adhit/Downloads/chatbot/frontend/.env.example)) with your deployed HTTPS backend URL (no trailing slash):

```env
VITE_API_BASE_URL=https://your-backend-service.onrender.com
```

> **Note:** Production and Android builds enforce a configuration check in [`frontend/src/api.ts`](file:///c:/Users/adhit/Downloads/chatbot/frontend/src/api.ts) so the APK will never silently fall back to the local Vite `/api` development proxy.

### 2. Rebuild and Package the APK
From the project root in Windows PowerShell:

```powershell
cd frontend
npm run lint
npm run build
npx cap sync android
cd android
.\gradlew.bat assembleDebug
cd ..\..
Copy-Item frontend\android\app\build\outputs\apk\debug\app-debug.apk .\The-Economic-Surveyer.apk -Force
```

### 3. APK Output Paths
- **Gradle Debug APK**: `frontend/android/app/build/outputs/apk/debug/app-debug.apk`
- **Root Copy**: `The-Economic-Surveyer.apk`

---

## Demo & Mobile Verification Questions

1. **Two-Quarter State Comparison (Table 8.12)**
   > How did Tamil Nadu’s unemployment rate change between April–June and July–September 2025?
2. **Single-Quarter State Lookup (Table 8.10)**
   > What was Tamil Nadu’s labour force participation rate in July–September 2025?
3. **Conceptual Explanation with Grounded Citations (Chapter 12 & Table Notes)**
   > What is the difference between unemployment rate and labour force participation rate?
