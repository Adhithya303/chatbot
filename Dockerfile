FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY data/echap12.pdf data/tab8.10.pdf data/tab8.12.pdf ./data/

# Generate data/processed/corpus.json and pre-compute FastEmbed embeddings at image build time
RUN python -m backend.ingest && python -c "from backend.retrieval import get_retriever; get_retriever()"

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

