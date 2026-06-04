# Self-Healing RAG Pipeline

A Retrieval-Augmented Generation system that **critiques its own output** and retries with reformulated queries when the answer quality is insufficient.

## Architecture

```
User Query
    |
    v
+--------------+
|   Retrieve   |  <-- Chroma vector store (5 chunks)
+------+-------+
       v
+--------------+
|   Generate   |  <-- Groq Llama 3.3 70B
+------+-------+
       v
+--------------+
|   Critique   |  <-- Faithfulness, Relevance, Completeness
+------+-------+
       |
   +---+---+
   |       |
 PASS    FAIL
   |       |
   v       v
 Return   Reformulate --> back to Retrieve (max 3 retries)
 Answer       |
              v (after max retries)
   "I don't have enough information"
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Vector Store | ChromaDB (local) |
| Orchestration | LangGraph |
| Frontend | Streamlit |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

The `.env` file is pre-configured with your Groq API key. To change it:

```bash
# Edit .env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Run the app

```bash
streamlit run app.py
```

## Usage

1. **Upload documents** — Use the sidebar to upload PDF or TXT files
2. **Click "Ingest Documents"** — Documents are chunked, embedded, and stored in Chroma
3. **Ask a question** — Type your question and click "Run Pipeline"
4. **Watch it self-heal** — The system critiques its own answer and retries if needed

## How It Works

1. **Retrieve**: Fetches the 5 most relevant chunks from the vector store
2. **Generate**: LLM produces an answer grounded in the retrieved context
3. **Critique**: A separate LLM call evaluates the answer on:
   - Faithfulness (0-1): Is it grounded in the context?
   - Relevance (0-1): Does it answer the question?
   - Completeness (0-1): Is it thorough enough?
4. **Decision**:
   - If all scores pass thresholds (>= 0.6/0.6/0.5): return the answer
   - If fail: the critique agent provides a reformulated query
5. **Retry**: The system retrieves with the new query and repeats (max 3 retries)

## Project Structure

```
.
├── app.py              # Streamlit UI
├── pipeline.py         # LangGraph self-healing RAG graph
├── ingest.py           # Document ingestion & embedding
├── requirements.txt    # Python dependencies
├── .env                # API key configuration
├── data/               # Drop files here (optional)
└── README.md
```

## License

MIT
