<h1 align="center">Self-Healing RAG Pipeline</h1>

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-orchestration-blue?logo=data:image/svg+xml" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Groq-LLM-purple?logo=fastapi&logoColor=white" alt="Groq"/>
  <img src="https://img.shields.io/badge/Chroma-DB-green?logo=database&logoColor=white" alt="Chroma"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Python-3.11+-yellow?logo=python&logoColor=white" alt="Python"/>
</p>

<p align="center">
  <em>A RAG system that doesn't just retrieve and generate — it <strong>critiques its own output</strong> and retries with reformulated queries when the answer isn't good enough.</em>
</p>

---

## How It Works

```
   USER ASKS A QUESTION
          |
          v
  +---------------+        +----------------+
  |   RETRIEVE    | <----- |  REFORMULATE   |
  |  (Chroma DB)  |        |  (New Query)   |
  +-------+-------+        +----------------+
          |
          v
  +---------------+
  |   GENERATE    |
  | (Llama 3.3)   |
  +-------+-------+
          |
          v
  +---------------+
  |   CRITIQUE    | -----> FAIL + retries < 3?
  | (LLM Judge)   |              |
  +---------------+         YES  |  NO
                              |   |
                              v   v
                          RETRY   RETURN ANSWER
                                  (or "I don't
                                   have enough
                                   info")
```

<details>
<summary><strong>Critique Criteria (click to expand)</strong></summary>

| Metric | Threshold | What it measures |
|--------|-----------|------------------|
| Faithfulness | >= 0.6 | Is the answer grounded in the retrieved context? |
| Relevance | >= 0.6 | Does it actually answer the user's question? |
| Completeness | >= 0.5 | Is the answer thorough enough? |

If any metric falls below its threshold, the system **fails** the answer and generates a reformulated query to retry retrieval.

</details>

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **LLM** | Groq `llama-3.3-70b-versatile` | Free, fast inference |
| **Embeddings** | `all-MiniLM-L6-v2` | Local, zero cost, no API needed |
| **Vector Store** | ChromaDB | Lightweight, persists to disk |
| **Orchestration** | LangGraph | Explicit state + conditional routing |
| **Frontend** | Streamlit | Clean UI, fast prototyping |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/URTD14/self-healing-RAG-pipeline.git
cd self-healing-RAG-pipeline
pip install -r requirements.txt
```

### 2. Configure API key

Get a free Groq API key at [console.groq.com](https://console.groq.com), then set it in `.env`:

```bash
GROQ_API_KEY=your_key_here
```

### 3. Run

```bash
streamlit run app.py
```

---

## Usage

| Step | Action |
|------|--------|
| **1. Upload** | Use the sidebar to drop PDF or TXT files |
| **2. Ingest** | Click "Ingest Documents" — files are chunked, embedded, stored |
| **3. Ask** | Type your question and hit "Run Pipeline" |
| **4. Self-Heal** | Watch the system critique itself and retry if needed |

---

## Pipeline Walkthrough

When you ask a question, the system:

1. **Retrieves** the 5 most relevant chunks from Chroma
2. **Generates** an answer using Groq Llama 3.3 70B
3. **Critiques** the answer with a separate LLM call (faithfulness, relevance, completeness)
4. **Decides**: pass → return answer, fail → reformulate query → retry (max 3x)
5. **Returns** the best answer — or gracefully says "I don't have enough information"

---

## Project Structure

```
self-healing-RAG-pipeline/
├── app.py              # Streamlit UI
├── pipeline.py         # LangGraph self-healing RAG graph
├── ingest.py           # Document ingestion & embedding
├── requirements.txt    # Python dependencies
├── .env                # API key (gitignored)
├── data/               # Drop files here
└── README.md
```

---

## Example

```
Question: "What is retrieval-augmented generation?"

Attempt 1:
  Context: [3 chunks about RAG]
  Answer: "Retrieval-augmented generation (RAG) is a technique that enhances
           language models by..."
  Critique: faithfulness=0.92, relevance=0.88, completeness=0.85
  Verdict: PASS

Final Answer returned.
```

```
Question: "How does the attention mechanism work in transformers?"

Attempt 1:
  Critique: faithfulness=0.45, relevance=0.60, completeness=0.30
  Verdict: FAIL — "Answer too shallow, missing key-value explanation"
  Reformulated query: "transformer self-attention mechanism key value query"

Attempt 2:
  Critique: faithfulness=0.88, relevance=0.91, completeness=0.82
  Verdict: PASS

Final Answer returned after 1 retry.
```

---

## License

Distributed under the [MIT License](LICENSE).

<p align="center">
  <sub>Built with LangGraph + Groq + Chroma + Streamlit</sub>
</p>
