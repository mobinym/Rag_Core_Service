

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- A running instance of Ollama

### 2. Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd rag_core_service
````

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configuration

Create a `config.yml` file in the root directory:

```yaml
paths:
  index_dir: "data/indices"

services:
  document_processor_url: "http://<ip>:<port>/your/chunking/endpoint/"
  embedding_service_url: "http://<ip>:<port>/your/embedding/endpoint/"
  ollama_base_url: "http://host.docker.internal:1134"

llm:
  model_name: "gemma3:4b"

defaults:
  extractor_strategy: "pypdf"
  chunker_strategy: "recursive"
  retrieval_strategy: "adaptive"
  top_k: 5

retriever_settings:
  adaptive:
    min_answer_length: 150
    retry_k: 7
```

### 4. Run with Docker

```bash
docker-compose up --build -d
```

Access services:

* **RAG UI:** [http://localhost:8000](http://localhost:8000)
* **Grafana:** [http://localhost:3000](http://localhost:3000) (admin/admin)
* **Prometheus:** [http://localhost:9090](http://localhost:9090)



### 3. Configuration

Create a `config.yml` file in the root directory:

```yaml
paths:
  index_dir: "data/indices"

services:
  document_processor_url: "http://<ip>:<port>/your/chunking/endpoint/"
  embedding_service_url: "http://<ip>:<port>/your/embedding/endpoint/"
  ollama_base_url: "http://host.docker.internal:1134"

llm:
  model_name: "gemma3:4b"

defaults:
  extractor_strategy: "pypdf"
  chunker_strategy: "recursive"
  retrieval_strategy: "adaptive"
  top_k: 5

retriever_settings:
  adaptive:
    min_answer_length: 150
    retry_k: 7
```

### 4. Run with Docker

```bash
docker-compose up --build -d
```

Access services:

* **RAG UI:** [http://localhost:8000](http://localhost:8000)
* **Grafana:** [http://localhost:3000](http://localhost:3000) (admin/admin)
* **Prometheus:** [http://localhost:9090](http://localhost:9090)

---

## 💻 How to Use

### Web UI

1. Visit `http://localhost:8000`
2. Upload a document → get a Session ID
3. Paste the Session ID, type your question → receive answer

### API Usage

#### Create a Session

```bash
curl -X 'POST' \
  'http://localhost:8000/sessions' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/document.pdf' \
  -F 'vector_store_strategy=faiss'
```

#### Ask a Question

```bash
curl -X 'POST' \
  'http://localhost:8000/sessions/YOUR_SESSION_ID/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the main topic of this document?",
    "retrieval_strategy": "adaptive"
  }'
```

---

## ⚙️ API Error Codes

| Error Code | HTTP Status | Description                                 |
| ---------- | ----------- | ------------------------------------------- |
| `30001`    | 400         | Unsupported strategy specified              |
| `30002`    | 404         | Session ID not found                        |
| `30003`    | 400         | No usable text chunks extracted             |
| `30004`    | 404 / 500   | Session corrupted or vector store not found |
| `40001`    | 503         | External microservice unavailable           |
| `40002`    | 500         | Error saving vector index                   |
| `40003`    | 500         | Error loading vector index                  |
| `40004`    | 500         | Retrieval/LLM generation failed             |
| `90001`    | 422         | Validation failed                           |
| `99999`    | 500         | Unknown internal server error               |

