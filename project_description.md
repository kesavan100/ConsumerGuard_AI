# ConsumerGuard AI

**ConsumerGuard AI** is an intelligent, locally-assisted educational platform designed to help Indian e-commerce consumers understand their rights, navigate platform policies, and resolve grievances. It acts as an autonomous multi-agent pipeline that analyzes user complaints, retrieves relevant legal and platform documents, and synthesizes clear, actionable guidance using a Large Language Model (LLM).

The application is built with a focus on speed, low API costs, and strict factual grounding (to avoid LLM hallucinations). It leverages a fast **Retrieval-Augmented Generation (RAG)** architecture using a local embedding model and the high-speed Gemini Flash model.

---

## 🏗️ Architecture & Workflow

When a user submits a complaint via the Streamlit UI, the request passes through a multi-agent orchestration pipeline (`pipeline.py`). The workflow consists of four main steps:

1. **Complaint Analysis (Agent 1):** Extracts key details (Platform, Issue Type, Product Category, Timeline) using fast, rule-based heuristics without calling an LLM.
2. **Context Retrieval (Agent 2):** Generates a vector embedding of the complaint and searches the local vector database for the top 3 most relevant policy chunks. It filters documents by platform (Amazon/Flipkart) and includes generic legal documents (CP Act 2019).
3. **Confidence Scoring:** The pipeline calculates a confidence score (HIGH/MEDIUM/LOW) based on the cosine similarity of the retrieved chunks.
4. **Resolution Generation (Agent 3):** 
   - If confidence is `HIGH`, the pipeline uses a fast-path, sending only the top 2 chunks and a restricted token limit to the LLM for maximum speed.
   - If `MEDIUM` or `LOW`, it sends the standard chunk set.
   - The Gemini model synthesizes the final response, strictly adhering to a JSON format that outputs lists of rules, consumer rights, and recommended actions.

---

## 🧩 Core Components

### 1. User Interface (`app.py`)
- Built using **Streamlit**.
- Provides a clean, academic, single-page layout for users to enter their complaints.
- Implements `@st.cache_resource` for the embedding model and vector database, avoiding cold starts and heavily reducing latency.
- Dynamically parses the JSON output from the backend and renders actionable bullet points (Complaint Summary, Relevant Policies, Consumer Rules, Rights, Actions).

### 2. Pipeline Orchestrator (`consumerguard/pipeline.py`)
- The central nerve center of the application.
- Receives the raw complaint string and pipes it sequentially through Agent 1 (Analysis), Agent 2 (Retrieval), and Agent 3 (Generation).
- Implements dynamic context scaling: reducing the context window and `max_tokens` when the retrieval confidence is high to save costs and reduce response latency to under 5 seconds.

### 3. The Agents (`consumerguard/agents.py`)
- **Agent 1 (Analyzer):** A rule-based parser that uses keyword matching to detect the platform, product category, and issue type. It is extremely fast and costs nothing to run.
- **Agent 3 (Resolver):** The LLM integration (`google-generativeai`). It uses a highly specific prompt containing hardcoded menus of the Consumer Protection Act and E-Commerce Rules. It is strictly instructed to return a JSON object populated with lists of rights and rules, forbidding internal reasoning or chain-of-thought leakage.

### 4. Retrieval System (`consumerguard/retriever.py` & `consumerguard/embedder.py`)
- **Embedder:** Uses `sentence-transformers/all-MiniLM-L6-v2` running locally on the CPU to generate 384-dimensional vector embeddings. 
- **Retriever:** Executes a cosine similarity search against the VectorStore. It filters the metadata (e.g., if the user mentions "Amazon", it will only search Amazon policies + general laws). It returns the `TOP_K` (3) chunks.

### 5. Knowledge Base & VectorStore (`consumerguard/ingest.py` & `consumerguard/vectorstore.py`)
- **VectorStore:** A custom, lightweight numpy-based vector database (persisted to `chroma_db/`) that stores embeddings, raw text chunks, and metadata (source document, chunk index, platform).
- **Ingestion Script (`ingest.py`):** An offline script that reads the 9 source PDFs in the `product_data/` directory using `pypdf`. It splits the complex legal text into overlapping windows (`chunk_size=700`, `overlap=100`), generates embeddings in batch, and populates the database. 

### 6. Configuration (`consumerguard/config.py`)
- The central brain for hyperparameter tuning.
- Stores variables such as the `GEMINI_MODEL` (currently `gemini-3.5-flash`), chunk sizes, retrieval bounds (`TOP_K`), similarity thresholds, and file paths.

---

## 🛠️ Technology Stack
- **Frontend:** Streamlit
- **LLM Integration:** Google Generative AI Python SDK (`gemini-3.5-flash`)
- **Local Embeddings:** HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) via PyTorch
- **Vector Search:** Custom Numpy-based VectorStore
- **PDF Processing:** `pypdf`
