**Title:** German Payroll Law RAG Assistant

**Goal:**
Build an end-to-end Retrieval-Augmented Generation (RAG) application that allows users to ask questions about German payroll law and receive accurate, context-based answers using PDF documents of relevant laws as the knowledge base.

**Requirements:**

1. **Data Ingestion**

   * Parse and chunk PDF documents of German payroll law.
   * Store embeddings in a vector database (e.g., ChromaDB, FAISS).
   * Automate ingestion with a Python script.

2. **Retrieval Flow**

   * Implement hybrid retrieval (vector search + keyword).
   * Apply document re-ranking to improve relevance.
   * Build prompts dynamically from retrieved chunks before sending to the LLM.

3. **LLM Integration**

   * Use OpenAI as the LLM.
   * Compare multiple prompt strategies and evaluate which yields the best legal Q\&A results.

4. **Evaluation**

   * Implement evaluation pipeline to test retrieval precision and answer quality.
   * Compare different chunking sizes and retrieval strategies.

5. **Interface**

   * Provide a **Streamlit** UI where users can upload questions and get answers.
   * Display retrieved document context alongside answers.

6. **Monitoring**

   * Collect user feedback on answer quality (thumbs up/down).
   * Build a simple monitoring dashboard (e.g., Streamlit) with at least 5 metrics (queries per day, latency, retrieval accuracy, user satisfaction, top queries).

7. **Containerization & Reproducibility**

   * Dockerize the app with `docker-compose` (app + vector DB).
   * Provide a `README.md` with setup instructions, dependencies, and usage guide.

**Deliverables:**

* Source code in a public GitHub repo.
* Documentation (`README.md`) with:

  * Problem description
  * Setup instructions
  * Example queries/answers with screenshots
  * Evaluation results
