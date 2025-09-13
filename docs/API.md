# API Documentation

This document describes the programmatic interfaces for the German Payroll Law RAG Assistant.

## 🏗️ Core Components

### HybridRetriever

The main retrieval component that combines vector search, keyword search, and re-ranking.

```python
from src.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()
```

#### Methods

##### `retrieve(query, use_hybrid=True, use_reranking=True, top_k=5)`
Main retrieval method that returns relevant documents.

**Parameters:**
- `query` (str): The search query
- `use_hybrid` (bool): Whether to use hybrid search (default: True)
- `use_reranking` (bool): Whether to apply re-ranking (default: True)
- `top_k` (int): Number of documents to return (default: 5)

**Returns:**
- List[Dict]: Retrieved documents with metadata and scores

**Example:**
```python
documents = retriever.retrieve(
    query="Wie werden Überstunden versteuert?",
    use_hybrid=True,
    use_reranking=True,
    top_k=3
)

for doc in documents:
    print(f"Score: {doc['combined_score']:.3f}")
    print(f"Content: {doc['content'][:200]}...")
    print(f"Source: {doc['metadata']['file_name']}")
```

##### `vector_search(query, top_k=5)`
Perform semantic vector search only.

**Parameters:**
- `query` (str): The search query
- `top_k` (int): Number of documents to return

**Returns:**
- List[Dict]: Documents with vector similarity scores

##### `hybrid_search(query, top_k=5, vector_weight=0.5, bm25_weight=0.3, tfidf_weight=0.2)`
Perform hybrid search combining multiple strategies.

**Parameters:**
- `query` (str): The search query
- `top_k` (int): Number of documents to return
- `vector_weight` (float): Weight for vector search
- `bm25_weight` (float): Weight for BM25 search
- `tfidf_weight` (float): Weight for TF-IDF search

**Returns:**
- List[Dict]: Documents with combined scores

##### `get_retrieval_stats()`
Get statistics about the retrieval system.

**Returns:**
- Dict: System statistics including collection count, model info, etc.

### LLMClient

Client for interacting with OpenAI LLM with different prompt strategies.

```python
from src.llm.llm_client import LLMClient
from src.llm.prompt_strategies import PromptStrategy

client = LLMClient()
```

#### Methods

##### `generate_answer(query, retrieved_documents, strategy=PromptStrategy.STRUCTURED, **kwargs)`
Generate an answer using the specified prompt strategy.

**Parameters:**
- `query` (str): User's question
- `retrieved_documents` (List[Dict]): Documents from retrieval
- `strategy` (PromptStrategy): Prompt strategy to use
- `include_sources` (bool): Include source information (default: True)
- `temperature` (float): LLM temperature override
- `max_tokens` (int): Maximum tokens override

**Returns:**
- Dict: Answer result with metadata

**Example:**
```python
result = client.generate_answer(
    query="Was sind Sozialversicherungsbeiträge?",
    retrieved_documents=documents,
    strategy=PromptStrategy.LEGAL_EXPERT,
    temperature=0.1
)

if result['success']:
    print(f"Answer: {result['answer']}")
    print(f"Tokens used: {result['usage']['total_tokens']}")
    print(f"Response time: {result['response_time']:.2f}s")
else:
    print(f"Error: {result['error']}")
```

##### `compare_strategies(query, retrieved_documents, strategies=None)`
Compare multiple prompt strategies for the same query.

**Parameters:**
- `query` (str): User's question
- `retrieved_documents` (List[Dict]): Documents from retrieval
- `strategies` (List[PromptStrategy]): Strategies to compare

**Returns:**
- Dict: Comparison results for each strategy

##### `get_usage_stats()`
Get current usage statistics.

**Returns:**
- Dict: Usage statistics including token counts, costs, etc.

### PDFProcessor

Handles PDF document processing and storage in vector database.

```python
from src.ingestion.pdf_processor import PDFProcessor

processor = PDFProcessor()
```

#### Methods

##### `process_pdf(pdf_path)`
Process a single PDF file.

**Parameters:**
- `pdf_path` (str): Path to PDF file

**Returns:**
- Dict: Processing result with status and metadata

##### `process_directory(directory_path)`
Process all PDF files in a directory.

**Parameters:**
- `directory_path` (str): Path to directory containing PDFs

**Returns:**
- Dict: Summary of processing results

##### `get_collection_stats()`
Get statistics about the document collection.

**Returns:**
- Dict: Collection statistics

### MetricsCollector

Collects and manages application metrics.

```python
from src.monitoring.metrics_collector import MetricsCollector

collector = MetricsCollector()
```

#### Methods

##### `record_query(query, response_time, tokens_used, retrieved_docs_count, strategy, success=True)`
Record a query and its metrics.

##### `record_feedback(query, feedback_type, timestamp=None, comments=None)`
Record user feedback.

##### `get_metrics(days_back=7)`
Get comprehensive metrics for the specified time period.

##### `export_metrics(format_type='json')`
Export metrics in JSON or CSV format.

## 🎯 Prompt Strategies

Available prompt strategies in `PromptStrategy` enum:

### BASIC
Simple, direct answers based on documents.

### STRUCTURED
Organized responses with clear sections:
- Summary
- Detailed explanation
- Legal foundations
- Practical notes

### LEGAL_EXPERT
Expert persona with professional tone and comprehensive analysis.

### STEP_BY_STEP
Systematic step-by-step analysis:
1. Problem identification
2. Legal classification
3. Document analysis
4. Law application
5. Conclusion
6. Implementation

### COMPARATIVE
Multi-perspective analysis comparing different aspects.

## 📊 Data Structures

### Document Structure
```python
{
    'content': str,           # Document text content
    'metadata': {
        'source': str,        # Original file path
        'file_name': str,     # File name
        'chunk_id': int,      # Chunk identifier
        'total_chunks': int,  # Total chunks in document
        'chunk_size': int,    # Size of this chunk
        'document_hash': str  # Document hash for tracking
    },
    'score': float,           # Relevance score
    'combined_score': float,  # Combined score (for hybrid search)
    'rerank_score': float,    # Re-ranking score
    'search_type': str,       # Type of search used
    'final_rank': int         # Final ranking position
}
```

### Answer Result Structure
```python
{
    'answer': str,           # Generated answer
    'strategy': str,         # Prompt strategy used
    'query': str,           # Original query
    'retrieved_documents': int,  # Number of documents used
    'response_time': float,  # Response time in seconds
    'usage': {
        'prompt_tokens': int,
        'completion_tokens': int,
        'total_tokens': int
    },
    'model': str,           # LLM model used
    'temperature': float,   # Temperature setting
    'success': bool,        # Success status
    'timestamp': str,       # ISO timestamp
    'sources': List[Dict]   # Source information (if included)
}
```

### Metrics Structure
```python
{
    'total_queries': int,
    'successful_queries': int,
    'success_rate': float,
    'avg_response_time': float,
    'avg_tokens': float,
    'queries_today': int,
    'satisfaction_rate': float,
    'most_used_strategy': str,
    'strategy_distribution': Dict[str, int],
    'peak_hour': int,
    'system_health_score': float,
    'recent_queries': List[Dict],
    'error_rate': float
}
```

## 🔧 Configuration API

### Settings
Access configuration through the `settings` object:

```python
from config import settings

# API settings
api_key = settings.openai_api_key
model = settings.llm_model

# Retrieval settings
chunk_size = settings.chunk_size
top_k = settings.top_k_retrieval

# Paths
pdf_path = settings.pdf_data_path
chroma_path = settings.chroma_persist_directory
```

### Environment Variables
Override settings using environment variables:

```bash
export CHUNK_SIZE=1500
export TOP_K_RETRIEVAL=7
export TEMPERATURE=0.2
```

## 🧪 Evaluation API

### RAGEvaluator

Comprehensive evaluation of the RAG system.

```python
from src.evaluation.metrics import RAGEvaluator

evaluator = RAGEvaluator()
```

#### Methods

##### `evaluate_end_to_end(query, retrieved_docs, answer, ground_truth)`
Evaluate complete RAG pipeline.

**Parameters:**
- `query` (str): Original query
- `retrieved_docs` (List[Dict]): Retrieved documents
- `answer` (str): Generated answer
- `ground_truth` (Dict): Expected results

**Returns:**
- Dict: Comprehensive evaluation metrics

##### `evaluate_retrieval(query, retrieved_docs, ground_truth)`
Evaluate retrieval performance only.

##### `evaluate_answer(answer, query, ground_truth, source_documents=None)`
Evaluate answer quality only.

### ExperimentRunner

Run systematic experiments to compare configurations.

```python
from src.evaluation.experiment_runner import ExperimentRunner

runner = ExperimentRunner()
```

#### Methods

##### `run_comprehensive_evaluation(test_dataset, retrieval_strategies=None, prompt_strategies=None)`
Run complete evaluation comparing multiple configurations.

##### `run_retrieval_strategy_experiment(queries, strategies=None)`
Compare different retrieval strategies.

##### `run_prompt_strategy_experiment(queries, strategies=None)`
Compare different prompt strategies.

## 🔌 Integration Examples

### Flask API Integration
```python
from flask import Flask, request, jsonify
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.llm_client import LLMClient

app = Flask(__name__)
retriever = HybridRetriever()
llm_client = LLMClient()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data['query']
    
    # Retrieve documents
    docs = retriever.retrieve(query)
    
    # Generate answer
    result = llm_client.generate_answer(query, docs)
    
    return jsonify(result)
```

### Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_queries_async(queries):
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(
                executor, 
                process_single_query, 
                query
            ) for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
    
    return results

def process_single_query(query):
    docs = retriever.retrieve(query)
    return llm_client.generate_answer(query, docs)
```

### Batch Processing
```python
def process_queries_batch(queries, batch_size=5):
    """Process queries in batches to manage API rate limits."""
    
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batch_results = []
        
        for query in batch:
            docs = retriever.retrieve(query)
            result = llm_client.generate_answer(query, docs)
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Rate limiting pause
        time.sleep(1)
    
    return results
```

## 🚨 Error Handling

### Common Exceptions

```python
from src.retrieval.hybrid_retriever import HybridRetriever

try:
    retriever = HybridRetriever()
    docs = retriever.retrieve("test query")
except Exception as e:
    print(f"Retrieval error: {e}")

# LLM Client errors
try:
    result = llm_client.generate_answer(query, docs)
except openai.RateLimitError:
    print("API rate limit exceeded")
except openai.AuthenticationError:
    print("Invalid API key")
except Exception as e:
    print(f"LLM error: {e}")
```

### Best Practices

1. **Always check result success status**
2. **Implement retry logic for API calls**
3. **Handle rate limiting gracefully**
4. **Validate inputs before processing**
5. **Log errors for debugging**

## 📈 Performance Optimization

### Caching
```python
import functools
from typing import List, Dict

@functools.lru_cache(maxsize=100)
def cached_retrieve(query: str) -> List[Dict]:
    """Cache retrieval results for common queries."""
    return retriever.retrieve(query)
```

### Batch Operations
```python
def process_documents_batch(pdf_paths: List[str], batch_size: int = 3):
    """Process PDFs in batches to manage memory usage."""
    
    for i in range(0, len(pdf_paths), batch_size):
        batch = pdf_paths[i:i + batch_size]
        
        for pdf_path in batch:
            processor.process_pdf(pdf_path)
        
        # Memory cleanup
        import gc
        gc.collect()
```

This API documentation provides comprehensive coverage of all programmatic interfaces. For implementation examples and integration patterns, refer to the example scripts in the `scripts/` directory.
