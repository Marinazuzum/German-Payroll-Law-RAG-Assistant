import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid retrieval system combining vector search, keyword search, and re-ranking."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(allow_reset=True)
        )
        
        try:
            self.collection = self.chroma_client.get_collection("german_payroll_law")
        except Exception as e:
            logger.error(f"Could not load collection: {e}")
            self.collection = None
        
        # Initialize re-ranker
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # German language processing
        self.stemmer = SnowballStemmer('german')
        self.german_stopwords = set(stopwords.words('german'))
        
        # Initialize BM25 and TF-IDF for keyword search
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.documents_cache = []
        self.metadata_cache = []
        
        # Load documents for keyword search
        self._initialize_keyword_search()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for keyword search."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', ' ', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text, language='german')
        
        # Remove stopwords and stem
        tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.german_stopwords and len(token) > 2
        ]
        
        return tokens
    
    def _initialize_keyword_search(self):
        """Initialize BM25 and TF-IDF for keyword search."""
        if not self.collection:
            logger.error("No collection available for keyword search initialization")
            return
        
        try:
            # Get all documents from ChromaDB
            all_docs = self.collection.get()
            
            if not all_docs['documents']:
                logger.warning("No documents found in collection")
                return
            
            self.documents_cache = all_docs['documents']
            self.metadata_cache = all_docs['metadatas']
            
            # Preprocess documents for BM25
            tokenized_docs = [
                self._preprocess_text(doc) for doc in self.documents_cache
            ]
            
            # Initialize BM25
            self.bm25 = BM25Okapi(tokenized_docs)
            
            # Initialize TF-IDF
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=list(self.german_stopwords),
                ngram_range=(1, 2),
                lowercase=True
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents_cache)
            
            logger.info(f"Initialized keyword search with {len(self.documents_cache)} documents")
            
        except Exception as e:
            logger.error(f"Error initializing keyword search: {e}")
    
    def vector_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        if not self.collection:
            return []
        
        top_k = top_k or settings.top_k_retrieval
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            documents = []
            if results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    documents.append({
                        'content': doc,
                        'metadata': metadata,
                        'score': 1 - distance,  # Convert distance to similarity score
                        'rank': i + 1,
                        'search_type': 'vector'
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def keyword_search_bm25(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        if not self.bm25 or not self.documents_cache:
            return []
        
        top_k = top_k or settings.top_k_retrieval
        
        try:
            # Preprocess query
            query_tokens = self._preprocess_text(query)
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k documents
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            documents = []
            for rank, idx in enumerate(top_indices):
                if scores[idx] > 0:  # Only include documents with positive scores
                    documents.append({
                        'content': self.documents_cache[idx],
                        'metadata': self.metadata_cache[idx],
                        'score': float(scores[idx]),
                        'rank': rank + 1,
                        'search_type': 'bm25'
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def keyword_search_tfidf(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform TF-IDF keyword search."""
        if not self.tfidf_vectorizer or not hasattr(self, 'tfidf_matrix'):
            return []
        
        top_k = top_k or settings.top_k_retrieval
        
        try:
            # Vectorize query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top-k documents
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            documents = []
            for rank, idx in enumerate(top_indices):
                if similarities[idx] > 0:  # Only include documents with positive scores
                    documents.append({
                        'content': self.documents_cache[idx],
                        'metadata': self.metadata_cache[idx],
                        'score': float(similarities[idx]),
                        'rank': rank + 1,
                        'search_type': 'tfidf'
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return []
    
    def hybrid_search(self, 
                     query: str, 
                     top_k: int = None,
                     vector_weight: float = 0.5,
                     bm25_weight: float = 0.3,
                     tfidf_weight: float = 0.2) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector, BM25, and TF-IDF search."""
        top_k = top_k or settings.top_k_retrieval
        
        # Perform individual searches
        vector_results = self.vector_search(query, top_k * 2)
        bm25_results = self.keyword_search_bm25(query, top_k * 2)
        tfidf_results = self.keyword_search_tfidf(query, top_k * 2)
        
        # Combine results with weighted scores
        combined_results = {}
        
        # Add vector search results
        for doc in vector_results:
            doc_id = doc['content'][:100]  # Use first 100 chars as ID
            if doc_id not in combined_results:
                combined_results[doc_id] = doc.copy()
                combined_results[doc_id]['combined_score'] = doc['score'] * vector_weight
                combined_results[doc_id]['search_types'] = ['vector']
            else:
                combined_results[doc_id]['combined_score'] += doc['score'] * vector_weight
                combined_results[doc_id]['search_types'].append('vector')
        
        # Add BM25 results
        for doc in bm25_results:
            doc_id = doc['content'][:100]
            if doc_id not in combined_results:
                combined_results[doc_id] = doc.copy()
                combined_results[doc_id]['combined_score'] = doc['score'] * bm25_weight
                combined_results[doc_id]['search_types'] = ['bm25']
            else:
                combined_results[doc_id]['combined_score'] += doc['score'] * bm25_weight
                combined_results[doc_id]['search_types'].append('bm25')
        
        # Add TF-IDF results
        for doc in tfidf_results:
            doc_id = doc['content'][:100]
            if doc_id not in combined_results:
                combined_results[doc_id] = doc.copy()
                combined_results[doc_id]['combined_score'] = doc['score'] * tfidf_weight
                combined_results[doc_id]['search_types'] = ['tfidf']
            else:
                combined_results[doc_id]['combined_score'] += doc['score'] * tfidf_weight
                combined_results[doc_id]['search_types'].append('tfidf')
        
        # Sort by combined score and return top-k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """Re-rank documents using cross-encoder model."""
        if not documents:
            return documents
        
        top_k = top_k or settings.rerank_top_k
        
        try:
            # Prepare query-document pairs for re-ranking
            pairs = [(query, doc['content']) for doc in documents]
            
            # Get re-ranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update documents with rerank scores
            for doc, score in zip(documents, rerank_scores):
                doc['rerank_score'] = float(score)
            
            # Sort by rerank score and return top-k
            reranked_docs = sorted(
                documents,
                key=lambda x: x['rerank_score'],
                reverse=True
            )
            
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in re-ranking: {e}")
            # Return original documents if re-ranking fails
            return documents[:top_k]
    
    def retrieve(self, 
                query: str, 
                use_hybrid: bool = True,
                use_reranking: bool = True,
                top_k: int = None,
                rerank_top_k: int = None) -> List[Dict[str, Any]]:
        """Main retrieval method."""
        top_k = top_k or settings.top_k_retrieval
        rerank_top_k = rerank_top_k or settings.rerank_top_k
        
        logger.info(f"Retrieving documents for query: {query[:100]}...")
        
        # Perform search
        if use_hybrid:
            documents = self.hybrid_search(query, top_k)
        else:
            documents = self.vector_search(query, top_k)
        
        if not documents:
            logger.warning("No documents retrieved")
            return []
        
        logger.info(f"Retrieved {len(documents)} documents before re-ranking")
        
        # Apply re-ranking if requested
        if use_reranking and len(documents) > 1:
            documents = self.rerank_documents(query, documents, rerank_top_k)
            logger.info(f"Re-ranked to {len(documents)} documents")
        
        # Add final ranks
        for i, doc in enumerate(documents):
            doc['final_rank'] = i + 1
        
        return documents
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        stats = {
            "collection_available": self.collection is not None,
            "bm25_available": self.bm25 is not None,
            "tfidf_available": self.tfidf_vectorizer is not None,
            "documents_cached": len(self.documents_cache),
            "embedding_model": settings.embedding_model,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }
        
        if self.collection:
            try:
                stats["collection_count"] = self.collection.count()
            except:
                stats["collection_count"] = "unknown"
        
        return stats

def main():
    """Test the hybrid retriever."""
    retriever = HybridRetriever()
    
    # Print stats
    stats = retriever.get_retrieval_stats()
    print("Retrieval System Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test query
    test_query = "Wie wird das Gehalt in Deutschland besteuert?"
    print(f"\nTest Query: {test_query}")
    
    results = retriever.retrieve(test_query)
    print(f"Retrieved {len(results)} documents")
    
    for i, doc in enumerate(results[:3]):
        print(f"\nDocument {i+1}:")
        print(f"  Content preview: {doc['content'][:200]}...")
        print(f"  Combined score: {doc.get('combined_score', 'N/A')}")
        print(f"  Rerank score: {doc.get('rerank_score', 'N/A')}")
        print(f"  Search types: {doc.get('search_types', 'N/A')}")

if __name__ == "__main__":
    main()
