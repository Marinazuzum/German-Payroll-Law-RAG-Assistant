import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import logging
from collections import Counter
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Evaluation metrics for RAG system performance."""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
    
    # Retrieval Metrics
    
    def precision_at_k(self, retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
        """Calculate precision@k for retrieved documents."""
        if not retrieved_docs or k <= 0:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        relevant_count = 0
        
        for doc in top_k_docs:
            doc_content = doc.get('content', '')
            if any(rel_doc in doc_content for rel_doc in relevant_docs):
                relevant_count += 1
        
        return relevant_count / min(k, len(retrieved_docs))
    
    def recall_at_k(self, retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
        """Calculate recall@k for retrieved documents."""
        if not relevant_docs or not retrieved_docs or k <= 0:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        found_relevant = 0
        
        for rel_doc in relevant_docs:
            for doc in top_k_docs:
                if rel_doc in doc.get('content', ''):
                    found_relevant += 1
                    break
        
        return found_relevant / len(relevant_docs)
    
    def mean_reciprocal_rank(self, retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)."""
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        for i, doc in enumerate(retrieved_docs):
            doc_content = doc.get('content', '')
            if any(rel_doc in doc_content for rel_doc in relevant_docs):
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg_at_k(self, retrieved_docs: List[Dict], relevance_scores: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@k)."""
        if not retrieved_docs or not relevance_scores or k <= 0:
            return 0.0
        
        def dcg_at_k(scores: List[float], k: int) -> float:
            scores = scores[:k]
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        
        # Calculate DCG
        actual_scores = relevance_scores[:k]
        dcg = dcg_at_k(actual_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = dcg_at_k(ideal_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    # Answer Quality Metrics
    
    def semantic_similarity(self, answer: str, reference: str) -> float:
        """Calculate semantic similarity between answer and reference."""
        if not answer or not reference:
            return 0.0
        
        try:
            embeddings = self.embedding_model.encode([answer, reference])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def bleu_score(self, answer: str, reference: str) -> float:
        """Calculate BLEU score (simplified implementation)."""
        if not answer or not reference:
            return 0.0
        
        # Tokenize
        answer_tokens = answer.lower().split()
        reference_tokens = reference.lower().split()
        
        if not answer_tokens or not reference_tokens:
            return 0.0
        
        # Calculate n-gram precision for n=1,2,3,4
        precisions = []
        for n in range(1, 5):
            answer_ngrams = [tuple(answer_tokens[i:i+n]) for i in range(len(answer_tokens)-n+1)]
            reference_ngrams = [tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens)-n+1)]
            
            if not answer_ngrams:
                precisions.append(0.0)
                continue
            
            answer_counter = Counter(answer_ngrams)
            reference_counter = Counter(reference_ngrams)
            
            overlap = sum(min(answer_counter[ngram], reference_counter[ngram]) 
                         for ngram in answer_counter)
            precision = overlap / len(answer_ngrams)
            precisions.append(precision)
        
        # Calculate geometric mean
        if any(p == 0 for p in precisions):
            return 0.0
        
        geometric_mean = np.exp(np.mean(np.log(precisions)))
        
        # Brevity penalty
        answer_len = len(answer_tokens)
        reference_len = len(reference_tokens)
        
        if answer_len > reference_len:
            bp = 1.0
        else:
            bp = np.exp(1 - reference_len / answer_len) if answer_len > 0 else 0.0
        
        return geometric_mean * bp
    
    def rouge_l(self, answer: str, reference: str) -> float:
        """Calculate ROUGE-L score."""
        if not answer or not reference:
            return 0.0
        
        answer_tokens = answer.lower().split()
        reference_tokens = reference.lower().split()
        
        if not answer_tokens or not reference_tokens:
            return 0.0
        
        # Find longest common subsequence
        def lcs_length(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(answer_tokens, reference_tokens)
        
        if lcs_len == 0:
            return 0.0
        
        precision = lcs_len / len(answer_tokens)
        recall = lcs_len / len(reference_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def answer_relevance(self, answer: str, query: str) -> float:
        """Calculate how relevant the answer is to the query."""
        return self.semantic_similarity(answer, query)
    
    def answer_completeness(self, answer: str, expected_elements: List[str]) -> float:
        """Calculate how complete the answer is based on expected elements."""
        if not expected_elements:
            return 1.0
        
        answer_lower = answer.lower()
        found_elements = sum(1 for element in expected_elements 
                           if element.lower() in answer_lower)
        
        return found_elements / len(expected_elements)
    
    def factual_consistency(self, answer: str, source_documents: List[str]) -> float:
        """Check factual consistency with source documents (simplified)."""
        if not source_documents:
            return 0.0
        
        # Simple approach: check if answer claims are supported by sources
        answer_sentences = re.split(r'[.!?]+', answer)
        consistent_sentences = 0
        
        for sentence in answer_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if any source document supports this sentence
            for doc in source_documents:
                similarity = self.semantic_similarity(sentence, doc)
                if similarity > 0.7:  # Threshold for consistency
                    consistent_sentences += 1
                    break
        
        total_sentences = len([s for s in answer_sentences if s.strip()])
        return consistent_sentences / total_sentences if total_sentences > 0 else 0.0

class RAGEvaluator:
    """Main evaluator for RAG system performance."""
    
    def __init__(self):
        self.metrics = EvaluationMetrics()
    
    def evaluate_retrieval(self, 
                          query: str,
                          retrieved_docs: List[Dict],
                          ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        
        relevant_docs = ground_truth.get('relevant_documents', [])
        relevance_scores = ground_truth.get('relevance_scores', [])
        
        results = {
            'precision_at_1': self.metrics.precision_at_k(retrieved_docs, relevant_docs, 1),
            'precision_at_3': self.metrics.precision_at_k(retrieved_docs, relevant_docs, 3),
            'precision_at_5': self.metrics.precision_at_k(retrieved_docs, relevant_docs, 5),
            'recall_at_1': self.metrics.recall_at_k(retrieved_docs, relevant_docs, 1),
            'recall_at_3': self.metrics.recall_at_k(retrieved_docs, relevant_docs, 3),
            'recall_at_5': self.metrics.recall_at_k(retrieved_docs, relevant_docs, 5),
            'mrr': self.metrics.mean_reciprocal_rank(retrieved_docs, relevant_docs),
        }
        
        if relevance_scores:
            results.update({
                'ndcg_at_1': self.metrics.ndcg_at_k(retrieved_docs, relevance_scores, 1),
                'ndcg_at_3': self.metrics.ndcg_at_k(retrieved_docs, relevance_scores, 3),
                'ndcg_at_5': self.metrics.ndcg_at_k(retrieved_docs, relevance_scores, 5),
            })
        
        return results
    
    def evaluate_answer(self, 
                       answer: str,
                       query: str,
                       ground_truth: Dict[str, Any],
                       source_documents: List[str] = None) -> Dict[str, float]:
        """Evaluate answer quality."""
        
        reference_answer = ground_truth.get('reference_answer', '')
        expected_elements = ground_truth.get('expected_elements', [])
        
        results = {
            'semantic_similarity': self.metrics.semantic_similarity(answer, reference_answer),
            'answer_relevance': self.metrics.answer_relevance(answer, query),
            'answer_completeness': self.metrics.answer_completeness(answer, expected_elements)
        }
        
        if reference_answer:
            results.update({
                'bleu_score': self.metrics.bleu_score(answer, reference_answer),
                'rouge_l': self.metrics.rouge_l(answer, reference_answer)
            })
        
        if source_documents:
            results['factual_consistency'] = self.metrics.factual_consistency(answer, source_documents)
        
        return results
    
    def evaluate_end_to_end(self,
                           query: str,
                           retrieved_docs: List[Dict],
                           answer: str,
                           ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate complete RAG pipeline."""
        
        source_docs = [doc.get('content', '') for doc in retrieved_docs]
        
        retrieval_metrics = self.evaluate_retrieval(query, retrieved_docs, ground_truth)
        answer_metrics = self.evaluate_answer(query, answer, ground_truth, source_docs)
        
        # Calculate composite scores
        retrieval_score = np.mean([
            retrieval_metrics.get('precision_at_3', 0),
            retrieval_metrics.get('recall_at_3', 0),
            retrieval_metrics.get('mrr', 0)
        ])
        
        answer_score = np.mean([
            answer_metrics.get('semantic_similarity', 0),
            answer_metrics.get('answer_relevance', 0),
            answer_metrics.get('factual_consistency', 0) if 'factual_consistency' in answer_metrics else 0
        ])
        
        overall_score = (retrieval_score + answer_score) / 2
        
        return {
            'query': query,
            'retrieval_metrics': retrieval_metrics,
            'answer_metrics': answer_metrics,
            'composite_scores': {
                'retrieval_score': retrieval_score,
                'answer_score': answer_score,
                'overall_score': overall_score
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate RAG system on a dataset of queries."""
        
        all_results = []
        
        for item in dataset:
            query = item['query']
            retrieved_docs = item.get('retrieved_docs', [])
            answer = item.get('answer', '')
            ground_truth = item.get('ground_truth', {})
            
            result = self.evaluate_end_to_end(query, retrieved_docs, answer, ground_truth)
            all_results.append(result)
        
        # Calculate aggregate metrics
        retrieval_metrics = {}
        answer_metrics = {}
        
        if all_results:
            # Aggregate retrieval metrics
            for metric in all_results[0]['retrieval_metrics'].keys():
                values = [r['retrieval_metrics'][metric] for r in all_results]
                retrieval_metrics[f'avg_{metric}'] = np.mean(values)
                retrieval_metrics[f'std_{metric}'] = np.std(values)
            
            # Aggregate answer metrics
            for metric in all_results[0]['answer_metrics'].keys():
                values = [r['answer_metrics'][metric] for r in all_results]
                answer_metrics[f'avg_{metric}'] = np.mean(values)
                answer_metrics[f'std_{metric}'] = np.std(values)
            
            # Aggregate composite scores
            retrieval_scores = [r['composite_scores']['retrieval_score'] for r in all_results]
            answer_scores = [r['composite_scores']['answer_score'] for r in all_results]
            overall_scores = [r['composite_scores']['overall_score'] for r in all_results]
            
            composite_scores = {
                'avg_retrieval_score': np.mean(retrieval_scores),
                'avg_answer_score': np.mean(answer_scores),
                'avg_overall_score': np.mean(overall_scores),
                'std_retrieval_score': np.std(retrieval_scores),
                'std_answer_score': np.std(answer_scores),
                'std_overall_score': np.std(overall_scores)
            }
        else:
            composite_scores = {}
        
        return {
            'dataset_size': len(dataset),
            'individual_results': all_results,
            'aggregate_retrieval_metrics': retrieval_metrics,
            'aggregate_answer_metrics': answer_metrics,
            'aggregate_composite_scores': composite_scores,
            'evaluation_timestamp': datetime.now().isoformat()
        }

def create_sample_evaluation_dataset() -> List[Dict[str, Any]]:
    """Create a sample evaluation dataset for testing."""
    
    return [
        {
            'query': 'Wie werden Überstunden in Deutschland besteuert?',
            'ground_truth': {
                'reference_answer': 'Überstunden werden als regulärer Arbeitslohn besteuert und unterliegen der Lohnsteuer sowie den Sozialversicherungsbeiträgen.',
                'relevant_documents': ['Überstunden', 'Lohnsteuer', 'Sozialversicherung'],
                'expected_elements': ['Lohnsteuer', 'Sozialversicherungsbeiträge', 'regulärer Arbeitslohn']
            }
        },
        {
            'query': 'Was ist die Lohnsteuerkarte?',
            'ground_truth': {
                'reference_answer': 'Die Lohnsteuerkarte ist ein elektronisches Verfahren zur Übermittlung der steuerlichen Merkmale an den Arbeitgeber.',
                'relevant_documents': ['Lohnsteuerkarte', 'ELStAM', 'steuerliche Merkmale'],
                'expected_elements': ['elektronisches Verfahren', 'steuerliche Merkmale', 'Arbeitgeber']
            }
        }
    ]

def main():
    """Test the evaluation system."""
    evaluator = RAGEvaluator()
    
    # Test with sample data
    sample_dataset = create_sample_evaluation_dataset()
    
    print("Testing evaluation system...")
    print(f"Sample dataset size: {len(sample_dataset)}")
    
    # Simulate some retrieval and answer data
    for item in sample_dataset:
        item['retrieved_docs'] = [
            {'content': 'Sample document content about the topic'},
            {'content': 'Another relevant document'}
        ]
        item['answer'] = 'Sample answer for testing'
    
    # Run evaluation
    results = evaluator.evaluate_dataset(sample_dataset)
    
    print(f"\nEvaluation Results:")
    print(f"Dataset size: {results['dataset_size']}")
    print(f"Average overall score: {results['aggregate_composite_scores'].get('avg_overall_score', 'N/A'):.3f}")
    
    # Print detailed metrics
    print(f"\nAggregate Retrieval Metrics:")
    for metric, value in results['aggregate_retrieval_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nAggregate Answer Metrics:")
    for metric, value in results['aggregate_answer_metrics'].items():
        print(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    main()
