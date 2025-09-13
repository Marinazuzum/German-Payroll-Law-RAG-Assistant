import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.llm_client import LLMClient
from src.llm.prompt_strategies import PromptStrategy
from src.evaluation.metrics import RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Run experiments to compare different RAG configurations."""
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm_client = LLMClient()
        self.evaluator = RAGEvaluator()
        
        # Create results directory
        self.results_dir = Path(settings.processed_data_path) / "experiments"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_chunking_experiment(self, 
                               queries: List[str],
                               chunk_sizes: List[int] = [500, 1000, 1500, 2000]) -> Dict[str, Any]:
        """Experiment with different chunk sizes."""
        logger.info("Running chunking size experiment...")
        
        results = {
            'experiment_type': 'chunking_sizes',
            'chunk_sizes': chunk_sizes,
            'queries': queries,
            'results': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for chunk_size in chunk_sizes:
            logger.info(f"Testing chunk size: {chunk_size}")
            
            # Note: In a real implementation, you would need to re-process documents
            # with different chunk sizes. For this demo, we'll simulate the effect.
            
            chunk_results = []
            for query in queries:
                # Simulate retrieval with different chunk size
                # In practice, you'd need separate collections for each chunk size
                retrieved_docs = self.retriever.retrieve(query, top_k=5)
                
                # Simulate different performance based on chunk size
                # Smaller chunks might have higher precision but lower coverage
                if chunk_size < 1000:
                    # Simulate higher precision, lower coverage
                    retrieved_docs = retrieved_docs[:3]  # Take fewer docs
                elif chunk_size > 1500:
                    # Simulate lower precision, higher coverage
                    for doc in retrieved_docs:
                        doc['score'] *= 0.9  # Slightly lower scores
                
                chunk_results.append({
                    'query': query,
                    'retrieved_docs_count': len(retrieved_docs),
                    'avg_score': sum(doc.get('combined_score', 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                    'avg_chunk_size': chunk_size  # Simulated
                })
            
            results['results'][chunk_size] = chunk_results
        
        # Save results
        results_file = self.results_dir / f"chunking_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Chunking experiment completed. Results saved to {results_file}")
        return results
    
    def run_retrieval_strategy_experiment(self, 
                                        queries: List[str],
                                        strategies: List[str] = None) -> Dict[str, Any]:
        """Experiment with different retrieval strategies."""
        logger.info("Running retrieval strategy experiment...")
        
        if strategies is None:
            strategies = ['vector_only', 'hybrid', 'hybrid_with_rerank']
        
        results = {
            'experiment_type': 'retrieval_strategies',
            'strategies': strategies,
            'queries': queries,
            'results': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for strategy in strategies:
            logger.info(f"Testing retrieval strategy: {strategy}")
            
            strategy_results = []
            for query in queries:
                start_time = datetime.now()
                
                if strategy == 'vector_only':
                    retrieved_docs = self.retriever.vector_search(query, top_k=5)
                elif strategy == 'hybrid':
                    retrieved_docs = self.retriever.hybrid_search(query, top_k=5)
                elif strategy == 'hybrid_with_rerank':
                    retrieved_docs = self.retriever.retrieve(query, use_hybrid=True, use_reranking=True)
                else:
                    retrieved_docs = self.retriever.retrieve(query)
                
                retrieval_time = (datetime.now() - start_time).total_seconds()
                
                strategy_results.append({
                    'query': query,
                    'retrieved_docs_count': len(retrieved_docs),
                    'retrieval_time': retrieval_time,
                    'avg_score': sum(doc.get('combined_score', doc.get('score', 0)) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                    'top_score': max((doc.get('combined_score', doc.get('score', 0)) for doc in retrieved_docs), default=0)
                })
            
            results['results'][strategy] = strategy_results
        
        # Save results
        results_file = self.results_dir / f"retrieval_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Retrieval strategy experiment completed. Results saved to {results_file}")
        return results
    
    def run_prompt_strategy_experiment(self, 
                                     queries: List[str],
                                     strategies: List[PromptStrategy] = None) -> Dict[str, Any]:
        """Experiment with different prompt strategies."""
        logger.info("Running prompt strategy experiment...")
        
        if strategies is None:
            strategies = [
                PromptStrategy.BASIC,
                PromptStrategy.STRUCTURED,
                PromptStrategy.LEGAL_EXPERT,
                PromptStrategy.STEP_BY_STEP
            ]
        
        results = {
            'experiment_type': 'prompt_strategies',
            'strategies': [s.value for s in strategies],
            'queries': queries,
            'results': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for strategy in strategies:
            logger.info(f"Testing prompt strategy: {strategy.value}")
            
            strategy_results = []
            for query in queries:
                # Retrieve documents
                retrieved_docs = self.retriever.retrieve(query, top_k=3)
                
                # Generate answer with specific strategy
                result = self.llm_client.generate_answer(
                    query=query,
                    retrieved_documents=retrieved_docs,
                    strategy=strategy
                )
                
                strategy_results.append({
                    'query': query,
                    'answer': result.get('answer', ''),
                    'response_time': result.get('response_time', 0),
                    'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                    'success': result.get('success', False),
                    'answer_length': len(result.get('answer', ''))
                })
            
            results['results'][strategy.value] = strategy_results
        
        # Save results
        results_file = self.results_dir / f"prompt_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Prompt strategy experiment completed. Results saved to {results_file}")
        return results
    
    def run_comprehensive_evaluation(self, 
                                   test_dataset: List[Dict[str, Any]],
                                   retrieval_strategies: List[str] = None,
                                   prompt_strategies: List[PromptStrategy] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation with multiple configurations."""
        logger.info("Running comprehensive evaluation...")
        
        if retrieval_strategies is None:
            retrieval_strategies = ['hybrid_with_rerank']
        
        if prompt_strategies is None:
            prompt_strategies = [PromptStrategy.STRUCTURED, PromptStrategy.LEGAL_EXPERT]
        
        results = {
            'experiment_type': 'comprehensive_evaluation',
            'dataset_size': len(test_dataset),
            'retrieval_strategies': retrieval_strategies,
            'prompt_strategies': [s.value for s in prompt_strategies],
            'results': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for ret_strategy in retrieval_strategies:
            for prompt_strategy in prompt_strategies:
                config_name = f"{ret_strategy}_{prompt_strategy.value}"
                logger.info(f"Evaluating configuration: {config_name}")
                
                config_results = []
                
                for item in test_dataset:
                    query = item['query']
                    ground_truth = item.get('ground_truth', {})
                    
                    # Retrieve documents
                    if ret_strategy == 'vector_only':
                        retrieved_docs = self.retriever.vector_search(query, top_k=5)
                    elif ret_strategy == 'hybrid':
                        retrieved_docs = self.retriever.hybrid_search(query, top_k=5)
                    else:  # hybrid_with_rerank
                        retrieved_docs = self.retriever.retrieve(query, use_hybrid=True, use_reranking=True)
                    
                    # Generate answer
                    answer_result = self.llm_client.generate_answer(
                        query=query,
                        retrieved_documents=retrieved_docs,
                        strategy=prompt_strategy
                    )
                    
                    # Evaluate
                    if answer_result.get('success', False):
                        evaluation = self.evaluator.evaluate_end_to_end(
                            query=query,
                            retrieved_docs=retrieved_docs,
                            answer=answer_result['answer'],
                            ground_truth=ground_truth
                        )
                        
                        config_results.append({
                            'query': query,
                            'retrieval_count': len(retrieved_docs),
                            'answer_generated': True,
                            'response_time': answer_result.get('response_time', 0),
                            'tokens_used': answer_result.get('usage', {}).get('total_tokens', 0),
                            'evaluation_metrics': evaluation
                        })
                    else:
                        config_results.append({
                            'query': query,
                            'retrieval_count': len(retrieved_docs),
                            'answer_generated': False,
                            'error': answer_result.get('error', 'Unknown error')
                        })
                
                results['results'][config_name] = config_results
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results['results'])
        results['summary'] = summary
        
        # Save results
        results_file = self.results_dir / f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive evaluation completed. Results saved to {results_file}")
        return results
    
    def _calculate_summary_stats(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate summary statistics for experiment results."""
        summary = {}
        
        for config_name, config_results in results.items():
            successful_results = [r for r in config_results if r.get('answer_generated', False)]
            
            if successful_results:
                # Calculate averages
                avg_response_time = sum(r.get('response_time', 0) for r in successful_results) / len(successful_results)
                avg_tokens = sum(r.get('tokens_used', 0) for r in successful_results) / len(successful_results)
                
                # Extract evaluation metrics
                overall_scores = []
                retrieval_scores = []
                answer_scores = []
                
                for r in successful_results:
                    eval_metrics = r.get('evaluation_metrics', {})
                    composite = eval_metrics.get('composite_scores', {})
                    
                    if composite:
                        overall_scores.append(composite.get('overall_score', 0))
                        retrieval_scores.append(composite.get('retrieval_score', 0))
                        answer_scores.append(composite.get('answer_score', 0))
                
                summary[config_name] = {
                    'total_queries': len(config_results),
                    'successful_queries': len(successful_results),
                    'success_rate': len(successful_results) / len(config_results),
                    'avg_response_time': avg_response_time,
                    'avg_tokens_used': avg_tokens,
                    'avg_overall_score': sum(overall_scores) / len(overall_scores) if overall_scores else 0,
                    'avg_retrieval_score': sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0,
                    'avg_answer_score': sum(answer_scores) / len(answer_scores) if answer_scores else 0
                }
            else:
                summary[config_name] = {
                    'total_queries': len(config_results),
                    'successful_queries': 0,
                    'success_rate': 0,
                    'avg_response_time': 0,
                    'avg_tokens_used': 0,
                    'avg_overall_score': 0,
                    'avg_retrieval_score': 0,
                    'avg_answer_score': 0
                }
        
        return summary
    
    def generate_experiment_report(self, results: Dict[str, Any]) -> str:
        """Generate a text report of experiment results."""
        report = []
        report.append(f"# Experiment Report: {results['experiment_type'].title()}")
        report.append(f"Generated on: {results['timestamp']}")
        report.append("")
        
        if results['experiment_type'] == 'comprehensive_evaluation':
            report.append("## Summary Statistics")
            report.append("")
            
            summary = results.get('summary', {})
            
            # Create comparison table
            if summary:
                report.append("| Configuration | Success Rate | Avg Overall Score | Avg Response Time | Avg Tokens |")
                report.append("|---------------|--------------|-------------------|-------------------|------------|")
                
                for config, stats in summary.items():
                    report.append(
                        f"| {config} | {stats['success_rate']:.1%} | "
                        f"{stats['avg_overall_score']:.3f} | "
                        f"{stats['avg_response_time']:.2f}s | "
                        f"{stats['avg_tokens_used']:.0f} |"
                    )
                report.append("")
            
            # Find best configuration
            if summary:
                best_config = max(summary.keys(), key=lambda k: summary[k]['avg_overall_score'])
                report.append(f"**Best Configuration:** {best_config}")
                report.append(f"- Overall Score: {summary[best_config]['avg_overall_score']:.3f}")
                report.append(f"- Success Rate: {summary[best_config]['success_rate']:.1%}")
                report.append(f"- Avg Response Time: {summary[best_config]['avg_response_time']:.2f}s")
                report.append("")
        
        elif results['experiment_type'] == 'retrieval_strategies':
            report.append("## Retrieval Strategy Comparison")
            report.append("")
            
            for strategy, strategy_results in results['results'].items():
                avg_score = sum(r['avg_score'] for r in strategy_results) / len(strategy_results)
                avg_time = sum(r['retrieval_time'] for r in strategy_results) / len(strategy_results)
                
                report.append(f"### {strategy}")
                report.append(f"- Average Score: {avg_score:.3f}")
                report.append(f"- Average Retrieval Time: {avg_time:.3f}s")
                report.append("")
        
        elif results['experiment_type'] == 'prompt_strategies':
            report.append("## Prompt Strategy Comparison")
            report.append("")
            
            for strategy, strategy_results in results['results'].items():
                successful = [r for r in strategy_results if r['success']]
                if successful:
                    avg_time = sum(r['response_time'] for r in successful) / len(successful)
                    avg_tokens = sum(r['tokens_used'] for r in successful) / len(successful)
                    avg_length = sum(r['answer_length'] for r in successful) / len(successful)
                    
                    report.append(f"### {strategy}")
                    report.append(f"- Success Rate: {len(successful)}/{len(strategy_results)} ({len(successful)/len(strategy_results):.1%})")
                    report.append(f"- Average Response Time: {avg_time:.2f}s")
                    report.append(f"- Average Tokens Used: {avg_tokens:.0f}")
                    report.append(f"- Average Answer Length: {avg_length:.0f} characters")
                    report.append("")
        
        return "\n".join(report)
    
    def save_experiment_report(self, results: Dict[str, Any]) -> str:
        """Save experiment report to file."""
        report = self.generate_experiment_report(results)
        
        report_file = self.results_dir / f"report_{results['experiment_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Experiment report saved to {report_file}")
        return str(report_file)

def create_test_dataset() -> List[Dict[str, Any]]:
    """Create a test dataset for experiments."""
    return [
        {
            'query': 'Wie wird das Gehalt in Deutschland versteuert?',
            'ground_truth': {
                'reference_answer': 'Das Gehalt wird über die Lohnsteuer versteuert, die vom Arbeitgeber direkt abgeführt wird.',
                'relevant_documents': ['Lohnsteuer', 'Gehaltsabrechnung'],
                'expected_elements': ['Lohnsteuer', 'Arbeitgeber', 'direkte Abführung']
            }
        },
        {
            'query': 'Was sind Sozialversicherungsbeiträge?',
            'ground_truth': {
                'reference_answer': 'Sozialversicherungsbeiträge sind Beiträge für Kranken-, Renten-, Pflege- und Arbeitslosenversicherung.',
                'relevant_documents': ['Sozialversicherung', 'Beiträge'],
                'expected_elements': ['Krankenversicherung', 'Rentenversicherung', 'Pflegeversicherung', 'Arbeitslosenversicherung']
            }
        },
        {
            'query': 'Wie funktioniert die Lohnabrechnung?',
            'ground_truth': {
                'reference_answer': 'Die Lohnabrechnung erfolgt monatlich und zeigt Brutto- und Nettolohn sowie alle Abzüge.',
                'relevant_documents': ['Lohnabrechnung', 'Bruttoentgelt', 'Nettoentgelt'],
                'expected_elements': ['monatlich', 'Bruttolohn', 'Nettolohn', 'Abzüge']
            }
        }
    ]

def main():
    """Run sample experiments."""
    runner = ExperimentRunner()
    
    # Create test dataset
    test_dataset = create_test_dataset()
    test_queries = [item['query'] for item in test_dataset]
    
    print("Running sample experiments...")
    
    # Run retrieval experiment
    print("\n1. Running retrieval strategy experiment...")
    retrieval_results = runner.run_retrieval_strategy_experiment(test_queries[:2])
    print(f"Retrieval experiment completed with {len(retrieval_results['results'])} strategies")
    
    # Run prompt experiment
    print("\n2. Running prompt strategy experiment...")
    prompt_results = runner.run_prompt_strategy_experiment(test_queries[:1])
    print(f"Prompt experiment completed with {len(prompt_results['results'])} strategies")
    
    # Generate reports
    print("\n3. Generating reports...")
    retrieval_report = runner.save_experiment_report(retrieval_results)
    prompt_report = runner.save_experiment_report(prompt_results)
    
    print(f"Reports saved:")
    print(f"- Retrieval: {retrieval_report}")
    print(f"- Prompt: {prompt_report}")

if __name__ == "__main__":
    main()
