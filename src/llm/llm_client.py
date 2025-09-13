import openai
from typing import List, Dict, Any, Optional
import logging
import time
import json
from datetime import datetime

from config import settings
from src.llm.prompt_strategies import PromptBuilder, PromptStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with OpenAI LLM with different prompt strategies."""
    
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = settings.openai_api_key
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.prompt_builder = PromptBuilder()
        
        # Track usage statistics
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_estimate": 0.0,
            "strategies_used": {},
            "errors": 0
        }
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_documents: List[Dict[str, Any]],
                       strategy: PromptStrategy = PromptStrategy.STRUCTURED,
                       include_sources: bool = True,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate answer using specified prompt strategy."""
        
        temperature = temperature or settings.temperature
        max_tokens = max_tokens or settings.max_tokens
        
        try:
            # Build prompt
            prompt = self.prompt_builder.build_prompt(
                strategy=strategy,
                query=query,
                retrieved_documents=retrieved_documents,
                include_sources=include_sources
            )
            
            logger.info(f"Generating answer with strategy: {strategy.value}")
            
            # Record start time
            start_time = time.time()
            
            # Make API call
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Sie sind ein Experte für deutsches Lohnsteuerrecht. Geben Sie präzise, gut begründete Antworten basierend auf den bereitgestellten Dokumenten. Antworten Sie immer auf Deutsch."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Update usage statistics
            self._update_usage_stats(strategy, response.usage, response_time)
            
            # Prepare result
            result = {
                "answer": answer,
                "strategy": strategy.value,
                "query": query,
                "retrieved_documents": len(retrieved_documents),
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": settings.llm_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            if include_sources:
                result["sources"] = self._extract_sources(retrieved_documents)
            
            logger.info(f"Answer generated successfully in {response_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            self.usage_stats["errors"] += 1
            
            return {
                "answer": f"Entschuldigung, es ist ein Fehler aufgetreten: {str(e)}",
                "strategy": strategy.value,
                "query": query,
                "retrieved_documents": len(retrieved_documents),
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def compare_strategies(self, 
                          query: str, 
                          retrieved_documents: List[Dict[str, Any]],
                          strategies: Optional[List[PromptStrategy]] = None) -> Dict[str, Any]:
        """Compare multiple prompt strategies for the same query."""
        
        if strategies is None:
            strategies = [
                PromptStrategy.BASIC,
                PromptStrategy.STRUCTURED,
                PromptStrategy.LEGAL_EXPERT
            ]
        
        logger.info(f"Comparing {len(strategies)} strategies for query")
        
        results = {}
        total_start_time = time.time()
        
        for strategy in strategies:
            logger.info(f"Testing strategy: {strategy.value}")
            result = self.generate_answer(query, retrieved_documents, strategy)
            results[strategy.value] = result
        
        total_time = time.time() - total_start_time
        
        comparison = {
            "query": query,
            "strategies_compared": len(strategies),
            "results": results,
            "total_comparison_time": total_time,
            "timestamp": datetime.now().isoformat(),
            "summary": self._generate_comparison_summary(results)
        }
        
        return comparison
    
    def _extract_sources(self, retrieved_documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from retrieved documents."""
        sources = []
        for doc in retrieved_documents:
            if 'metadata' in doc:
                metadata = doc['metadata']
                source = {
                    "file_name": metadata.get('file_name', 'Unbekannt'),
                    "source_path": metadata.get('source', 'Unbekannt'),
                    "chunk_id": metadata.get('chunk_id', 'Unbekannt'),
                    "content_preview": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                }
                sources.append(source)
        return sources
    
    def _update_usage_stats(self, strategy: PromptStrategy, usage, response_time: float):
        """Update usage statistics."""
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_tokens"] += usage.total_tokens
        
        # Rough cost estimation (GPT-3.5-turbo pricing)
        cost_per_1k_tokens = 0.002  # Approximate
        cost = (usage.total_tokens / 1000) * cost_per_1k_tokens
        self.usage_stats["total_cost_estimate"] += cost
        
        # Track strategy usage
        strategy_name = strategy.value
        if strategy_name not in self.usage_stats["strategies_used"]:
            self.usage_stats["strategies_used"][strategy_name] = {
                "count": 0,
                "total_tokens": 0,
                "total_time": 0,
                "avg_time": 0
            }
        
        strategy_stats = self.usage_stats["strategies_used"][strategy_name]
        strategy_stats["count"] += 1
        strategy_stats["total_tokens"] += usage.total_tokens
        strategy_stats["total_time"] += response_time
        strategy_stats["avg_time"] = strategy_stats["total_time"] / strategy_stats["count"]
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of strategy comparison."""
        summary = {
            "successful_strategies": 0,
            "failed_strategies": 0,
            "avg_response_time": 0,
            "total_tokens_used": 0,
            "best_performing_strategy": None,
            "fastest_strategy": None
        }
        
        response_times = []
        successful_results = []
        
        for strategy_name, result in results.items():
            if result.get("success", False):
                summary["successful_strategies"] += 1
                successful_results.append((strategy_name, result))
                response_times.append(result.get("response_time", 0))
                summary["total_tokens_used"] += result.get("usage", {}).get("total_tokens", 0)
            else:
                summary["failed_strategies"] += 1
        
        if response_times:
            summary["avg_response_time"] = sum(response_times) / len(response_times)
            
            # Find fastest strategy
            fastest_idx = response_times.index(min(response_times))
            summary["fastest_strategy"] = successful_results[fastest_idx][0]
        
        # Note: "best_performing_strategy" would need evaluation metrics to determine
        summary["best_performing_strategy"] = "Requires evaluation metrics"
        
        return summary
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats.copy()
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_estimate": 0.0,
            "strategies_used": {},
            "errors": 0
        }
        logger.info("Usage statistics reset")

def main():
    """Test the LLM client."""
    try:
        # Create client
        client = LLMClient()
        
        # Sample data
        sample_query = "Wie werden Überstunden in Deutschland besteuert?"
        sample_documents = [
            {
                'content': "Überstunden sind grundsätzlich steuerpflichtiger Arbeitslohn. Sie unterliegen der Lohnsteuer und den Sozialversicherungsbeiträgen.",
                'metadata': {'source': '/path/to/doc1.pdf', 'file_name': 'lohnsteuer_grundlagen.pdf'},
                'score': 0.85
            }
        ]
        
        print("Testing LLM Client with sample query...")
        print(f"Query: {sample_query}\n")
        
        # Test single strategy
        result = client.generate_answer(
            query=sample_query,
            retrieved_documents=sample_documents,
            strategy=PromptStrategy.STRUCTURED
        )
        
        print("Single Strategy Result:")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Response time: {result['response_time']:.2f}s")
            print(f"Tokens used: {result['usage']['total_tokens']}")
        
        print(f"\nUsage Stats: {client.get_usage_stats()}")
        
    except Exception as e:
        print(f"Error testing LLM client: {e}")
        print("Note: This requires a valid OpenAI API key in config")

if __name__ == "__main__":
    main()
