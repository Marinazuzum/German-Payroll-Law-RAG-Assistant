#!/usr/bin/env python3
"""
Experimental runner script for evaluating different RAG configurations.
This script can be used to run comprehensive experiments and generate reports.
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.experiment_runner import ExperimentRunner, create_test_dataset
from src.llm.prompt_strategies import PromptStrategy
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run RAG system experiments')
    parser.add_argument('--experiment', choices=['chunking', 'retrieval', 'prompt', 'comprehensive'], 
                       default='comprehensive', help='Type of experiment to run')
    parser.add_argument('--output-dir', default='./data/experiments', 
                       help='Output directory for results')
    parser.add_argument('--time-range', type=int, default=7, 
                       help='Time range in days for data analysis')
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Create test dataset
    test_dataset = create_test_dataset()
    test_queries = [item['query'] for item in test_dataset]
    
    print(f"🧪 Running {args.experiment} experiment...")
    print(f"📊 Test dataset size: {len(test_dataset)}")
    print(f"📁 Output directory: {args.output_dir}")
    
    try:
        if args.experiment == 'chunking':
            results = runner.run_chunking_experiment(test_queries)
            
        elif args.experiment == 'retrieval':
            results = runner.run_retrieval_strategy_experiment(test_queries)
            
        elif args.experiment == 'prompt':
            results = runner.run_prompt_strategy_experiment(test_queries)
            
        elif args.experiment == 'comprehensive':
            results = runner.run_comprehensive_evaluation(
                test_dataset,
                retrieval_strategies=['vector_only', 'hybrid', 'hybrid_with_rerank'],
                prompt_strategies=[PromptStrategy.BASIC, PromptStrategy.STRUCTURED, PromptStrategy.LEGAL_EXPERT]
            )
        
        # Generate and save report
        report_path = runner.save_experiment_report(results)
        print(f"✅ Experiment completed successfully!")
        print(f"📄 Report saved to: {report_path}")
        
        # Print summary
        if 'summary' in results:
            print("\n📈 Summary:")
            for config, stats in results['summary'].items():
                print(f"  {config}: {stats['avg_overall_score']:.3f} overall score, "
                      f"{stats['success_rate']:.1%} success rate")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
