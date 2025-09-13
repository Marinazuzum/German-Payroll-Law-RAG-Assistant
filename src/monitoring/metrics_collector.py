import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from collections import defaultdict, Counter
import threading

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and manage application metrics."""
    
    def __init__(self, metrics_file: str = None):
        self.metrics_file = Path(metrics_file or settings.metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics
        self._metrics = {
            'queries': [],
            'feedback': [],
            'system_events': [],
            'performance': [],
            'errors': []
        }
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Load existing metrics
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from file."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self._metrics.update(data)
                logger.info(f"Loaded metrics from {self.metrics_file}")
            else:
                logger.info("No existing metrics file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            with self._lock:
                with open(self.metrics_file, 'w') as f:
                    json.dump(self._metrics, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def record_query(self, 
                    query: str, 
                    response_time: float,
                    tokens_used: int,
                    retrieved_docs_count: int,
                    strategy: str,
                    success: bool = True,
                    error_message: str = None):
        """Record a query and its metrics."""
        
        query_record = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_length': len(query),
            'response_time': response_time,
            'tokens_used': tokens_used,
            'retrieved_docs_count': retrieved_docs_count,
            'strategy': strategy,
            'success': success,
            'error_message': error_message,
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'date': datetime.now().date().isoformat()
        }
        
        with self._lock:
            self._metrics['queries'].append(query_record)
        
        # Save periodically (every 10 queries)
        if len(self._metrics['queries']) % 10 == 0:
            self._save_metrics()
        
        logger.info(f"Recorded query: {query[:50]}... (success: {success})")
    
    def record_feedback(self, 
                       query: str, 
                       feedback_type: str,
                       timestamp: str = None,
                       comments: str = None):
        """Record user feedback."""
        
        feedback_record = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'query': query,
            'feedback_type': feedback_type,  # 'positive', 'negative', 'neutral'
            'comments': comments,
            'date': datetime.now().date().isoformat()
        }
        
        with self._lock:
            self._metrics['feedback'].append(feedback_record)
        
        self._save_metrics()
        
        logger.info(f"Recorded feedback: {feedback_type} for query: {query[:50]}...")
    
    def record_system_event(self, 
                           event_type: str, 
                           description: str,
                           details: Dict[str, Any] = None):
        """Record system events."""
        
        event_record = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'description': description,
            'details': details or {},
            'date': datetime.now().date().isoformat()
        }
        
        with self._lock:
            self._metrics['system_events'].append(event_record)
        
        logger.info(f"Recorded system event: {event_type} - {description}")
    
    def record_performance_metric(self, 
                                 metric_name: str, 
                                 value: float,
                                 unit: str = None):
        """Record performance metrics."""
        
        perf_record = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'date': datetime.now().date().isoformat()
        }
        
        with self._lock:
            self._metrics['performance'].append(perf_record)
    
    def record_error(self, 
                    error_type: str, 
                    error_message: str,
                    context: Dict[str, Any] = None):
        """Record application errors."""
        
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {},
            'date': datetime.now().date().isoformat()
        }
        
        with self._lock:
            self._metrics['errors'].append(error_record)
        
        logger.error(f"Recorded error: {error_type} - {error_message}")
    
    def get_metrics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get comprehensive metrics for the last N days."""
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).date()
        
        with self._lock:
            # Filter recent data
            recent_queries = [
                q for q in self._metrics['queries']
                if datetime.fromisoformat(q['timestamp']).date() >= cutoff_date
            ]
            
            recent_feedback = [
                f for f in self._metrics['feedback']
                if datetime.fromisoformat(f['timestamp']).date() >= cutoff_date
            ]
            
            recent_errors = [
                e for e in self._metrics['errors']
                if datetime.fromisoformat(e['timestamp']).date() >= cutoff_date
            ]
        
        # Calculate metrics
        metrics = {}
        
        # Query metrics
        if recent_queries:
            successful_queries = [q for q in recent_queries if q['success']]
            
            metrics.update({
                'total_queries': len(recent_queries),
                'successful_queries': len(successful_queries),
                'success_rate': len(successful_queries) / len(recent_queries),
                'avg_response_time': sum(q['response_time'] for q in successful_queries) / len(successful_queries) if successful_queries else 0,
                'avg_tokens': sum(q['tokens_used'] for q in successful_queries) / len(successful_queries) if successful_queries else 0,
                'avg_retrieved_docs': sum(q['retrieved_docs_count'] for q in successful_queries) / len(successful_queries) if successful_queries else 0,
                'queries_today': len([q for q in recent_queries if datetime.fromisoformat(q['timestamp']).date() == datetime.now().date()]),
                'recent_queries': recent_queries[-10:]  # Last 10 queries
            })
            
            # Strategy usage
            strategy_counts = Counter(q['strategy'] for q in recent_queries)
            metrics['most_used_strategy'] = strategy_counts.most_common(1)[0][0] if strategy_counts else 'N/A'
            metrics['strategy_distribution'] = dict(strategy_counts)
            
            # Peak usage analysis
            hour_counts = Counter(q['hour'] for q in recent_queries)
            metrics['peak_hour'] = hour_counts.most_common(1)[0][0] if hour_counts else 'N/A'
            metrics['hourly_distribution'] = dict(hour_counts)
            
            # Daily trends
            daily_counts = Counter(q['date'] for q in recent_queries)
            metrics['daily_query_counts'] = dict(daily_counts)
            
        else:
            metrics.update({
                'total_queries': 0,
                'successful_queries': 0,
                'success_rate': 0,
                'avg_response_time': 0,
                'avg_tokens': 0,
                'avg_retrieved_docs': 0,
                'queries_today': 0,
                'recent_queries': [],
                'most_used_strategy': 'N/A',
                'strategy_distribution': {},
                'peak_hour': 'N/A',
                'hourly_distribution': {},
                'daily_query_counts': {}
            })
        
        # Feedback metrics
        if recent_feedback:
            feedback_counts = Counter(f['feedback_type'] for f in recent_feedback)
            total_feedback = len(recent_feedback)
            
            metrics.update({
                'total_feedback': total_feedback,
                'positive_feedback': feedback_counts.get('positive', 0),
                'negative_feedback': feedback_counts.get('negative', 0),
                'neutral_feedback': feedback_counts.get('neutral', 0),
                'satisfaction_rate': feedback_counts.get('positive', 0) / total_feedback if total_feedback > 0 else 0,
                'feedback_distribution': dict(feedback_counts)
            })
        else:
            metrics.update({
                'total_feedback': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'neutral_feedback': 0,
                'satisfaction_rate': 0,
                'feedback_distribution': {}
            })
        
        # Error metrics
        metrics.update({
            'total_errors': len(recent_errors),
            'error_rate': len(recent_errors) / len(recent_queries) if recent_queries else 0,
            'recent_errors': recent_errors[-5:]  # Last 5 errors
        })
        
        if recent_errors:
            error_type_counts = Counter(e['error_type'] for e in recent_errors)
            metrics['error_distribution'] = dict(error_type_counts)
        else:
            metrics['error_distribution'] = {}
        
        # Performance trends
        metrics['performance_trends'] = self._calculate_performance_trends(recent_queries)
        
        # System health score
        metrics['system_health_score'] = self._calculate_health_score(metrics)
        
        return metrics
    
    def _calculate_performance_trends(self, queries: List[Dict]) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if not queries:
            return {}
        
        # Group by day
        daily_performance = defaultdict(list)
        for query in queries:
            date = query['date']
            daily_performance[date].append({
                'response_time': query['response_time'],
                'tokens_used': query['tokens_used'],
                'success': query['success']
            })
        
        trends = {}
        for date, day_queries in daily_performance.items():
            successful = [q for q in day_queries if q['success']]
            if successful:
                trends[date] = {
                    'avg_response_time': sum(q['response_time'] for q in successful) / len(successful),
                    'avg_tokens': sum(q['tokens_used'] for q in successful) / len(successful),
                    'success_rate': len(successful) / len(day_queries),
                    'query_count': len(day_queries)
                }
        
        return trends
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # Penalize based on success rate
        success_rate = metrics.get('success_rate', 1.0)
        score *= success_rate
        
        # Penalize based on error rate
        error_rate = metrics.get('error_rate', 0)
        score *= (1 - min(error_rate, 0.5))  # Cap error penalty at 50%
        
        # Penalize based on response time (target: < 5 seconds)
        avg_response_time = metrics.get('avg_response_time', 0)
        if avg_response_time > 5:
            score *= 0.8
        elif avg_response_time > 10:
            score *= 0.6
        
        # Bonus for user satisfaction
        satisfaction_rate = metrics.get('satisfaction_rate', 0)
        score *= (0.5 + 0.5 * satisfaction_rate)  # Scale from 50% to 100%
        
        return max(0, min(100, score))
    
    def get_top_queries(self, limit: int = 10, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get most common queries."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).date()
        
        recent_queries = [
            q for q in self._metrics['queries']
            if datetime.fromisoformat(q['timestamp']).date() >= cutoff_date
        ]
        
        query_counts = Counter(q['query'] for q in recent_queries)
        
        return [
            {'query': query, 'count': count}
            for query, count in query_counts.most_common(limit)
        ]
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export metrics in specified format."""
        metrics = self.get_metrics(days_back=30)  # Last 30 days
        
        if format_type == 'json':
            filename = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.metrics_file.parent / filename
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            return str(filepath)
        
        elif format_type == 'csv':
            import pandas as pd
            
            # Export queries as CSV
            queries_df = pd.DataFrame(self._metrics['queries'])
            filename = f"queries_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.metrics_file.parent / filename
            
            queries_df.to_csv(filepath, index=False)
            return str(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old metrics data."""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date()
        
        original_counts = {
            'queries': len(self._metrics['queries']),
            'feedback': len(self._metrics['feedback']),
            'errors': len(self._metrics['errors'])
        }
        
        with self._lock:
            # Filter each metric type
            self._metrics['queries'] = [
                q for q in self._metrics['queries']
                if datetime.fromisoformat(q['timestamp']).date() >= cutoff_date
            ]
            
            self._metrics['feedback'] = [
                f for f in self._metrics['feedback']
                if datetime.fromisoformat(f['timestamp']).date() >= cutoff_date
            ]
            
            self._metrics['errors'] = [
                e for e in self._metrics['errors']
                if datetime.fromisoformat(e['timestamp']).date() >= cutoff_date
            ]
        
        new_counts = {
            'queries': len(self._metrics['queries']),
            'feedback': len(self._metrics['feedback']),
            'errors': len(self._metrics['errors'])
        }
        
        # Save cleaned data
        self._save_metrics()
        
        logger.info(f"Cleaned up old data. Removed: "
                   f"queries: {original_counts['queries'] - new_counts['queries']}, "
                   f"feedback: {original_counts['feedback'] - new_counts['feedback']}, "
                   f"errors: {original_counts['errors'] - new_counts['errors']}")
        
        return original_counts, new_counts

def main():
    """Test the metrics collector."""
    collector = MetricsCollector()
    
    # Record some sample data
    collector.record_query(
        query="Wie werden Überstunden versteuert?",
        response_time=2.5,
        tokens_used=150,
        retrieved_docs_count=5,
        strategy="structured"
    )
    
    collector.record_feedback(
        query="Wie werden Überstunden versteuert?",
        feedback_type="positive"
    )
    
    # Get metrics
    metrics = collector.get_metrics()
    
    print("Sample Metrics:")
    print(f"Total queries: {metrics['total_queries']}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Average response time: {metrics['avg_response_time']:.2f}s")
    print(f"System health score: {metrics['system_health_score']:.1f}")

if __name__ == "__main__":
    main()
