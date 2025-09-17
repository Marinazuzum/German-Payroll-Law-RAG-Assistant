import streamlit as st
import sys
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring.metrics_collector import MetricsCollector
from config import settings

# Page configuration
st.set_page_config(
    page_title="RAG System Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        margin: 0;
        color: #374151;
        font-size: 1.5rem;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        color: #6b7280;
        font-size: 0.875rem;
    }
    .alert-red {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-yellow {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-green {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize metrics collector
@st.cache_resource
def get_metrics_collector():
    return MetricsCollector()

def main():
    """Main dashboard function."""
    
    st.title("📊 RAG System Monitoring Dashboard")
    st.markdown("Real-time monitoring and analytics for the German Payroll Law Assistant")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        
        # Time range selection
        time_range = st.selectbox(
            "Time Range",
            options=[1, 7, 30, 90],
            index=1,
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
        )
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            st.empty()  # Placeholder for refresh timer
        
        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_resource.clear()
        
        st.divider()
        
        # Export options
        st.subheader("📤 Export Data")
        
        if st.button("Export as JSON"):
            collector = get_metrics_collector()
            filepath = collector.export_metrics('json')
            st.success(f"Exported to: {filepath}")
        
        if st.button("Export as CSV"):
            collector = get_metrics_collector()
            filepath = collector.export_metrics('csv')
            st.success(f"Exported to: {filepath}")
        
        st.divider()
        
        # Data cleanup
        st.subheader("🗑️ Data Management")
        
        if st.button("Cleanup Old Data (90+ days)"):
            collector = get_metrics_collector()
            old_counts, new_counts = collector.cleanup_old_data(90)
            st.success("Data cleanup completed!")
            st.json({"removed": {k: old_counts[k] - new_counts[k] for k in old_counts}})
    
    # Load metrics
    try:
        collector = get_metrics_collector()
        metrics = collector.get_metrics(days_back=time_range)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return
    
    # Auto-refresh mechanism
    if auto_refresh:
        import time
        time.sleep(1)  # Small delay to prevent too frequent refreshes
        st.rerun()
    
    # Key Metrics Overview
    st.header("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Queries Today</p>
        </div>
        """.format(metrics.get('queries_today', 0)), unsafe_allow_html=True)
    
    with col2:
        response_time = metrics.get('avg_response_time', 0)
        color = "green" if response_time < 3 else "orange" if response_time < 6 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color}">{response_time:.2f}s</h3>
            <p>Avg Response Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        success_rate = metrics.get('success_rate', 0)
        color = "green" if success_rate > 0.95 else "orange" if success_rate > 0.85 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color}">{success_rate:.1%}</h3>
            <p>Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        satisfaction = metrics.get('satisfaction_rate', 0)
        color = "green" if satisfaction > 0.8 else "orange" if satisfaction > 0.6 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color}">{satisfaction:.1%}</h3>
            <p>User Satisfaction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        health_score = metrics.get('system_health_score', 0)
        color = "green" if health_score > 80 else "orange" if health_score > 60 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color}">{health_score:.1f}</h3>
            <p>Health Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System Alerts
    st.header("🚨 System Alerts")
    
    alerts = []
    
    # Check for performance issues
    if metrics.get('avg_response_time', 0) > 5:
        alerts.append(("warning", f"High response time: {metrics['avg_response_time']:.2f}s"))
    
    # Check for error rate
    error_rate = metrics.get('error_rate', 0)
    if error_rate > 0.1:
        alerts.append(("error", f"High error rate: {error_rate:.1%}"))
    
    # Check for low satisfaction
    if metrics.get('satisfaction_rate', 1) < 0.6:
        alerts.append(("warning", f"Low user satisfaction: {metrics['satisfaction_rate']:.1%}"))
    
    # Check for system health
    if health_score < 70:
        alerts.append(("error", f"Low system health score: {health_score:.1f}"))
    
    if alerts:
        for alert_type, message in alerts:
            if alert_type == "error":
                st.markdown(f'<div class="alert-red">🚨 <strong>Error:</strong> {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-yellow">⚠️ <strong>Warning:</strong> {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-green">✅ <strong>All Systems Operating Normally</strong></div>', unsafe_allow_html=True)
    
    # Charts and Visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Query Volume Over Time
        st.subheader("📈 Query Volume Trends")
        
        daily_counts = metrics.get('daily_query_counts', {})
        if daily_counts:
            dates = list(daily_counts.keys())
            counts = list(daily_counts.values())
            
            fig = px.line(
                x=dates, 
                y=counts,
                title="Daily Query Volume",
                labels={'x': 'Date', 'y': 'Number of Queries'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No query data available for the selected time range")
        
        # Strategy Usage Distribution
        st.subheader("🎯 Strategy Usage")
        
        strategy_dist = metrics.get('strategy_distribution', {})
        if strategy_dist:
            fig = px.pie(
                values=list(strategy_dist.values()),
                names=list(strategy_dist.keys()),
                title="Prompt Strategy Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strategy data available")
    
    with col_right:
        # Response Time Distribution
        st.subheader("⏱️ Response Time Analysis")
        
        performance_trends = metrics.get('performance_trends', {})
        if performance_trends:
            dates = list(performance_trends.keys())
            response_times = [performance_trends[date]['avg_response_time'] for date in dates]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=response_times,
                mode='lines+markers',
                name='Avg Response Time',
                line=dict(color='blue')
            ))
            fig.add_hline(y=3, line_dash="dash", line_color="green", 
                         annotation_text="Target: 3s")
            fig.update_layout(
                title="Response Time Trends",
                xaxis_title="Date",
                yaxis_title="Response Time (seconds)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available")
        
        # User Feedback Distribution
        st.subheader("👍 User Feedback")
        
        feedback_dist = metrics.get('feedback_distribution', {})
        if feedback_dist:
            fig = px.bar(
                x=list(feedback_dist.keys()),
                y=list(feedback_dist.values()),
                title="Feedback Distribution",
                color=list(feedback_dist.keys()),
                color_discrete_map={
                    'positive': 'green',
                    'negative': 'red',
                    'neutral': 'gray'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feedback data available")
    
    # Detailed Analytics
    st.header("📊 Detailed Analytics")
    
    # Usage patterns
    col_pattern1, col_pattern2 = st.columns(2)
    
    with col_pattern1:
        st.subheader("🕐 Hourly Usage Pattern")
        
        hourly_dist = metrics.get('hourly_distribution', {})
        if hourly_dist:
            hours = list(range(24))
            counts = [hourly_dist.get(str(h), 0) for h in hours]
            
            fig = px.bar(
                x=hours,
                y=counts,
                title="Queries by Hour of Day",
                labels={'x': 'Hour', 'y': 'Number of Queries'}
            )
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly usage data available")
    
    with col_pattern2:
        st.subheader("🎯 Top Queries")
        
        try:
            top_queries = collector.get_top_queries(limit=10, days_back=time_range)
            if top_queries:
                df_queries = pd.DataFrame(top_queries)
                
                fig = px.bar(
                    df_queries,
                    x='count',
                    y='query',
                    orientation='h',
                    title="Most Frequent Queries",
                    labels={'count': 'Frequency', 'query': 'Query'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No query frequency data available")
        except Exception as e:
            st.error(f"Error loading top queries: {e}")
    
    # System Performance Table
    st.subheader("🔧 System Performance Details")
    
    perf_data = {
        'Metric': [
            'Total Queries',
            'Successful Queries',
            'Average Response Time (seconds)',
            'Average Tokens Used',
            'Average Documents Retrieved',
            'Total Feedback',
            'Positive Feedback Rate (%)',
            'Error Count',
            'Most Used Strategy'
        ],
        'Value': [
            str(metrics.get('total_queries', 0)),
            str(metrics.get('successful_queries', 0)),
            f"{metrics.get('avg_response_time', 0):.2f}",
            f"{metrics.get('avg_tokens', 0):.0f}",
            f"{metrics.get('avg_retrieved_docs', 0):.1f}",
            str(metrics.get('total_feedback', 0)),
            f"{metrics.get('satisfaction_rate', 0)*100:.1f}",
            str(metrics.get('total_errors', 0)),
            str(metrics.get('most_used_strategy', 'N/A'))
        ]
    }
    
    df_performance = pd.DataFrame(perf_data)
    st.dataframe(df_performance, use_container_width=True, hide_index=True)
    
    # Recent Activity
    st.subheader("🕒 Recent Activity")
    
    col_recent1, col_recent2 = st.columns(2)
    
    with col_recent1:
        st.write("**Recent Queries**")
        recent_queries = metrics.get('recent_queries', [])
        if recent_queries:
            for i, query in enumerate(recent_queries[-5:]):
                status = "✅" if query.get('success', True) else "❌"
                timestamp = datetime.fromisoformat(query['timestamp']).strftime('%H:%M:%S')
                st.caption(f"{status} {timestamp} - {query['query'][:60]}...")
        else:
            st.info("No recent queries")
    
    with col_recent2:
        st.write("**Recent Errors**")
        recent_errors = metrics.get('recent_errors', [])
        if recent_errors:
            for error in recent_errors[-5:]:
                timestamp = datetime.fromisoformat(error['timestamp']).strftime('%H:%M:%S')
                st.caption(f"🚨 {timestamp} - {error['error_type']}: {error['error_message'][:50]}...")
        else:
            st.info("No recent errors")
    
    # Footer
    st.divider()
    st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"Showing data for the last {time_range} day{'s' if time_range > 1 else ''}")

if __name__ == "__main__":
    main()
