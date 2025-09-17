import streamlit as st
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.llm_client import LLMClient
from src.llm.prompt_strategies import PromptStrategy
from src.monitoring.metrics_collector import MetricsCollector
from config import settings

# Page configuration
st.set_page_config(
    page_title="German Payroll Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Only fix text areas and inputs for question/answer visibility */
    .stTextArea textarea, .stTextInput input {
        color: #1f2937 !important;
        background-color: white !important;
        border: 1px solid #d1d5db !important;
        caret-color: #1f2937 !important;
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .question-box {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        color: #1f2937;
    }
    .answer-box {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e0f2fe;
        margin: 1rem 0;
        color: #1f2937;
    }
    .source-box {
        background-color: #fafaf9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #78716c;
        margin: 0.5rem 0;
        font-size: 0.9em;
        color: #1f2937 !important;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .feedback-section {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'metrics_collector' not in st.session_state:
    st.session_state.metrics_collector = MetricsCollector()

if 'retriever' not in st.session_state:
    st.session_state.retriever_loading = True
    
    # Show loading indicator
    loading_placeholder = st.empty()
    loading_placeholder.info("🔄 Loading retrieval system... Please wait.")
    
    try:
        st.session_state.retriever = HybridRetriever()
        st.session_state.retriever_error = None
        loading_placeholder.success("✅ Retrieval system loaded successfully!")
        # Clear the message after a short delay
        import time
        time.sleep(1)
        loading_placeholder.empty()
    except Exception as e:
        st.session_state.retriever = None
        st.session_state.retriever_error = str(e)
        loading_placeholder.error(f"❌ Failed to load retrieval system: {e}")
    
    st.session_state.retriever_loading = False

if 'llm_client' not in st.session_state:
    try:
        st.session_state.llm_client = LLMClient()
    except Exception as e:
        st.session_state.llm_client = None
        st.session_state.llm_error = str(e)

def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ German Payroll Law Assistant</h1>
        <p>AI-powered assistant for German payroll law questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Check system status
        st.subheader("System Status")
        
        if st.session_state.retriever:
            st.success("✅ Retrieval System")
        else:
            st.error("❌ Retrieval System")
            if hasattr(st.session_state, 'retriever_error') and st.session_state.retriever_error:
                st.error(f"Error: {st.session_state.retriever_error}")
        
        if st.session_state.llm_client:
            st.success("✅ LLM Client")
        else:
            st.error("❌ LLM Client")
            if hasattr(st.session_state, 'llm_error'):
                st.error(f"Error: {st.session_state.llm_error}")
        
        st.divider()
        
        # Configuration options
        st.subheader("Configuration")
        
        # Prompt strategy selection
        prompt_strategy = st.selectbox(
            "Prompt Strategy",
            options=[strategy.value for strategy in PromptStrategy],
            index=1,  # Default to STRUCTURED
            help="Choose the prompt strategy for generating answers"
        )
        
        # Retrieval settings
        use_hybrid = st.checkbox("Use Hybrid Retrieval", value=True)
        use_reranking = st.checkbox("Use Re-ranking", value=True)
        top_k = st.slider("Number of Documents to Retrieve", 1, 10, 5)
        
        st.divider()
        
        # Collection statistics
        if st.session_state.retriever:
            st.subheader("Collection Statistics")
            try:
                stats = st.session_state.retriever.get_retrieval_stats()
                if stats.get('collection_count'):
                    st.metric("Documents", stats['collection_count'])
                st.metric("Cached Documents", stats.get('documents_cached', 0))
            except Exception as e:
                st.error(f"Could not load stats: {e}")
        
        st.divider()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question input
        st.subheader("💬 Ask Your Question")
        
        question = st.text_area(
            "Enter your question about German payroll law:",
            placeholder="Beispiel: Wie werden Überstunden in Deutschland besteuert?",
            height=100,
            key="question_input"
        )
        
        col_ask, col_examples = st.columns([1, 1])
        
        with col_ask:
            ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
        
        with col_examples:
            if st.button("💡 Show Examples", use_container_width=True):
                st.session_state.show_examples = not getattr(st.session_state, 'show_examples', False)
        
        # Example questions
        if getattr(st.session_state, 'show_examples', False):
            st.subheader("Example Questions")
            example_questions = [
                "Wie werden Überstunden in Deutschland besteuert?",
                "Was sind Sozialversicherungsbeiträge?",
                "Wie funktioniert die Lohnabrechnung?",
                "Was ist die Lohnsteuerkarte?",
                "Welche Steuerfreibeträge gibt es für Arbeitnehmer?",
                "Wie wird das 13. Monatsgehalt versteuert?",
                "Was sind die Unterschiede zwischen Brutto- und Nettolohn?"
            ]
            
            for example in example_questions:
                if st.button(f"📋 {example}", key=f"example_{hash(example)}"):
                    st.session_state.question_input = example
                    st.rerun()
        
        # Process question
        if ask_button and question.strip():
            if not st.session_state.retriever or not st.session_state.llm_client:
                st.error("System not properly initialized. Please check the configuration.")
                return
            
            with st.spinner("Processing your question..."):
                try:
                    # Record start time
                    start_time = time.time()
                    
                    # Retrieve relevant documents
                    retrieved_docs = st.session_state.retriever.retrieve(
                        query=question,
                        use_hybrid=use_hybrid,
                        use_reranking=use_reranking,
                        top_k=top_k
                    )
                    
                    retrieval_time = time.time() - start_time
                    
                    # Generate answer
                    answer_start_time = time.time()
                    
                    selected_strategy = PromptStrategy(prompt_strategy)
                    result = st.session_state.llm_client.generate_answer(
                        query=question,
                        retrieved_documents=retrieved_docs,
                        strategy=selected_strategy
                    )
                    
                    answer_time = time.time() - answer_start_time
                    total_time = time.time() - start_time
                    
                    if result.get('success', False):
                        # Store in conversation history
                        conversation_item = {
                            'question': question,
                            'answer': result['answer'],
                            'retrieved_docs': retrieved_docs,
                            'strategy': prompt_strategy,
                            'timestamp': datetime.now().isoformat(),
                            'metrics': {
                                'retrieval_time': retrieval_time,
                                'answer_time': answer_time,
                                'total_time': total_time,
                                'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                                'docs_retrieved': len(retrieved_docs)
                            }
                        }
                        
                        st.session_state.conversation_history.append(conversation_item)
                        
                        # Collect metrics
                        st.session_state.metrics_collector.record_query(
                            query=question,
                            response_time=total_time,
                            tokens_used=result.get('usage', {}).get('total_tokens', 0),
                            retrieved_docs_count=len(retrieved_docs),
                            strategy=prompt_strategy
                        )
                        
                        st.success("✅ Answer generated successfully!")
                        
                    else:
                        st.error(f"❌ Error generating answer: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("📜 Conversation History")
            
            for i, item in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q: {item['question'][:100]}..." if len(item['question']) > 100 else f"Q: {item['question']}", expanded=(i == 0)):
                    
                    # Question
                    st.markdown(f"""
                    <div class="question-box">
                        <strong>🤔 Your Question:</strong><br>
                        {item['question']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Answer
                    st.markdown(f"""
                    <div class="answer-box">
                        <strong>🤖 Assistant's Answer:</strong><br>
                        {item['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    col_metrics = st.columns(4)
                    with col_metrics[0]:
                        st.metric("Response Time", f"{item['metrics']['total_time']:.2f}s")
                    with col_metrics[1]:
                        st.metric("Tokens Used", item['metrics']['tokens_used'])
                    with col_metrics[2]:
                        st.metric("Documents Retrieved", item['metrics']['docs_retrieved'])
                    with col_metrics[3]:
                        st.metric("Strategy", item['strategy'])
                    
                    # Source documents
                    if st.checkbox(f"Show Source Documents", key=f"sources_{i}"):
                        st.markdown("**📚 Source Documents:**")
                        for j, doc in enumerate(item['retrieved_docs'][:3]):  # Show top 3
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Document {j+1}</strong> (Score: {doc.get('combined_score', doc.get('score', 0)):.3f})<br>
                                {doc['content'][:300]}{'...' if len(doc['content']) > 300 else ''}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Feedback section
                    st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
                    st.markdown("**Was this answer helpful?**")
                    col_feedback = st.columns(3)
                    
                    with col_feedback[0]:
                        if st.button("👍 Yes", key=f"thumbs_up_{i}"):
                            st.session_state.metrics_collector.record_feedback(
                                query=item['question'],
                                feedback_type="positive",
                                timestamp=item['timestamp']
                            )
                            st.success("Thank you for your feedback!")
                    
                    with col_feedback[1]:
                        if st.button("👎 No", key=f"thumbs_down_{i}"):
                            st.session_state.metrics_collector.record_feedback(
                                query=item['question'],
                                feedback_type="negative",
                                timestamp=item['timestamp']
                            )
                            st.info("Thank you for your feedback. We'll work to improve!")
                    
                    with col_feedback[2]:
                        if st.button("💬 Report Issue", key=f"report_{i}"):
                            st.text_area("Describe the issue:", key=f"issue_desc_{i}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # System metrics panel
        st.subheader("📊 System Metrics")
        
        try:
            metrics = st.session_state.metrics_collector.get_metrics()
            
            # Key metrics
            st.metric("Total Queries Today", metrics.get('queries_today', 0))
            st.metric("Average Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
            st.metric("User Satisfaction", f"{metrics.get('satisfaction_rate', 0):.1%}")
            
            # Recent activity
            st.subheader("🕒 Recent Activity")
            recent_queries = metrics.get('recent_queries', [])
            if recent_queries:
                for query in recent_queries[-5:]:  # Show last 5
                    st.caption(f"• {query['query'][:50]}...")
            else:
                st.caption("No recent activity")
            
            # Quick stats
            st.subheader("📈 Quick Stats")
            stats_data = {
                "Most Common Strategy": metrics.get('most_used_strategy', 'N/A'),
                "Average Tokens per Query": f"{metrics.get('avg_tokens', 0):.0f}",
                "Peak Usage Hour": f"{metrics.get('peak_hour', 'N/A')}:00",
                "Total Documents Processed": metrics.get('total_docs_processed', 'N/A')
            }
            
            for key, value in stats_data.items():
                st.metric(key, value)
        
        except Exception as e:
            st.error(f"Could not load metrics: {e}")
        
        # System health
        st.subheader("💚 System Health")
        
        health_checks = [
            ("Vector Database", st.session_state.retriever is not None),
            ("LLM Service", st.session_state.llm_client is not None),
            ("Metrics Collection", True),
            ("Document Processing", True)
        ]
        
        for check_name, status in health_checks:
            if status:
                st.success(f"✅ {check_name}")
            else:
                st.error(f"❌ {check_name}")

if __name__ == "__main__":
    main()
