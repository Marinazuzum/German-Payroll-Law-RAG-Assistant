# Setup Guide

This guide provides detailed instructions for setting up the German Payroll Law RAG Assistant.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows 10/11
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for documents and models
- **Internet**: Stable connection for API calls and model downloads

### Software Requirements
- **Python**: 3.11 or higher
- **Docker**: Latest version (optional but recommended)
- **Git**: For cloning the repository

### API Requirements
- **OpenAI API Key**: Required for LLM functionality
  - Sign up at [OpenAI Platform](https://platform.openai.com/)
  - Generate an API key from the dashboard
  - Ensure sufficient credits for usage

## 🚀 Installation Methods

### Method 1: Docker Setup (Recommended)

This is the easiest method for getting started quickly.

#### Step 1: Clone Repository
```bash
git clone <repository-url>
cd German-Payroll-Law-RAG-Assistant
```

#### Step 2: Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit the .env file with your settings
nano .env  # or use your preferred editor
```

Add your OpenAI API key to the `.env` file:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

#### Step 3: Start Services
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

#### Step 4: Verify Installation
- Main App: http://localhost:8501
- Monitoring: http://localhost:8502
- ChromaDB: http://localhost:8000

### Method 2: Manual Python Setup

For development or when Docker isn't available.

#### Step 1: Clone and Setup
```bash
git clone <repository-url>
cd German-Payroll-Law-RAG-Assistant
chmod +x scripts/setup.sh
./scripts/setup.sh
```

#### Step 2: Manual Configuration
If the setup script doesn't work, follow these manual steps:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create directories
mkdir -p data/raw data/processed data/chroma_db

# Create environment file
cp .env.example .env
```

#### Step 3: Configure Environment
Edit the `.env` file with your settings:
```env
OPENAI_API_KEY=your_api_key_here
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
PDF_DATA_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed
```

#### Step 4: Start Application
```bash
# Activate virtual environment
source venv/bin/activate

# Start main application
streamlit run app/main.py

# Or start monitoring dashboard
streamlit run app/monitoring_dashboard.py --server.port 8502
```

## 📄 Adding Documents

### Step 1: Prepare PDF Documents
Place your German payroll law PDF documents in the `data/raw/` directory:

```bash
# Example structure
data/raw/
├── einkommensteuergesetz.pdf
├── sozialgesetzbuch.pdf
├── arbeitsrecht-grundlagen.pdf
└── lohnsteuer-richtlinien.pdf
```

### Step 2: Process Documents

#### Using Docker
```bash
# Run document processing
docker-compose run pdf-processor

# Or run specific processing job
docker-compose exec rag-app python src/ingestion/pdf_processor.py
```

#### Manual Processing
```bash
# Activate environment
source venv/bin/activate

# Process documents
python src/ingestion/pdf_processor.py
```

### Step 3: Verify Processing
Check the processing results:
- View logs for any errors
- Check `data/processed/` for output files
- Verify documents in ChromaDB collection

## ⚙️ Configuration Options

### Basic Configuration
Essential settings in `.env`:

```env
# API Configuration
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-3.5-turbo  # or gpt-4 for better quality

# Performance Settings
CHUNK_SIZE=1000          # Adjust based on document complexity
CHUNK_OVERLAP=200        # Overlap between chunks
TOP_K_RETRIEVAL=5        # Number of documents to retrieve
TEMPERATURE=0.1          # LLM creativity (0-1)
```

### Advanced Configuration
Fine-tuning options:

```env
# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval Settings
RERANK_TOP_K=3          # Documents after re-ranking
VECTOR_WEIGHT=0.5       # Weight for vector search
BM25_WEIGHT=0.3         # Weight for BM25 search
TFIDF_WEIGHT=0.2        # Weight for TF-IDF search

# Monitoring
ENABLE_MONITORING=true
METRICS_FILE=./data/metrics.json
```

### Docker Configuration
Customize `docker-compose.yml` for your environment:

```yaml
services:
  rag-app:
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHUNK_SIZE=1500  # Custom chunk size
    ports:
      - "8501:8501"     # Change external port if needed
```

## 🧪 Testing the Setup

### Basic Functionality Test
```bash
# Test imports and basic functionality
python -c "
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.llm_client import LLMClient
print('✅ All components imported successfully')
"
```

### End-to-End Test
```bash
# Run a test query
python -c "
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.llm_client import LLMClient
from src.llm.prompt_strategies import PromptStrategy

retriever = HybridRetriever()
stats = retriever.get_retrieval_stats()
print(f'Collection available: {stats[\"collection_available\"]}')
print(f'Documents cached: {stats[\"documents_cached\"]}')
"
```

### Web Interface Test
1. Navigate to http://localhost:8501
2. Enter a test question: "Was ist Lohnsteuer?"
3. Verify the system responds with an answer
4. Check that sources are displayed

## 🔧 Troubleshooting

### Common Setup Issues

#### Python Version Issues
```bash
# Check Python version
python3 --version

# If version is too old, install Python 3.11+
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.11 python3.11-venv

# On macOS with Homebrew:
brew install python@3.11
```

#### Virtual Environment Issues
```bash
# Remove and recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Docker Issues
```bash
# Check Docker status
docker --version
docker-compose --version

# Restart Docker service
sudo systemctl restart docker  # Linux
# Or restart Docker Desktop on macOS/Windows

# Clean up Docker resources
docker-compose down
docker system prune -f
docker-compose up -d
```

#### ChromaDB Connection Issues
```bash
# Check ChromaDB logs
docker-compose logs chromadb

# Reset ChromaDB data
docker-compose down
rm -rf data/chroma_db/*
docker-compose up -d
```

#### OpenAI API Issues
```bash
# Test API key
python -c "
import openai
openai.api_key = 'your_key_here'
try:
    response = openai.Model.list()
    print('✅ OpenAI API key is valid')
except Exception as e:
    print(f'❌ API key error: {e}')
"
```

### Performance Issues

#### Memory Issues
- Reduce `CHUNK_SIZE` in configuration
- Limit concurrent document processing
- Increase Docker memory allocation

#### Slow Response Times
- Use `gpt-3.5-turbo` instead of `gpt-4`
- Reduce `TOP_K_RETRIEVAL` value
- Enable response caching

#### Network Issues
- Check firewall settings for ports 8501, 8502, 8000
- Verify internet connectivity for API calls
- Consider using a proxy if behind corporate firewall

## 📊 Monitoring Setup

### Enable Monitoring
Ensure monitoring is configured in `.env`:
```env
ENABLE_MONITORING=true
METRICS_FILE=./data/metrics.json
```

### Access Monitoring Dashboard
- URL: http://localhost:8502
- Features: Real-time metrics, system health, user feedback
- Data: Automatically collected during application usage

### Custom Metrics
Add custom monitoring by modifying `src/monitoring/metrics_collector.py`:
```python
# Example: Add custom performance metric
collector.record_performance_metric(
    metric_name="custom_response_time",
    value=response_time,
    unit="seconds"
)
```

## 🔄 Maintenance

### Regular Tasks
```bash
# Update Python dependencies
pip install --upgrade -r requirements.txt

# Clean old monitoring data (keeps last 90 days)
python -c "
from src.monitoring.metrics_collector import MetricsCollector
collector = MetricsCollector()
collector.cleanup_old_data(90)
"

# Backup vector database
cp -r data/chroma_db data/chroma_db_backup_$(date +%Y%m%d)
```

### Updating the System
```bash
# Pull latest changes
git pull origin main

# Rebuild Docker images
docker-compose build --no-cache

# Restart services
docker-compose down
docker-compose up -d
```

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs**: `docker-compose logs` or application console output
2. **Review configuration**: Ensure all environment variables are set correctly
3. **Test components individually**: Use the test scripts provided
4. **Consult the troubleshooting section**: Common issues and solutions
5. **Create an issue**: If the problem persists, create a GitHub issue with:
   - System information (OS, Python version, Docker version)
   - Error messages and logs
   - Steps to reproduce the issue
   - Configuration details (without sensitive information)

## ✅ Setup Checklist

- [ ] Python 3.11+ installed
- [ ] Docker installed (if using Docker method)
- [ ] Repository cloned
- [ ] OpenAI API key obtained
- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Application starts successfully
- [ ] Documents processed (if applicable)
- [ ] Test query works
- [ ] Monitoring dashboard accessible
- [ ] All health checks pass

Once all items are checked, your German Payroll Law RAG Assistant is ready to use!
