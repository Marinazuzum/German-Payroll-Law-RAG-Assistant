#!/bin/bash

# German Payroll Law RAG Assistant - Setup Script
# This script sets up the development environment

set -e  # Exit on any error

echo "🚀 Setting up German Payroll Law RAG Assistant..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.11+ is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.9+ required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        print_success "Docker found"
        
        if command -v docker-compose &> /dev/null; then
            print_success "Docker Compose found"
        else
            print_warning "Docker Compose not found. Installing via pip..."
            pip3 install docker-compose
        fi
    else
        print_warning "Docker not found. Please install Docker for containerized deployment."
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Download NLTK data
setup_nltk() {
    print_status "Setting up NLTK data..."
    
    python3 -c "
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Error downloading NLTK data: {e}')
"
    print_success "NLTK data setup complete"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/chroma_db
    mkdir -p logs
    
    print_success "Directory structure created"
}

# Create environment file template
create_env_template() {
    print_status "Creating environment configuration..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Data Paths
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
PDF_DATA_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed

# Model Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
TEMPERATURE=0.1
MAX_TOKENS=1000

# Retrieval Configuration
TOP_K_RETRIEVAL=5
RERANK_TOP_K=3

# Monitoring
ENABLE_MONITORING=true
METRICS_FILE=./data/metrics.json
EOF
        print_success "Environment file created (.env)"
        print_warning "Please edit .env file and add your OpenAI API key"
    else
        print_status "Environment file already exists"
    fi
}

# Test basic functionality
test_setup() {
    print_status "Testing basic setup..."
    
    # Test imports
    python3 -c "
import sys
sys.path.append('.')

try:
    from config import settings
    print('✓ Config module loaded')
    
    import streamlit
    print('✓ Streamlit imported')
    
    import chromadb
    print('✓ ChromaDB imported')
    
    import openai
    print('✓ OpenAI imported')
    
    from sentence_transformers import SentenceTransformer
    print('✓ SentenceTransformers imported')
    
    print('All imports successful!')
    
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Basic functionality test passed"
    else
        print_error "Setup test failed"
        exit 1
    fi
}

# Main setup process
main() {
    echo "🎯 German Payroll Law RAG Assistant Setup"
    echo "=========================================="
    
    check_python
    check_docker
    setup_venv
    install_dependencies
    setup_nltk
    create_directories
    create_env_template
    test_setup
    
    echo ""
    echo "=========================================="
    print_success "Setup completed successfully! 🎉"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file and add your OpenAI API key"
    echo "2. Add PDF documents to data/raw/ directory"
    echo "3. Run the application:"
    echo "   - Development: source venv/bin/activate && streamlit run app/main.py"
    echo "   - Docker: docker-compose up"
    echo ""
    echo "Useful commands:"
    echo "- Process PDFs: python src/ingestion/pdf_processor.py"
    echo "- Run tests: python -m pytest"
    echo "- View monitoring: streamlit run app/monitoring_dashboard.py"
    echo ""
}

# Run main function
main "$@"
