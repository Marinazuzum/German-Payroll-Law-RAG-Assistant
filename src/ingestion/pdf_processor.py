import os
import json
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import hashlib
from tqdm import tqdm
import logging

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents and store embeddings in ChromaDB."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(allow_reset=True)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="german_payroll_law",
            metadata={"description": "German payroll law documents"}
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_document(self, text: str, source: str) -> List[Document]:
        """Split document into chunks."""
        documents = [Document(page_content=text, metadata={"source": source})]
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk.page_content)
            })
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def create_document_hash(self, pdf_path: str) -> str:
        """Create hash of PDF file for tracking changes."""
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def is_document_processed(self, pdf_path: str) -> bool:
        """Check if document is already processed."""
        doc_hash = self.create_document_hash(pdf_path)
        
        # Check if hash exists in collection metadata
        try:
            results = self.collection.query(
                query_texts=[""],
                n_results=1,
                where={"document_hash": doc_hash}
            )
            return len(results['ids'][0]) > 0
        except:
            return False
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file."""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Check if already processed
        if self.is_document_processed(pdf_path):
            logger.info(f"Document {pdf_path} already processed, skipping...")
            return {"status": "skipped", "reason": "already_processed"}
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"status": "failed", "reason": "no_text_extracted"}
        
        # Create chunks
        chunks = self.chunk_document(text, pdf_path)
        if not chunks:
            return {"status": "failed", "reason": "no_chunks_created"}
        
        # Generate embeddings
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = self.generate_embeddings(chunk_texts)
        
        # Prepare data for ChromaDB
        doc_hash = self.create_document_hash(pdf_path)
        ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
        
        metadatas = []
        for chunk in chunks:
            metadata = chunk.metadata.copy()
            metadata["document_hash"] = doc_hash
            metadata["file_name"] = Path(pdf_path).name
            metadatas.append(metadata)
        
        # Store in ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully processed {len(chunks)} chunks from {pdf_path}")
            return {
                "status": "success",
                "chunks_processed": len(chunks),
                "document_hash": doc_hash
            }
            
        except Exception as e:
            logger.error(f"Error storing chunks in ChromaDB: {e}")
            return {"status": "failed", "reason": f"storage_error: {e}"}
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """Process all PDF files in a directory."""
        logger.info(f"Processing PDFs in directory: {directory_path}")
        
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return {"processed": 0, "failed": 0, "skipped": 0, "results": []}
        
        results = []
        processed = 0
        failed = 0
        skipped = 0
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            result = self.process_pdf(str(pdf_file))
            results.append({"file": str(pdf_file), "result": result})
            
            if result["status"] == "success":
                processed += 1
            elif result["status"] == "failed":
                failed += 1
            elif result["status"] == "skipped":
                skipped += 1
        
        summary = {
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "total_files": len(pdf_files),
            "results": results
        }
        
        logger.info(f"Processing complete: {processed} processed, {failed} failed, {skipped} skipped")
        return summary
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            count = self.collection.count()
            
            # Get sample documents to analyze
            if count > 0:
                sample_results = self.collection.query(
                    query_texts=[""],
                    n_results=min(10, count)
                )
                
                unique_sources = set()
                chunk_sizes = []
                
                for metadata in sample_results['metadatas'][0]:
                    unique_sources.add(metadata.get('source', 'unknown'))
                    chunk_sizes.append(metadata.get('chunk_size', 0))
                
                avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
                
                return {
                    "total_chunks": count,
                    "unique_documents": len(unique_sources),
                    "average_chunk_size": avg_chunk_size,
                    "sample_sources": list(unique_sources)
                }
            else:
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "average_chunk_size": 0,
                    "sample_sources": []
                }
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

def main():
    """Main function for running PDF processing."""
    processor = PDFProcessor()
    
    # Process PDFs in the configured directory
    if Path(settings.pdf_data_path).exists():
        results = processor.process_directory(settings.pdf_data_path)
        
        # Save processing results
        results_file = Path(settings.processed_data_path) / "processing_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Get collection statistics
        stats = processor.get_collection_stats()
        logger.info(f"Collection stats: {stats}")
        
        # Save stats
        stats_file = Path(settings.processed_data_path) / "collection_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
    else:
        logger.error(f"PDF data directory not found: {settings.pdf_data_path}")

if __name__ == "__main__":
    main()
