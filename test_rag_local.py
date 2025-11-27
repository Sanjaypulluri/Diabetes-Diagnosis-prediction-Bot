"""
Local RAG Testing Script
Run directly in terminal for testing accuracy
No Streamlit required - Pure Python

Usage:
    python test_rag_local.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PyPDF2 import PdfReader
from typing import List, Dict
import uuid
import os
from datetime import datetime


class LocalRAGTester:
    """
    Simple RAG system for local testing
    Uses Qdrant in local/in-memory mode
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        qdrant_path: str = "./qdrant_local_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Initialize local RAG system"""
        
        print("🚀 Initializing Local RAG Tester...")
        
        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = "clinical_docs"
        
        # Initialize Qdrant (local mode)
        print("📦 Initializing Qdrant (local mode)...")
        self.qdrant_client = QdrantClient(path=qdrant_path)
        
        # Initialize Embedding Model
        print("🧠 Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create collection if doesn't exist
        self._setup_collection()
        
        # Initialize Gemini
        print("🤖 Connecting to Gemini...")
        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel('gemini-2.5-flash')
        
        print("✅ RAG System Ready!\n")
    
    
    def _setup_collection(self):
        """Setup Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {self.collection_name}")
            else:
                print(f"✅ Using existing collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Error setting up collection: {e}")
            raise
    
    
    def load_pdfs(self, pdf_paths: List[str]):
        """
        Load PDFs from file paths
        
        Args:
            pdf_paths: List of PDF file paths
        """
        print(f"\n📚 Loading {len(pdf_paths)} PDFs...")
        
        all_chunks = []
        all_metadata = []
        total_pages = 0
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"⚠️  File not found: {pdf_path}")
                continue
            
            print(f"\n📄 Processing: {os.path.basename(pdf_path)}")
            
            # Extract text
            text, page_count = self._extract_pdf_text(pdf_path)
            total_pages += page_count
            
            # Chunk text
            chunks = self._chunk_text(text)
            
            print(f"   ✓ {page_count} pages → {len(chunks)} chunks")
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': os.path.basename(pdf_path),
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'page_count': page_count
                })
        
        # Index to Qdrant
        if all_chunks:
            print(f"\n💾 Indexing {len(all_chunks)} chunks to Qdrant...")
            self._index_chunks(all_chunks, all_metadata)
            
            print(f"\n✅ Indexing Complete!")
            print(f"   - Total PDFs: {len(pdf_paths)}")
            print(f"   - Total Pages: {total_pages}")
            print(f"   - Total Chunks: {len(all_chunks)}")
        else:
            print("❌ No chunks to index!")
    
    
    def _extract_pdf_text(self, pdf_path: str) -> tuple:
        """Extract text from PDF"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                text += page.extract_text() + "\n\n"
                if (i + 1) % 50 == 0:
                    print(f"   - Extracted {i+1}/{total_pages} pages")
            
            return text, total_pages
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return "", 0
    
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    
    def _index_chunks(self, chunks: List[str], metadata: List[Dict], batch_size: int = 100):
        """Index chunks to Qdrant"""
        total = len(chunks)
        
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_meta = metadata[i:i + batch_size]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(batch_chunks).tolist()
            
            # Create points
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={'text': chunk, 'metadata': meta}
                )
                for chunk, meta, embedding in zip(batch_chunks, batch_meta, embeddings)
            ]
            
            # Upload
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            if (i + batch_size) % 500 == 0:
                print(f"   - Indexed {min(i + batch_size, total)}/{total} chunks")
    
    
    def query(self, question: str, k: int = 5) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            
        Returns:
            Dict with answer, sources, and metadata
        """
        print(f"\n🔍 Query: {question}")
        print("="*70)
        
        # Retrieve relevant chunks
        print(f"📖 Retrieving top {k} relevant chunks...")
        docs = self._retrieve(question, k)
        
        if not docs:
            return {
                'question': question,
                'answer': "No relevant documents found.",
                'sources': [],
                'num_sources': 0
            }
        
        print(f"✓ Found {len(docs)} relevant chunks")
        
        # Generate answer
        print("🤖 Generating answer with Gemini...")
        answer = self._generate_answer(question, docs)
        
        return {
            'question': question,
            'answer': answer,
            'sources': docs,
            'num_sources': len(docs)
        }
    
    
    def _retrieve(self, query: str, k: int) -> List[Dict]:
        """Retrieve relevant documents"""
        # Generate query embedding
        query_vector = self.embedding_model.encode([query])[0].tolist()
        # Search Qdrant using a compatible method for the installed client
        results = None

        def try_call(fn):
            try:
                return fn()
            except Exception:
                return None

        attempts = [
            lambda: getattr(self.qdrant_client, 'search')(collection_name=self.collection_name, query_vector=query_vector, limit=k),
            lambda: getattr(self.qdrant_client, 'search')(collection_name=self.collection_name, query_vector=query_vector, limit=k, with_payload=True, with_vectors=False),
            lambda: getattr(self.qdrant_client, 'query_points')(collection_name=self.collection_name, query=query_vector, limit=k),
            lambda: getattr(self.qdrant_client, 'search_points')(collection_name=self.collection_name, vector=query_vector, limit=k),
        ]

        for attempt in attempts:
            results = try_call(attempt)
            if results:
                break

        if results is None:
            raise RuntimeError('No compatible search/query method found on QdrantClient')

        # Normalize responses (some return an object with `.points`)
        if hasattr(results, 'points'):
            results = results.points

        docs = []
        for result in results:
            payload = getattr(result, 'payload', result)
            score = getattr(result, 'score', None)

            if isinstance(payload, dict):
                text = payload.get('text')
                metadata = payload.get('metadata')
            else:
                text = getattr(payload, 'text', None)
                metadata = getattr(payload, 'metadata', None)

            docs.append({'text': text, 'metadata': metadata, 'score': score})

        return docs
    
    
    def _generate_answer(self, question: str, context: List[Dict]) -> str:
        """Generate answer using Gemini"""
        
        # Build context
        context_text = ""
        for i, doc in enumerate(context, 1):
            source = doc['metadata']['source']
            score = doc['score']
            context_text += f"\n[Source {i}: {source} | Relevance: {score:.3f}]\n{doc['text']}\n"
        
        # Create prompt
        prompt = f"""You are a medical assistant. Answer based ONLY on the provided context.

Context:
{context_text}

Question: {question}

Instructions:
- Answer clearly and accurately
- Cite sources (e.g., "According to Source 1...")
- If context is insufficient, state that clearly
- Use medical terminology appropriately

Answer:"""
        
        try:
            response = self.gemini.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"
    
    
    def interactive_mode(self):
        """Interactive Q&A mode"""
        print("\n" + "="*70)
        print("🎯 INTERACTIVE MODE")
        print("="*70)
        print("Type your questions below. Type 'exit' to quit.\n")
        
        while True:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Query
            result = self.query(question, k=5)
            
            # Display answer
            print(f"\n💡 Answer:\n{result['answer']}\n")
            
            # Display sources
            print(f"📚 Sources ({result['num_sources']}):")
            for i, src in enumerate(result['sources'], 1):
                print(f"   {i}. {src['metadata']['source']} (score: {src['score']:.3f})")
                print(f"      {src['text'][:150]}...\n")
            
            print("="*70 + "\n")
    
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            collection = self.qdrant_client.get_collection(self.collection_name)
            return {
                'total_chunks': collection.points_count,
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedding_dim
            }
        except:
            return {'total_chunks': 0}
    
    
    def clear_database(self):
        """Clear all data"""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._setup_collection()
            print("✅ Database cleared!")
        except Exception as e:
            print(f"❌ Error clearing database: {e}")

    def close(self):
        """Safely close the Qdrant client"""
        try:
            if hasattr(self, 'qdrant_client') and self.qdrant_client:
                try:
                    self.qdrant_client.close()
                except Exception:
                    pass
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# MAIN TESTING SCRIPT
# =============================================================================

def main():
    """Main testing function"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         Clinical RAG System - Local Testing               ║
    ║         Qdrant Local + Gemini                             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    GEMINI_API_KEY = "REMOVED_KEY"  # Replace with your key
    PDF_PATHS = [
        "./pdfs/allchapters.pdf",
        "./pdfs/codebook15_llcp.pdf",
        "./pdfs/Dietary Advice For Individuals with Diabetes - Endotext - NCBI Bookshelf.pdf",
        "./pdfs/National Diabetes Statistics Report_Diabetes_CDC.pdf",
    ]
    
    # Initialize RAG
    rag = LocalRAGTester(
        gemini_api_key=GEMINI_API_KEY,
        qdrant_path="./qdrant_local_db",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Check if we need to load PDFs
    stats = rag.get_stats()
    
    if stats['total_chunks'] == 0:
        print("\n📚 No documents in database. Loading PDFs...")
        
        # Check which PDFs exist
        existing_pdfs = [p for p in PDF_PATHS if os.path.exists(p)]
        
        if not existing_pdfs:
            print("\n⚠️  No PDF files found!")
            print("Please place your PDFs in the ./pdfs/ folder or update PDF_PATHS")
            return
        
        rag.load_pdfs(existing_pdfs)
    else:
        print(f"\n✅ Database loaded: {stats['total_chunks']} chunks available")
    
    # Test mode selection
    print("\n" + "="*70)
    print("Select Testing Mode:")
    print("  1. Interactive Q&A (type questions)")
    print("  2. Run predefined test cases")
    print("  3. Single query test")
    print("="*70)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Interactive mode
        rag.interactive_mode()
    
    elif choice == "2":
        # Predefined test cases
        run_test_cases(rag)
    
    elif choice == "3":
        # Single query
        question = input("\nEnter your question: ").strip()
        result = rag.query(question, k=5)
        
        print(f"\n💡 Answer:\n{result['answer']}\n")
        print(f"📚 Sources ({result['num_sources']}):")
        for i, src in enumerate(result['sources'], 1):
            print(f"\n{i}. {src['metadata']['source']} (score: {src['score']:.3f})")
            print(f"   {src['text'][:200]}...")


def run_test_cases(rag):
    """Run predefined test cases"""
    
    print("\n" + "="*70)
    print("🧪 RUNNING TEST CASES")
    print("="*70)
    
    test_cases = [
        {
            "question": "What is the treatment for diabetes?",
            "expected_keywords": ["insulin", "metformin", "glucose", "medication"]
        },
        {
            "question": "What are the symptoms of hypertension?",
            "expected_keywords": ["blood pressure", "headache", "dizziness"]
        },
        {
            "question": "What is the normal range for cholesterol?",
            "expected_keywords": ["LDL", "HDL", "cholesterol", "mg/dL"]
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"{'─'*70}")
        
        # Run query
        result = rag.query(test['question'], k=5)
        
        # Check keywords
        answer_lower = result['answer'].lower()
        found_keywords = [kw for kw in test['expected_keywords'] 
                         if kw.lower() in answer_lower]
        
        # Display results
        print(f"\n❓ Question: {test['question']}")
        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n✓ Keywords found: {found_keywords} ({len(found_keywords)}/{len(test['expected_keywords'])})")
        print(f"✓ Sources used: {result['num_sources']}")
        print(f"✓ Top source: {result['sources'][0]['metadata']['source'] if result['sources'] else 'None'}")
        
        # Show top source
        if result['sources']:
            top_src = result['sources'][0]
            print(f"✓ Relevance score: {top_src['score']:.3f}")
            print(f"\n📄 Top source text:\n{top_src['text'][:300]}...")
        
        results.append({
            'question': test['question'],
            'keywords_found': len(found_keywords),
            'keywords_total': len(test['expected_keywords']),
            'num_sources': result['num_sources'],
            'top_score': result['sources'][0]['score'] if result['sources'] else 0
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    
    avg_keyword_rate = sum(r['keywords_found']/r['keywords_total'] for r in results) / len(results)
    avg_score = sum(r['top_score'] for r in results) / len(results)
    
    print(f"Total tests: {len(test_cases)}")
    print(f"Average keyword match rate: {avg_keyword_rate*100:.1f}%")
    print(f"Average relevance score: {avg_score:.3f}")
    print(f"\nTarget metrics:")
    print(f"  - Keyword match rate: >70% ✓" if avg_keyword_rate > 0.7 else f"  - Keyword match rate: >70% ✗")
    print(f"  - Relevance score: >0.7 ✓" if avg_score > 0.7 else f"  - Relevance score: >0.7 ✗")


if __name__ == "__main__":
    main()