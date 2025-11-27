"""
Optimized RAG System with:
- Smart rechunking (different sizes for textbook vs specialized PDFs)
- MMR (Maximal Marginal Relevance) for diversity
- Source balancing and similarity threshold
- Reduced K with higher quality results

Usage:
    python optimized_rag.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PyPDF2 import PdfReader
from typing import List, Dict, Tuple
import numpy as np
import uuid
import os
from datetime import datetime


class OptimizedRAG:
    """
    Optimized RAG system with smart chunking and advanced retrieval
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        qdrant_path: str = "./qdrant_optimized_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        textbook_chunk_size: int = 300,      # Smaller for textbook
        specialized_chunk_size: int = 500,    # Larger for focused PDFs
        chunk_overlap: int = 75,
        similarity_threshold: float = 0.70,   # Minimum relevance score
        mmr_lambda: float = 0.6               # Balance relevance vs diversity
    ):
        """Initialize optimized RAG system"""
        
        print("🚀 Initializing Optimized RAG System...")
        
        # Configuration
        self.textbook_chunk_size = textbook_chunk_size
        self.specialized_chunk_size = specialized_chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.mmr_lambda = mmr_lambda
        self.collection_name = "clinical_docs_optimized"
        
        # Textbook identifier (adjust based on your filename)
        self.textbook_name = "allchapters.pdf"
        
        # Initialize Qdrant
        print("📦 Initializing Qdrant...")
        self.qdrant_client = QdrantClient(path=qdrant_path)
        
        # Initialize Embedding Model
        print("🧠 Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create collection
        self._setup_collection()
        
        # Initialize Gemini
        print("🤖 Connecting to Gemini...")
        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel('gemini-2.5-flash')
        
        print("✅ Optimized RAG System Ready!")
        print(f"   - Textbook chunk size: {textbook_chunk_size} words")
        print(f"   - Specialized chunk size: {specialized_chunk_size} words")
        print(f"   - Similarity threshold: {similarity_threshold}")
        print(f"   - MMR lambda: {mmr_lambda}\n")
    
    
    def _setup_collection(self):
        """Setup Qdrant collection"""
        try:
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
        Load PDFs with smart chunking strategy
        
        Args:
            pdf_paths: List of PDF file paths
        """
        print(f"\n📚 Loading {len(pdf_paths)} PDFs with optimized chunking...")
        
        all_chunks = []
        all_metadata = []
        total_pages = 0
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"⚠️  File not found: {pdf_path}")
                continue
            
            filename = os.path.basename(pdf_path)
            print(f"\n📄 Processing: {filename}")
            
            # Determine if this is the textbook
            is_textbook = self.textbook_name.lower() in filename.lower()
            chunk_size = self.textbook_chunk_size if is_textbook else self.specialized_chunk_size
            
            doc_type = "TEXTBOOK (small chunks)" if is_textbook else "SPECIALIZED (larger chunks)"
            print(f"   Type: {doc_type}")
            print(f"   Chunk size: {chunk_size} words")
            
            # Extract text
            text, page_count = self._extract_pdf_text(pdf_path)
            total_pages += page_count
            
            # Chunk with appropriate size
            chunks = self._chunk_text(text, chunk_size)
            
            print(f"   ✓ {page_count} pages → {len(chunks)} chunks")
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': filename,
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'page_count': page_count,
                    'is_textbook': is_textbook,
                    'doc_type': 'textbook' if is_textbook else 'specialized'
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
    
    
    def _extract_pdf_text(self, pdf_path: str) -> Tuple[str, int]:
        """Extract text from PDF"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                text += page.extract_text() + "\n\n"
                if (i + 1) % 100 == 0:
                    print(f"   - Extracted {i+1}/{total_pages} pages")
            
            return text, total_pages
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return "", 0
    
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks with specified size"""
        words = text.split()
        chunks = []
        
        step_size = chunk_size - self.chunk_overlap
        
        for i in range(0, len(words), step_size):
            chunk = ' '.join(words[i:i + chunk_size])
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
            
            if (i + batch_size) % 1000 == 0:
                print(f"   - Indexed {min(i + batch_size, total)}/{total} chunks")
    
    
    def query(self, question: str, k: int = 4, use_mmr: bool = True, 
              enforce_diversity: bool = True) -> Dict:
        """
        Query with optimized retrieval strategy
        
        Args:
            question: User question
            k: Number of chunks to retrieve (default: 4, reduced from 5)
            use_mmr: Apply MMR for diversity
            enforce_diversity: Ensure source diversity
            
        Returns:
            Dict with answer, sources, and metadata
        """
        print(f"\n🔍 Query: {question}")
        print("="*70)
        
        # Retrieve with optimization
        print(f"📖 Retrieving top chunks (K={k})...")
        
        if use_mmr:
            docs = self._retrieve_with_mmr(question, k, enforce_diversity)
        else:
            docs = self._retrieve_basic(question, k)
        
        if not docs:
            print(f"⚠️  No documents above threshold ({self.similarity_threshold})")
            print(f"💡 Trying with lower threshold (0.5)...")
            
            # Fallback: try with lower threshold
            docs = self._retrieve_with_mmr(question, k, enforce_diversity, fallback_threshold=0.5)
            
            if not docs:
                return {
                    'question': question,
                    'answer': "No relevant documents found in the database.",
                    'sources': [],
                    'num_sources': 0
                }
        
        print(f"✓ Found {len(docs)} high-quality chunks")
        
        # Show source distribution
        sources = {}
        for doc in docs:
            src = doc['metadata']['source']
            sources[src] = sources.get(src, 0) + 1
        
        print(f"✓ Source distribution:")
        for src, count in sources.items():
            print(f"   - {src}: {count} chunks")
        
        # Generate answer
        print("🤖 Generating answer with Gemini...")
        answer = self._generate_answer(question, docs)
        
        return {
            'question': question,
            'answer': answer,
            'sources': docs,
            'num_sources': len(docs),
            'source_distribution': sources
        }
    
    
    def _retrieve_basic(self, query: str, k: int) -> List[Dict]:
        """Basic retrieval with similarity threshold"""
        query_vector = self.embedding_model.encode([query])[0].tolist()
        
        # Get more results initially to filter
        results = self._search_qdrant(query_vector, k * 3)
        
        # Filter by similarity threshold
        filtered = [
            {
                'text': r.payload['text'],
                'metadata': r.payload['metadata'],
                'score': r.score,
                'vector': query_vector  # Store for MMR
            }
            for r in results
            if r.score >= self.similarity_threshold
        ]
        
        return filtered[:k]
    
    
    def _retrieve_with_mmr(self, query: str, k: int, enforce_diversity: bool = True, 
                           fallback_threshold: float = None) -> List[Dict]:
        """
        Retrieve with Maximal Marginal Relevance for diversity
        
        MMR Formula: MMR = λ * Similarity(query, doc) - (1-λ) * max(Similarity(doc, selected))
        """
        query_vector = self.embedding_model.encode([query])[0]
        
        # Get initial candidates (more than needed)
        candidate_count = min(k * 4, 20)
        results = self._search_qdrant(query_vector.tolist(), candidate_count)
        
        # Use fallback threshold if provided
        threshold = fallback_threshold if fallback_threshold is not None else self.similarity_threshold
        
        # Filter by similarity threshold
        candidates = [
            {
                'text': r.payload['text'],
                'metadata': r.payload['metadata'],
                'score': r.score,
                'vector': np.array(self.embedding_model.encode([r.payload['text']])[0])
            }
            for r in results
            if r.score >= threshold
        ]
        
        if not candidates:
            # Show top scores for debugging
            if results:
                print(f"   Top scores found: {[f'{r.score:.3f}' for r in results[:3]]}")
            return []
        
        # Apply MMR
        selected = []
        selected_vectors = []
        
        # Select first document (highest relevance)
        selected.append(candidates[0])
        selected_vectors.append(candidates[0]['vector'])
        candidates.pop(0)
        
        # Select remaining documents with MMR
        while len(selected) < k and candidates:
            mmr_scores = []
            
            for candidate in candidates:
                # Relevance to query
                relevance = candidate['score']
                
                # Max similarity to already selected documents
                if selected_vectors:
                    similarities = [
                        np.dot(candidate['vector'], sel_vec) / 
                        (np.linalg.norm(candidate['vector']) * np.linalg.norm(sel_vec))
                        for sel_vec in selected_vectors
                    ]
                    max_similarity = max(similarities)
                else:
                    max_similarity = 0
                
                # MMR score
                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_similarity
                mmr_scores.append((mmr_score, candidate))
            
            # Select document with highest MMR score
            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            best_candidate = mmr_scores[0][1]
            
            selected.append(best_candidate)
            selected_vectors.append(best_candidate['vector'])
            candidates.remove(best_candidate)
        
        # Enforce source diversity if requested
        if enforce_diversity:
            selected = self._enforce_source_diversity(selected, k)
        
        # Clean up vectors from results
        for doc in selected:
            doc.pop('vector', None)
        
        return selected
    
    
    def _enforce_source_diversity(self, docs: List[Dict], target_k: int) -> List[Dict]:
        """
        Ensure we have chunks from multiple sources
        Boost specialized PDFs when they're relevant
        """
        # Group by source
        by_source = {}
        for doc in docs:
            source = doc['metadata']['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(doc)
        
        # If we already have good diversity, return as is
        if len(by_source) >= 2:
            return docs[:target_k]
        
        # Otherwise, ensure at least 2 sources if possible
        diverse_docs = []
        sources_used = set()
        
        # First pass: one from each source
        for doc in docs:
            source = doc['metadata']['source']
            if source not in sources_used:
                diverse_docs.append(doc)
                sources_used.add(source)
                if len(diverse_docs) >= target_k:
                    break
        
        # Second pass: fill remaining slots by score
        if len(diverse_docs) < target_k:
            remaining = [d for d in docs if d not in diverse_docs]
            diverse_docs.extend(remaining[:target_k - len(diverse_docs)])
        
        return diverse_docs
    
    
    def _search_qdrant(self, query_vector: List[float], limit: int):
        """Search Qdrant with compatibility handling"""
        try:
            return self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
        except AttributeError:
            # Fallback for different versions
            return self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            ).points
    
    
    def _generate_answer(self, question: str, context: List[Dict]) -> str:
        """Generate answer using Gemini"""
        
        # Build context with source annotations
        context_text = ""
        for i, doc in enumerate(context, 1):
            source = doc['metadata']['source']
            doc_type = doc['metadata'].get('doc_type', 'unknown')
            score = doc['score']
            
            context_text += f"\n[Source {i}: {source} | Type: {doc_type} | Relevance: {score:.3f}]\n{doc['text']}\n"
        
        # Create prompt
        prompt = f"""You are a medical assistant specializing in diabetes. Answer based ONLY on the provided context.

Context from clinical documents:
{context_text}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite sources explicitly (e.g., "According to Source 1 (Dietary Advice)...")
- Prioritize information from specialized documents when available
- If context is insufficient, state that clearly
- Use appropriate medical terminology

Answer:"""
        
        try:
            response = self.gemini.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"
    
    
    def interactive_mode(self):
        """Interactive Q&A mode"""
        print("\n" + "="*70)
        print("🎯 INTERACTIVE MODE (Optimized)")
        print("="*70)
        print("Type your questions. Type 'exit' to quit.\n")
        
        while True:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Query with optimizations
            result = self.query(question, k=4, use_mmr=True, enforce_diversity=True)
            
            # Display answer
            print(f"\n💡 Answer:\n{result['answer']}\n")
            
            # Display sources
            print(f"📚 Sources ({result['num_sources']}):")
            for i, src in enumerate(result['sources'], 1):
                doc_type = src['metadata'].get('doc_type', 'unknown')
                print(f"\n   {i}. {src['metadata']['source']} [{doc_type.upper()}]")
                print(f"      Score: {src['score']:.3f}")
                print(f"      {src['text'][:150]}...\n")
            
            print("="*70 + "\n")
    
    
    def compare_mode(self, question: str):
        """
        Compare old vs new retrieval approach
        """
        print("\n" + "="*70)
        print("🔬 COMPARISON MODE")
        print("="*70)
        print(f"Question: {question}\n")
        
        print("--- OLD APPROACH (No MMR, K=5, No threshold) ---")
        old_results = self._retrieve_basic(question, k=5)
        old_results = [r for r in old_results if r['score'] >= 0.5]  # Lower threshold
        
        print(f"Retrieved: {len(old_results)} chunks")
        old_sources = {}
        for r in old_results:
            src = r['metadata']['source']
            old_sources[src] = old_sources.get(src, 0) + 1
        print("Source distribution:")
        for src, count in old_sources.items():
            print(f"  - {src}: {count}")
        
        print("\n--- NEW APPROACH (MMR, K=4, Threshold=0.7, Diversity) ---")
        new_results = self._retrieve_with_mmr(question, k=4, enforce_diversity=True)
        
        print(f"Retrieved: {len(new_results)} chunks")
        new_sources = {}
        for r in new_results:
            src = r['metadata']['source']
            new_sources[src] = new_sources.get(src, 0) + 1
        print("Source distribution:")
        for src, count in new_sources.items():
            print(f"  - {src}: {count}")
        
        print("\n" + "="*70)
    
    
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


# =============================================================================
# MAIN TESTING SCRIPT
# =============================================================================

def main():
    """Main testing function"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║      Optimized Clinical RAG System                         ║
    ║      Smart Chunking + MMR + Source Diversity               ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    GEMINI_API_KEY = "AIzaSyDKIIVCT8i8wgNP-nZi3KVSDAAAAcvB-h0"  # Add your key
    PDF_PATHS = [
        "./pdfs/allchapters.pdf",  # Textbook - will use 300-word chunks
        "./pdfs/codebook15_llcp.pdf",  # Specialized - will use 500-word chunks
        "./pdfs/Dietary Advice For Individuals with Diabetes - Endotext - NCBI Bookshelf.pdf",
        "./pdfs/National Diabetes Statistics Report__Diabetes__CDC.pdf",
    ]
    
    # Initialize Optimized RAG
    rag = OptimizedRAG(
        gemini_api_key=GEMINI_API_KEY,
        qdrant_path="./qdrant_optimized_db",
        textbook_chunk_size=300,          # Smaller for textbook
        specialized_chunk_size=500,       # Larger for focused PDFs
        chunk_overlap=75,                 # Good balance
        similarity_threshold=0.60,        # Lowered from 0.70 (was too strict)
        mmr_lambda=0.6                    # Balance relevance and diversity
    )
    
    # Check if we need to load PDFs
    stats = rag.get_stats()
    
    if stats['total_chunks'] == 0:
        print("\n📚 No documents in database. Loading PDFs with optimized chunking...")
        
        existing_pdfs = [p for p in PDF_PATHS if os.path.exists(p)]
        
        if not existing_pdfs:
            print("\n⚠️  No PDF files found!")
            print("Please place your PDFs in the ./pdfs/ folder")
            return
        
        rag.load_pdfs(existing_pdfs)
    else:
        print(f"\n✅ Database loaded: {stats['total_chunks']} chunks available")
    
    # Test mode selection
    print("\n" + "="*70)
    print("Select Testing Mode:")
    print("  1. Interactive Q&A (optimized retrieval)")
    print("  2. Compare old vs new approach")
    print("  3. Single query test")
    print("="*70)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        rag.interactive_mode()
    
    elif choice == "2":
        question = input("\nEnter test question: ").strip()
        if question:
            rag.compare_mode(question)
    
    elif choice == "3":
        question = input("\nEnter your question: ").strip()
        if question:
            result = rag.query(question, k=4, use_mmr=True, enforce_diversity=True)
            
            print(f"\n💡 Answer:\n{result['answer']}\n")
            print(f"📚 Sources ({result['num_sources']}):")
            for i, src in enumerate(result['sources'], 1):
                doc_type = src['metadata'].get('doc_type', 'unknown')
                print(f"\n{i}. {src['metadata']['source']} [{doc_type.upper()}]")
                print(f"   Score: {src['score']:.3f}")
                print(f"   {src['text'][:200]}...")


if __name__ == "__main__":
    main()