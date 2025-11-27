"""
Optimized Upload Script for RAG System
Uploads PDFs with smart chunking strategy:
- Textbook PDFs: 300-word chunks (smaller, more precise)
- Specialized PDFs: 500-word chunks (larger, focused)

Usage examples:
  python upload_folder_optimized.py --folder "./pdfs" --dry-run
  python upload_folder_optimized.py --folder ./pdfs --collection clinical_docs_optimized
  python upload_folder_optimized.py --folder ./pdfs --qdrant-url https://xyz.qdrant.io --qdrant-api-key YOUR_KEY
"""
import argparse
import os
from typing import List, Dict, Tuple
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


# =============================================================================
# CONFIGURATION
# =============================================================================

# Textbook identifiers - PDFs with these keywords will use smaller chunks
TEXTBOOK_KEYWORDS = [
    'allchapters',
    'textbook',
    'manual',
    'handbook',
    'comprehensive'
]

# Chunk sizes
TEXTBOOK_CHUNK_SIZE = 300      # Smaller for broad/general content
SPECIALIZED_CHUNK_SIZE = 500   # Larger for focused content
CHUNK_OVERLAP = 75             # Overlap between chunks


# =============================================================================
# PDF PROCESSING
# =============================================================================

def find_pdfs(folder: str) -> List[str]:
    """Recursively find PDF files in folder"""
    pdfs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdfs.append(os.path.join(root, f))
    return sorted(pdfs)


def is_textbook(filename: str) -> bool:
    """Determine if PDF is a textbook based on filename"""
    filename_lower = filename.lower()
    for keyword in TEXTBOOK_KEYWORDS:
        if keyword in filename_lower:
            return True
    return False


def extract_pdf_text(pdf_path: str) -> Tuple[str, int]:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        total_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            text_part = page.extract_text() or ""
            text += text_part + "\n\n"
            
            # Progress indicator for large PDFs
            if (i + 1) % 100 == 0:
                print(f"    Extracting... {i+1}/{total_pages} pages")
        
        return text, total_pages
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return "", 0


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size
    
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def process_pdf(pdf_path: str, chunk_overlap: int, textbook_chunk_size: int, specialized_chunk_size: int) -> Dict:
    """
    Process a single PDF with smart chunking
    
    Returns:
        Dict with filename, is_textbook, pages, chunk_size, chunks, text
    """
    filename = os.path.basename(pdf_path)
    is_tb = is_textbook(filename)
    chunk_size = textbook_chunk_size if is_tb else specialized_chunk_size
    
    print(f"\n📄 Processing: {filename}")
    print(f"   Type: {'TEXTBOOK' if is_tb else 'SPECIALIZED'}")
    print(f"   Chunk size: {chunk_size} words")
    
    text, pages = extract_pdf_text(pdf_path)
    
    if not text:
        print(f"   ⚠️  No text extracted")
        return None
    
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    print(f"   ✓ {pages} pages → {len(chunks)} chunks")
    
    return {
        'filename': filename,
        'path': pdf_path,
        'is_textbook': is_tb,
        'pages': pages,
        'chunk_size': chunk_size,
        'chunks': chunks,
        'text': text
    }


# =============================================================================
# QDRANT OPERATIONS
# =============================================================================

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create collection if it doesn't exist"""
    try:
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Created collection: {collection_name}")
        else:
            print(f"✅ Using existing collection: {collection_name}")
    except Exception as e:
        print(f"❌ Error ensuring collection: {e}")
        raise


def index_chunks(
    client: QdrantClient, 
    collection_name: str, 
    embedder: SentenceTransformer, 
    chunks: List[str], 
    metadata: List[Dict], 
    batch_size: int = 100
):
    """Upload chunks to Qdrant"""
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]
        
        # Generate embeddings
        embeddings = embedder.encode(batch_chunks).tolist()
        
        # Create points
        points = [
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=emb, 
                payload={'text': txt, 'metadata': meta}
            )
            for txt, meta, emb in zip(batch_chunks, batch_meta, embeddings)
        ]
        
        # Upload
        client.upsert(collection_name=collection_name, points=points)
        
        # Progress
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= total:
            print(f"   - Uploaded {min(i + batch_size, total)}/{total} chunks")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Upload PDFs to RAG system with optimized chunking'
    )
    parser.add_argument('--folder', required=True, help='Folder containing PDFs')
    parser.add_argument('--qdrant-path', default='./qdrant_optimized_db', 
                       help='Local qdrant path')
    parser.add_argument('--qdrant-url', default=None, 
                       help='Qdrant cloud URL (e.g. https://xyz.qdrant.io)')
    parser.add_argument('--qdrant-api-key', default=None, 
                       help='Qdrant cloud API key')
    parser.add_argument('--collection', default='clinical_docs_optimized', 
                       help='Qdrant collection name')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', 
                       help='SentenceTransformer model')
    parser.add_argument('--textbook-chunk-size', type=int, default=TEXTBOOK_CHUNK_SIZE,
                       help='Chunk size for textbooks (default: 300)')
    parser.add_argument('--specialized-chunk-size', type=int, default=SPECIALIZED_CHUNK_SIZE,
                       help='Chunk size for specialized PDFs (default: 500)')
    parser.add_argument('--chunk-overlap', type=int, default=CHUNK_OVERLAP,
                       help='Chunk overlap (default: 75)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Only list files and chunk counts without uploading')
    
    args = parser.parse_args()
    
    # Use local variables instead of modifying globals
    textbook_chunk_size = args.textbook_chunk_size
    specialized_chunk_size = args.specialized_chunk_size
    chunk_overlap = args.chunk_overlap
    
    # Validate folder
    folder = args.folder
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        return
    
    # Find PDFs
    print(f"🔍 Scanning folder: {folder}")
    pdfs = find_pdfs(folder)
    
    if not pdfs:
        print(f"❌ No PDFs found in {folder}")
        return
    
    print(f"\n✅ Found {len(pdfs)} PDF(s):")
    for p in pdfs:
        print(f"   - {os.path.basename(p)}")
    
    # Load embedding model
    print(f"\n🧠 Loading embedding model: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model)
    
    # Process all PDFs
    print(f"\n📚 Processing PDFs with smart chunking...")
    processed_pdfs = []
    
    for pdf_path in pdfs:
        result = process_pdf(pdf_path, chunk_overlap, textbook_chunk_size, specialized_chunk_size)
        if result:
            processed_pdfs.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    
    total_pages = sum(p['pages'] for p in processed_pdfs)
    total_chunks = sum(len(p['chunks']) for p in processed_pdfs)
    
    textbook_count = sum(1 for p in processed_pdfs if p['is_textbook'])
    specialized_count = len(processed_pdfs) - textbook_count
    
    print(f"Total PDFs processed: {len(processed_pdfs)}")
    print(f"  - Textbooks: {textbook_count} (using {textbook_chunk_size}-word chunks)")
    print(f"  - Specialized: {specialized_count} (using {specialized_chunk_size}-word chunks)")
    print(f"Total pages: {total_pages}")
    print(f"Total chunks: {total_chunks}")
    print(f"Chunk overlap: {chunk_overlap} words")
    
    print(f"\nPer-file breakdown:")
    for p in processed_pdfs:
        doc_type = "TEXTBOOK" if p['is_textbook'] else "SPECIALIZED"
        print(f"  [{doc_type}] {p['filename']}: {p['pages']} pages → {len(p['chunks'])} chunks")
    
    # Dry run check
    if args.dry_run:
        print(f"\n✅ Dry-run complete. Use without --dry-run to upload to Qdrant.")
        return
    
    # Connect to Qdrant
    print(f"\n🔌 Connecting to Qdrant...")
    if args.qdrant_url:
        print(f"   Mode: CLOUD")
        print(f"   URL: {args.qdrant_url}")
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    else:
        print(f"   Mode: LOCAL")
        print(f"   Path: {args.qdrant_path}")
        client = QdrantClient(path=args.qdrant_path)
    
    # Setup collection
    vector_size = embedder.get_sentence_embedding_dimension()
    ensure_collection(client, args.collection, vector_size)
    
    # Prepare all chunks and metadata
    print(f"\n💾 Preparing data for upload...")
    all_chunks = []
    all_metadata = []
    
    for pdf_info in processed_pdfs:
        for i, chunk in enumerate(pdf_info['chunks']):
            all_chunks.append(chunk)
            all_metadata.append({
                'source': pdf_info['filename'],
                'chunk_id': i,
                'total_chunks': len(pdf_info['chunks']),
                'page_count': pdf_info['pages'],
                'is_textbook': pdf_info['is_textbook'],
                'doc_type': 'textbook' if pdf_info['is_textbook'] else 'specialized',
                'chunk_size': pdf_info['chunk_size']
            })
    
    # Upload to Qdrant
    print(f"\n🚀 Uploading {len(all_chunks)} chunks to collection '{args.collection}'...")
    index_chunks(client, args.collection, embedder, all_chunks, all_metadata)
    
    print(f"\n{'='*70}")
    print("✅ UPLOAD COMPLETE!")
    print(f"{'='*70}")
    print(f"Collection: {args.collection}")
    print(f"Total chunks uploaded: {len(all_chunks)}")
    print(f"Ready for querying! 🎉")


if __name__ == '__main__':
    main()