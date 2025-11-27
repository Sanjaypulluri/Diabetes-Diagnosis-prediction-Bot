"""
Upload all PDFs in a folder into the local RAG (Qdrant) collection.

Usage examples:
  python upload_folder.py --folder "./pdfs/ADA standards of Diabetic Care" --dry-run
  python upload_folder.py --folder ./pdfs --collection clinical_docs

The script supports `--dry-run` to only list found files and chunk counts.
"""
import argparse
import os
from typing import List, Dict
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


def find_pdfs(folder: str) -> List[str]:
    """Recursively find PDF files in folder"""
    pdfs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdfs.append(os.path.join(root, f))
    return sorted(pdfs)


def extract_pdf_text(pdf_path: str) -> (str, int):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        total_pages = len(reader.pages)
        for page in reader.pages:
            text_part = page.extract_text() or ""
            text += text_part + "\n\n"
        return text, total_pages
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return "", 0


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
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


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    try:
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Using existing collection: {collection_name}")
    except Exception as e:
        print(f"Error ensuring collection: {e}")
        raise


def index_chunks(client: QdrantClient, collection_name: str, embedder: SentenceTransformer, chunks: List[str], metas: List[Dict], batch_size: int = 100):
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_meta = metas[i:i + batch_size]
        embeddings = embedder.encode(batch_chunks).tolist()
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=emb, payload={'text': txt, 'metadata': meta})
            for txt, meta, emb in zip(batch_chunks, batch_meta, embeddings)
        ]
        client.upsert(collection_name=collection_name, points=points)
        print(f"  - Uploaded {min(i+batch_size, total)}/{total} chunks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='Folder containing PDFs')
    parser.add_argument('--qdrant-path', default='./qdrant_local_db', help='Local qdrant path')
    parser.add_argument('--qdrant-url', default=None, help='Qdrant cloud URL (e.g. https://your-cluster.a.qdrant.cloud)')
    parser.add_argument('--qdrant-api-key', default=None, help='Qdrant cloud API key')
    parser.add_argument('--collection', default='clinical_docs', help='Qdrant collection name')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', help='SentenceTransformer model')
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--chunk-overlap', type=int, default=50)
    parser.add_argument('--dry-run', action='store_true', help='Only list files and chunk counts')
    args = parser.parse_args()

    folder = args.folder
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return

    pdfs = find_pdfs(folder)
    if not pdfs:
        print(f"No PDFs found in {folder}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {folder}:")
    for p in pdfs:
        print(' -', p)

    # Quick dry-run: show per-file page/chunk counts
    embedder = SentenceTransformer(args.embedding_model)

    per_file_counts = []
    total_chunks = 0
    for pdf in pdfs:
        text, pages = extract_pdf_text(pdf)
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        per_file_counts.append((pdf, pages, len(chunks)))
        total_chunks += len(chunks)

    print('\nSummary:')
    for pdf, pages, chunks in per_file_counts:
        print(f" - {os.path.basename(pdf)}: {pages} pages → {chunks} chunks")
    print(f"Total chunks: {total_chunks}")

    if args.dry_run:
        print('\nDry-run complete. Use without --dry-run to upload to Qdrant.')
        return

    # Connect to Qdrant and upload (cloud or local)
    if args.qdrant_url:
        # Use cloud connection with API key (if provided)
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    else:
        client = QdrantClient(path=args.qdrant_path)
    vector_size = embedder.get_sentence_embedding_dimension()
    ensure_collection(client, args.collection, vector_size)

    all_chunks = []
    all_meta = []
    for pdf, pages, _ in per_file_counts:
        text, _ = extract_pdf_text(pdf)
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_meta.append({'source': os.path.basename(pdf), 'chunk_id': i, 'total_chunks': len(chunks), 'page_count': pages})

    print(f"\nUploading {len(all_chunks)} chunks to collection '{args.collection}'...")
    index_chunks(client, args.collection, embedder, all_chunks, all_meta)
    print('\nUpload complete!')


if __name__ == '__main__':
    main()
