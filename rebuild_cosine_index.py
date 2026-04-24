# scripts/rebuild_cosine_index.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Loading data...")
with open('data/ccpa_full_sections.json', 'r', encoding='utf-8') as f:
    full_sections = json.load(f)
with open('data/ccpa_sub_chunks.json', 'r', encoding='utf-8') as f:
    sub_chunks = json.load(f)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Build full sections index with cosine similarity
print("Building full sections index (cosine)...")
full_texts = [s['embed_text'] for s in full_sections]
full_embs  = embedder.encode(full_texts, show_progress_bar=True)
full_embs  = np.array(full_embs).astype('float32')
faiss.normalize_L2(full_embs)  # normalize for cosine
index_full = faiss.IndexFlatIP(full_embs.shape[1])
index_full.add(full_embs)
faiss.write_index(index_full, 'data/ccpa_full.index')
print(f"✅ Full index: {index_full.ntotal} vectors")

# Build sub-chunks index with cosine similarity
print("Building sub-chunks index (cosine)...")
sub_texts = [s['chunk_text'] for s in sub_chunks]
sub_embs  = embedder.encode(sub_texts, show_progress_bar=True)
sub_embs  = np.array(sub_embs).astype('float32')
faiss.normalize_L2(sub_embs)
index_sub = faiss.IndexFlatIP(sub_embs.shape[1])
index_sub.add(sub_embs)
faiss.write_index(index_sub, 'data/ccpa_sub.index')
print(f"✅ Sub index: {index_sub.ntotal} vectors")

print("\n🎉 Indexes rebuilt with cosine similarity!")
print("Now save retriever.py in app/ and run: python app/retriever.py")