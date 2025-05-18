import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Paths
DATA_PATH = "rag_data/geology/geological_facts.jsonl"
INDEX_DIR = "rag_data/geology/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Load model
print("🔄 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read data
print("📚 Reading dataset...")
texts, metadata = [], []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = item.get("text")
        source = item.get("source", "geology")
        if text:
            texts.append(text)
            metadata.append({"text": text, "source": source})

# Embed texts
print(f"🧠 Embedding {len(texts)} texts...")
embeddings = model.encode(texts, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
print("💾 Saving FAISS index and metadata...")
faiss.write_index(index, os.path.join(INDEX_DIR, "geology.index"))
with open(os.path.join(INDEX_DIR, "geology_meta.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("✅ Indexing complete.")
