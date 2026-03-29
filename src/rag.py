from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DOCUMENTS

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(DOCUMENTS)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

def retrieve(query):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    return DOCUMENTS[I[0][0]]
