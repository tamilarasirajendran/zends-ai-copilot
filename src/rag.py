# This module retrieves the most relevant document using embeddings and FAISS.

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DOCUMENTS

# I used a pretrained sentence transformer model to convert text into embeddings.”
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# Convert Documents → Embeddings
doc_embeddings = embed_model.encode(DOCUMENTS)

# I used FAISS IndexFlatL2 for similarity search based on Euclidean distance.
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

#Retrieval Function
def retrieve(query):
    query_embedding = embed_model.encode([query]) #User input → vector
    D, I = index.search(np.array(query_embedding), k=1) #Search for closest document
    return DOCUMENTS[I[0][0]] #Return the most relevant document
