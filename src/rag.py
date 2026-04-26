# Retrieve the most relevant document using semantic search (FAISS + embeddings)
# Search engineof my chatbot

from sentence_transformers import SentenceTransformer #convert text → vectors
import faiss #fast similarity search
import numpy as np
from config import DOCUMENTS

# Load embedding model
# I used a pre-trained sentence transformer model to generate embeddings that capture semantic meaning.
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# Build FAISS Index
def build_index(filtered_docs): #Create search index
    texts = [doc["text"] for doc in filtered_docs]  #Extract text
    embeddings = embed_model.encode(texts) #Convert to embeddings Text → vectors
    
# I normalized embeddings to use cosine similarity for better semantic comparison.

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # Normalize vectors
#Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1]) #Used for similarity search
    index.add(embeddings) #Add vectors Store vectors

    return index, filtered_docs

# Main Retrieval Function
# I filtered documents based on intent to narrow down the search space.
# I retrieve the top-k most similar documents using FAISS.
def retrieve(query, intent):
    filtered_docs = [doc for doc in DOCUMENTS if doc["intent"] == intent] #Only relevant category

    index, docs = build_index(filtered_docs) #Build index

    query_embedding = embed_model.encode([query]) #Query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True) #Normalize again

    D, I = index.search(query_embedding, k=3) #D → similarity scores,I → indices

    return docs[I[0][0]]["text"]