from transformers import pipeline
from src.rag import retrieve

llm = pipeline("text-generation", model="gpt2")

def generate_response(query):
    
    context = retrieve(query)
    return context

    prompt = f"Customer asked: {query}\nContext: {context}\nAnswer:"

    result = llm(
        prompt,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=50256
    )

    response = result[0]['generated_text']

    # Answer 
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    # First sentence 
    if "." in response:
        response = response.split(".")[0].strip() + "."

    return response