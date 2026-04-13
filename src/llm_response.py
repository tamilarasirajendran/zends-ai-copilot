from transformers import pipeline
from src.rag import retrieve

# Load LLM
# I used GPT-2 for generating responses based on context.
llm = pipeline("text-generation", model="gpt2")


#Main Function
# This function takes user input, retrieves relevant context, and generates a response using the LLM.
def generate_response(query):
    
    context = retrieve(query)
    return context
# Prompt Creation
# I used GPT-2 to generate responses based on the prompt.
    prompt = f"Customer asked: {query}\nContext: {context}\nAnswer:"

    result = llm(
        prompt,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=50256
    )
# Extracting the generated answer from the LLM output.  
    response = result[0]['generated_text']

    # Answer 
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    # First sentence 
    if "." in response:
        response = response.split(".")[0].strip() + "."

    return response