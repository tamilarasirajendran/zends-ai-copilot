#Generate a final response using LLM (GPT-2) with retrieved context (RAG)

from transformers import pipeline #Loads pre-trained models easily
from src.rag import retrieve  #Used to get relevant data (context)

# I used GPT-2 for generating natural language responses.
llm = pipeline("text-generation", model="gpt2")


#Main Function
# I retrieve relevant information based on the query before generating the response.
def generate_response(query, intent):
    
    context = retrieve(query, intent) #This is RAG part
    
# This is prompt engineering
# I used prompt engineering to guide the model with context and question.
    prompt = f"""
    You are a helpful customer support assistant.

    Question: {query}
    Context: {context}

    Answer:
    """
# I controlled the generation using parameters like max tokens and deterministic decoding.
    result = llm(
        prompt,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=50256
    )
# Model Output
# I post-processed the output to extract only the final answer.
    response = result[0]['generated_text']

    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip() #Clean response

    return response