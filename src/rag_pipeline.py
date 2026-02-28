from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded (specifically GROQ_API_KEY)
load_dotenv()

def get_qa_chain(vectorstore):
    """
    Builds the Retrieval components with a strict anti-hallucination prompt.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    
    # Strict prompt template telling the LLM to only use context or report information not available
    template = """You are an AI assistant designed to answer questions strictly based on the provided Swiggy Annual Report context.

Use the following pieces of context to answer the user's question. 

CRITICAL INSTRUCTIONS:
1. You must answer ONLY using the information provided in the Context below.
2. If the answer cannot be found in the Context provided, you must reply EXACTLY with this phrase: "This information is not available in the Swiggy Annual Report."
3. Do not include outside knowledge, assumptions, or hallucinations.
4. If you find the answer, provide a clear and concise response based purely on the text.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    return {"retriever": retriever, "prompt": prompt, "llm": llm}

def ask(chain_components, question: str) -> dict:
    """
    Executes a query against the components and returns the dictionary with 'answer' and 'sources'.
    """
    retriever = chain_components["retriever"]
    prompt = chain_components["prompt"]
    llm = chain_components["llm"]
    
    # Retrieve documents
    docs = retriever.invoke(question)
    
    # Build context string
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Format prompt and call LLM
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    
    return {
        "answer": response.content,
        "sources": docs
    }
