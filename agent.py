import json
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import numpy as np
from tool import create_tools  

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODELLO")
llm = ChatOpenAI(model_name=model, api_key=openai_api_key)

def load_embeddings(embedding_path):
    with open(embedding_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"], np.array(data["embeddings"])


def generate_assistance(query: str, faq_result: dict, web_results: list = None) -> str:
    system_message = """Sei un assistente dedicato ai clienti perché L'azienda **TechAssist Srl**, 
    specializzata nella vendita di hardware e software, ha difficoltà nella gestione delle richieste 
    di assistenza dei clienti rispondi sempre in maniera tecnica e mai fuori contesto e sopratutto a meno che non venga chiesto cerca su intenet se non trovi nulla.
    se non e specificato il sogetto della domanda si prende per scontato che si stanno rivolgendo a TechAssist Srl o a uno dei suoi servizzi
    """
    
    context_message = f"Conoscenza interna: {faq_result['risposta']}"
    if web_results:
        context_message += f"\nRisultati dalla ricerca web: {web_results}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"input": f"""Un utente ha chiesto: "{query}"
    {context_message}
    Fornisci una risposta chiara e utile.
    """})
    return result.content.strip()

def pulisci_query_agent(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un assistente che riscrive domande per ottimizzare la ricerca, mantenendo la lingua originale."),
        ("human", "{input}")
    ])
    chain = prompt | llm
    result = chain.invoke({"input": f"Pulisci la seguente domanda: {query}"})
    return result.content.strip()

# Carica gli embedding
faq_chunks, faq_embeddings = load_embeddings("data/embeddings_faq.json")
kb_chunks, kb_embeddings = load_embeddings("data/embeddings_kb.json")

# Crea i tools
web_tool, faq_tool = create_tools(faq_embeddings, faq_chunks, kb_embeddings, kb_chunks)

def process_query(domanda: str) -> str:
    """
    Processa una query dell'utente utilizzando i tool di Langchain.
    """
    domanda_pulita = pulisci_query_agent(domanda)
    
    # Prima cerca nelle FAQ
    faq_result = faq_tool.run(domanda_pulita)
    
    # Se necessario, cerca anche su web
    web_results = None
    if faq_result["fonte"] == "none":
        web_results = web_tool.run(domanda_pulita)
    
    # Genera la risposta finale
    risposta = generate_assistance(domanda_pulita, faq_result, web_results)
    return risposta

# Esempio di utilizzo

domanda = "ciao come sono i nuovi processori usciti?"
risposta = process_query(domanda)
print("Risposta AI:", risposta)