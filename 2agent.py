from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

# Carica le variabili globali
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODELLO")
llm = ChatOpenAI(model_name=model, api_key=openai_api_key)


def classifica_sentimento_agent(query: str) -> str:
    '''
    agente per capire il sentimento dell utente e classificarlo
    '''
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Analizza il sentimento del seguente testo e classificalo come:
                - Molto positivo        
                - Positivo
                - Neutro
                - Negativo
                - Molto negativo                
                Rispondi solo con una delle cinque categorie sopra.
                """
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"input": query})
    return response.content.strip()

def pulisci_query_agent(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Sei un assistente che riscrive domande per ottimizzare la ricerca, mantenendo la lingua originale."
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"input": f"Pulisci la seguente domanda: {query}"})
    return result.content.strip()

def classify_query_agent(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Sei un assistente intelligente.
                Decidi dove è meglio cercare la risposta:
                - 'FAQ' se la domanda è comune
                - 'Knowledge Base' se serve più dettaglio
                - 'Web' se la risposta non è nei dati interni o se
                Rispondi SOLO con 'FAQ', 'Knowledge Base' o 'Web'.
                """
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"input": f"Classifica la seguente domanda: {query}"})
    return result.content.strip().lower()
