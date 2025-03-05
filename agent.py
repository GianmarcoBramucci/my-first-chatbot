import json
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tool import create_tools  

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

model = os.getenv("MODELLO")
llm = ChatOpenAI(model_name=model, api_key=openai_api_key)

def openJson(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)

def pulisci_query_agent(query: str) -> str:
    '''
    agente per pulire la query dell utente 
    '''
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un assistente che riscrive domande per ottimizzare la ricerca, traducendo italiano."),
        ("human", "{input}")
    ])
    chain = prompt | llm
    result = chain.invoke({"input": f"Pulisci la seguente domanda: {query}"})
    return result.content.strip()

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

def generate_assistance(query: str, sentiment: str,user: str, faq_result: dict, web_results: list = None) -> str:
    system_message = """
    Ruolo: Sei un assistente specializzato nell'assistenza clienti per l'azienda TechAssist Srl, che si occupa della vendita di hardware e software.  

    Contesto:TechAssist Srl ha difficoltà nella gestione delle richieste di assistenza, quindi il tuo compito è fornire risposte precise e tecniche ai clienti.  
    
    Obiettivo: Rispondi sempre in modo tecnico e pertinente, senza uscire dal contesto dell'azienda e dei suoi servizi.  
    Vincoli: 
    - Se una richiesta non specifica un soggetto, assumi che riguardi TechAssist Srl o i suoi servizi.  
    - rispondi in base al sentimento e al ruolo dell utente 
    - Effettua ricerche su Internet solo se l'utente lo richiede esplicitamente e se non trovi informazioni internamente.  
    - Cerca informazioni in italiano, ma rispondi nella lingua dell'utente.  

    Stile: Mantieni un tono professionale e tecnico, fornendo risposte chiare e dettagliate.  
    """
    users=openJson("data/users.json")
    context_message = f"il sentimento del utente: {sentiment}"
    context_message += f"\ncome comportarsi in base al ruolo di questo utente: {user}:{users['utenti'][user]}"
    context_message += f"\nConoscenza interna: {faq_result['risposta']}"
    if web_results:
        context_message += f"\nRisultati dalla ricerca web: {web_results}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"input": f"""Un utente ha posto una domanda. Rispondi nella stessa lingua dell'utente: "{query}"
    {context_message}
    Fornisci una risposta chiara e utile.
    """})
    return result.content.strip()


# Carica gli embedding
faq= openJson("data/faq.json")
kb= openJson("data/knowledgeBase.json")
faq_embeddings = [dizionario['dVec'] for dizionario in faq]
kb_embeddings = [dizionario['vec'] for dizionario in kb]
# Crea i tools
web_tool, faq_tool = create_tools(faq_embeddings, faq, kb_embeddings, kb)

def process_query(domanda: str,user: str) -> str:
    """
    Processa una query dell'utente utilizzando i tool di Langchain.
    """
    sentiment=classifica_sentimento_agent(domanda)
    domanda_pulita = pulisci_query_agent(domanda)
    # Prima cerca nelle FAQ
    faq_result = faq_tool.run(domanda_pulita)
    # Se necessario, cerca anche su web
    web_results = None
    if faq_result["fonte"] == "none":
        web_results = web_tool.run(domanda_pulita)
    # Genera la risposta finale
    risposta = generate_assistance(domanda,sentiment,user, faq_result, web_results)
    return risposta

# Esempio di utilizzo

domanda = "Where can I find your head office?"
risposta = process_query(domanda,"Cliente Occasionale")
print("Risposta AI:", risposta)