from typing import Dict, List
from langchain.tools import Tool
from googlesearch import search
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

def cerca_su_internet(domanda: str) -> List[str]:
    """Cerca informazioni su Internet utilizzando Google Search."""
    try:
            risultati = list(search(domanda, num=2, stop=2, lang="it"))
            if risultati:
                return risultati
            else:
                return "Non ho trovato risultati rilevanti su Internet."
    except Exception as e:
            return f"Errore nella ricerca online: {e}"

def find_best_match(question, faq_embeddings, faq_chunks, kb_embeddings, kb_chunks, theshold=0.65):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
    question_embedding = np.array(embeddings_model.embed_documents([question]))
    
    faq_similarities = cosine_similarity(question_embedding, faq_embeddings)[0]
    kb_similarities = cosine_similarity(question_embedding, kb_embeddings)[0]
    
    best_faq_idx = np.argmax(faq_similarities)
    best_faq_score = faq_similarities[best_faq_idx]
    
    if best_faq_score > theshold:
        return {"source": "faq", "content": faq_chunks[best_faq_idx]}
    
    best_kb_idx = np.argmax(kb_similarities)
    best_kb_score = kb_similarities[best_kb_idx]
    
    if best_kb_score > 0.5:
        return {"source": "kb", "content": kb_chunks[best_kb_idx]}
    
    return {"source": "none", "content": "Nessuna risposta trovata nelle FAQ o Knowledge Base."}

def cerca_nelle_faq(domanda: str, faq_embeddings, faq_chunks, kb_embeddings, kb_chunks) -> Dict[str, str]:
    """Cerca una risposta nelle FAQ utilizzando similarity search."""
    contesto = find_best_match(domanda, faq_embeddings, faq_chunks, kb_embeddings, kb_chunks)
    return {
        "risposta": contesto["content"],
        "fonte": contesto["source"]
    }

def create_tools(faq_embeddings, faq_chunks, kb_embeddings, kb_chunks):
    """Crea e restituisce i tools configurati."""
    
    web_tool = Tool(
        name="Web Search",
        func=cerca_su_internet,
        description="Cerca informazioni su Internet quando richiesto esplicitamente o quando non si trovano risposte nelle FAQ."
    )

    # Creiamo una closure per passare gli embeddings a cerca_nelle_faq
    def cerca_nelle_faq_configured(domanda: str) -> Dict[str, str]:
        return cerca_nelle_faq(domanda, faq_embeddings, faq_chunks, kb_embeddings, kb_chunks)

    faq_tool = Tool(
        name="FAQ Search",
        func=cerca_nelle_faq_configured,
        description="Cerca risposte nelle FAQ e nella Knowledge Base interna."
    )

    return web_tool, faq_tool