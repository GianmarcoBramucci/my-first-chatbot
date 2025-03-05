from typing import Dict, List
from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import numpy as np
from langchain_openai import OpenAIEmbeddings

def cerca_su_internet(domanda: str) -> List[dict]:
    """
    Cerca informazioni su Internet e restituisce il contenuto testuale delle pagine.
    """
    try:
        # Esegue la ricerca su Google
        risultati_url = list(search(domanda, num=3, stop=3, lang="it"))
        
        contenuti_pagine = []
        
        # Scarica e analizza il contenuto di ogni pagina
        for url in risultati_url:
            try:
                # Invia una richiesta GET alla pagina
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                risposta = requests.get(url, headers=headers, timeout=10)
                
                # Verifica che la richiesta sia andata a buon fine
                risposta.raise_for_status()
                
                # Analizza il contenuto HTML
                soup = BeautifulSoup(risposta.text, 'html.parser')
                
                # Rimuove script, style e altri tag non testuali
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Estrae il testo principale
                testo = soup.get_text(separator=' ', strip=True)
                
                # Pulisce il testo (rimuove spazi extra, newline)
                testo = ' '.join(testo.split())
                
                # Tronca il testo se troppo lungo (ad esempio, primi 2000 caratteri)
                testo = testo[:2000]
                
                # Aggiunge il risultato alla lista
                contenuti_pagine.append({
                    'url': url,
                    'contenuto': testo
                })
                
            except requests.RequestException as e:
                print(f"Errore nel recuperare {url}: {e}")
        
        # Restituisce i risultati
        return contenuti_pagine if contenuti_pagine else [{"errore": "Nessun risultato trovato"}]
    
    except Exception as e:
        return [{"errore": f"Errore nella ricerca online: {e}"}]
    
def find_best_match(question, faq_embeddings, faq, kb_embeddings, kb, threshold=0.55):
    """
    Finds the most relevant answer by comparing the question with FAQ and KB embeddings
    using cosine similarity. Handles different embedding formats safely.
    """
    print("Attivo ricerca interna")

    # Get the question embedding
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
    question_embedding = embeddings_model.embed_documents([question])[0]

    def compute_similarity(question_embedding, embeddings_list):
        similarities = []
        for i, emb in enumerate(embeddings_list):
            try:
                q_array = np.array(question_embedding, dtype=np.float32)
                emb_array = np.array(emb, dtype=np.float32)

                # Assicurati che le dimensioni combacino
                if len(q_array.shape) == 1:
                    q_array = q_array.reshape(1, -1)
                if len(emb_array.shape) == 1:
                    emb_array = emb_array.reshape(1, -1)

                min_dim = min(q_array.shape[1], emb_array.shape[1])
                q_array = q_array[:, :min_dim]
                emb_array = emb_array[:, :min_dim]

                # Calcola la similarità
                sim = np.dot(q_array, emb_array.T).flatten()[0] / (
                    np.linalg.norm(q_array) * np.linalg.norm(emb_array)
                )
                similarities.append(float(sim))
            except Exception as e:
                print(f"Errore con l'elemento {i}: {e}")
                similarities.append(-1.0)
        return similarities

    # Calcola le similarità
    faq_similarities = compute_similarity(question_embedding, faq_embeddings)
    kb_similarities = compute_similarity(question_embedding, kb_embeddings)

    # Trova il miglior match nelle FAQ
    best_faq_idx, best_faq_score = (np.argmax(faq_similarities), max(faq_similarities)) if faq_similarities else (-1, -1)
    best_faq_answer = faq[best_faq_idx]["risposta"] if best_faq_score > threshold else None

    # Trova il miglior match nella KB
    best_kb_idx, best_kb_score = (np.argmax(kb_similarities), max(kb_similarities)) if kb_similarities else (-1, -1)
    best_kb_answer = {k: v for k, v in kb[best_kb_idx].items() if k != 'vec'} if best_kb_score > 0.40 else None

    # Costruisci la risposta combinata
    risposta = {}
    if best_faq_answer:
        risposta["faq"] = best_faq_answer
    if best_kb_answer:
        risposta["kb"] = best_kb_answer

    if risposta:
        return {"source": "faq+kb" if "faq" in risposta and "kb" in risposta else ("faq" if "faq" in risposta else "kb"),
                "content": risposta}
    
    return {"source": "none", "content": "Nessuna risposta trovata nelle FAQ o Knowledge Base."}

def cerca_nelle_faq(domanda: str, faq_embeddings, faq, kb_embeddings, kb) -> Dict[str, str]:
    """Cerca una risposta nelle FAQ utilizzando similarity search."""
    contesto = find_best_match(domanda, faq_embeddings, faq, kb_embeddings, kb)
    print(contesto)
    return {
        "risposta": contesto["content"],
        "fonte": contesto["source"]
    }

def create_tools(faq_embeddings, faq, kb_embeddings, kb):
    """Crea e restituisce i tools configurati."""
    
    web_tool = Tool(
        name="Web Search",
        func=cerca_su_internet,
        description="Cerca informazioni su Internet quando richiesto esplicitamente o quando non si trovano risposte nelle FAQ."
    )

    def cerca_nelle_faq_configured(domanda: str) -> Dict[str, str]:
        return cerca_nelle_faq(domanda, faq_embeddings, faq, kb_embeddings, kb)

    faq_tool = Tool(
        name="FAQ Search",
        func=cerca_nelle_faq_configured,
        description="Cerca risposte nelle FAQ e nella Knowledge Base interna."
    )

    return web_tool, faq_tool