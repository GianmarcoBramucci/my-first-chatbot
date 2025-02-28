from typing import Dict, List
from langchain.tools import Tool
from googlesearch import search
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

def cerca_su_internet(domanda: str) -> List[str]:
    """Cerca informazioni su Internet utilizzando Google Search."""
    try:
            print("attivo internet")
            risultati = list(search(domanda, num=2, stop=2, lang="it"))
            if risultati:
                return risultati
            else:
                return "Non ho trovato risultati rilevanti su Internet."
    except Exception as e:
            return f"Errore nella ricerca online: {e}"
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