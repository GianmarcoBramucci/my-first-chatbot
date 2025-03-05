import json
import os
import uuid
from datetime import datetime, timedelta
from transformers import pipeline

def analyze_sentiment(text, sentiment_analyzer=None):
    """
    Analizza il sentiment di un testo
    
    Args:
        text (str): Testo da analizzare
        sentiment_analyzer (pipeline, optional): Analizzatore di sentiment pre-inizializzato
    
    Returns:
        dict: Risultato dell'analisi del sentiment
    """
    if sentiment_analyzer is None:
        sentiment_analyzer = pipeline("sentiment-analysis")
    
    try:
        result = sentiment_analyzer(text)[0]
        return {
            "label": result['label'],
            "score": result['score']
        }
    except Exception:
        # Fallback con classificazione manuale
        return _manual_sentiment_classification(text)

def _manual_sentiment_classification(text):
    """Classificazione manuale del sentiment basata su parole chiave"""
    positive_words = ['bene', 'grazie', 'ottimo', 'perfetto', 'eccellente']
    negative_words = ['problema', 'guasto', 'difficoltà', 'errore', 'non funziona']
    
    text_lower = text.lower()
    positive_count = sum(word in text_lower for word in positive_words)
    negative_count = sum(word in text_lower for word in negative_words)
    
    if positive_count > negative_count:
        return {"label": "POSITIVE", "score": 0.7}
    elif negative_count > positive_count:
        return {"label": "NEGATIVE", "score": 0.7}
    else:
        return {"label": "NEUTRAL", "score": 0.5}

def save_conversation_memory(
    conversation_id, 
    user_query, 
    ai_response, 
    user_role, 
    sentiment=None,
    memory_file=None
):
    """
    Salva l'interazione in memoria
    
    Args:
        conversation_id (str): ID univoco della conversazione
        user_query (str): Query dell'utente
        ai_response (str): Risposta dell'IA
        user_role (str): Ruolo dell'utente
        sentiment (str, optional): Sentiment dell'interazione
        memory_file (str, optional): Percorso del file di memoria
    """
    if memory_file is None:
        memory_file = f"data/conversation_memory_{conversation_id}.json"
    
    # Carica memoria esistente
    memory = []
    if os.path.exists(memory_file):
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
    
    # Analizza sentiment se non fornito
    if sentiment is None:
        sentiment = analyze_sentiment(user_query)['label']
    
    # Nuova interazione
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "ai_response": ai_response,
        "user_role": user_role,
        "sentiment": sentiment
    }
    
    # Aggiungi interazione
    memory.append(interaction)
    
    # Salva memoria
    os.makedirs(os.path.dirname(memory_file) or '.', exist_ok=True)
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    
    # Genera sommario ogni 10 interazioni
    if len(memory) % 10 == 0:
        generate_conversation_summary(conversation_id, memory)

def generate_conversation_summary(conversation_id, memory):
    """
    Genera un sommario della conversazione
    
    Args:
        conversation_id (str): ID della conversazione
        memory (list): Memoria della conversazione
    """
    summary = {
        "conversation_id": conversation_id,
        "total_interactions": len(memory),
        "date_range": {
            "start": memory[0]['timestamp'],
            "end": memory[-1]['timestamp']
        },
        "sentiment_distribution": _calculate_sentiment_distribution(memory),
        "summary_timestamp": datetime.now().isoformat()
    }
    
    summary_file = f"conversation_summary_{conversation_id}.json"
    os.makedirs('conversation_summaries', exist_ok=True)
    
    with open(os.path.join('conversation_summaries', summary_file), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def _calculate_sentiment_distribution(memory):
    """Calcola la distribuzione dei sentiment"""
    sentiments = [interaction['sentiment'] for interaction in memory]
    return {
        sentiment: sentiments.count(sentiment) 
        for sentiment in set(sentiments)
    }

def create_ticket(
    query, 
    sentiment, 
    user_role, 
    conversation_id,
    tickets_file='data/tickets.json'
):
    """
    Crea un nuovo ticket se necessario
    
    Args:
        query (str): Query dell'utente
        sentiment (str): Sentiment dell'interazione
        user_role (str): Ruolo dell'utente
        tickets_file (str, optional): Percorso del file dei ticket
    
    Returns:
        dict or None: Ticket creato o None
    """
    # Verifica se è necessario creare un ticket
    if sentiment not in ["Negativo", "Molto Negativo"]:
        return None
    
    # Carica ticket esistenti
    tickets = {"aperti": [], "chiusi": [], "statistiche": {}}
    if os.path.exists(tickets_file):
        with open(tickets_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
    
    # Genera motivazione
    motivazioni_map = {
        "Molto Negativo": [
            "Problema grave che richiede attenzione immediata",
            "Esperienza utente fortemente compromessa"
        ],
        "Negativo": [
            "Insoddisfazione del cliente da gestire",
            "Problema tecnico da risolvere"
        ]
    }
    
    ruoli_priorita = {
        "Cliente Premium": "Alta",
        "Cliente Registrato": "Media",
        "Cliente Occasionale": "Bassa"
    }
    
    motivazioni = motivazioni_map.get(sentiment, motivazioni_map["Negativo"])
    priorita = ruoli_priorita.get(user_role, "Bassa")
    
    # Genera ticket
    ticket = {
        "id": conversation_id,
        "timestamp": datetime.now().isoformat(),
        "query_originale": query,
        "sentiment": sentiment,
        "ruolo_utente": user_role,
        "motivazione": f"{motivazioni[0]} - Priorità {priorita}",
        "stato": "Aperto",
        "data_scadenza": (datetime.now() + timedelta(days=2)).isoformat()
    }
    
    # Aggiorna statistiche
    tickets["aperti"].append(ticket)
    tickets["statistiche"]["totale_ticket"] = tickets["statistiche"].get("totale_ticket", 0) + 1
    
    tickets["statistiche"]["ticket_per_ruolo"] = tickets["statistiche"].get("ticket_per_ruolo", {})
    tickets["statistiche"]["ticket_per_ruolo"][user_role] = tickets["statistiche"]["ticket_per_ruolo"].get(user_role, 0) + 1
    
    tickets["statistiche"]["ticket_per_sentimento"] = tickets["statistiche"].get("ticket_per_sentimento", {})
    tickets["statistiche"]["ticket_per_sentimento"][sentiment] = tickets["statistiche"]["ticket_per_sentimento"].get(sentiment, 0) + 1
    
    # Salva ticket
    os.makedirs(os.path.dirname(tickets_file) or '.', exist_ok=True)
    with open(tickets_file, 'w', encoding='utf-8') as f:
        json.dump(tickets, f, ensure_ascii=False, indent=2)
    
    return ticket

def genera_messaggio_ticket(ticket):
    """
    Genera un messaggio per l'utente sulla creazione del ticket
    
    Args:
        ticket (dict): Ticket generato
    
    Returns:
        str: Messaggio per l'utente
    """
    if not ticket:
        return ""
    
    template_messaggi = {
        "Molto Negativo": "Ci scusiamo per il disagio. Un nostro operatore ti contatterà al più presto per risolvere immediatamente la tua problematica.",
        "Negativo": "Abbiamo registrato la tua segnalazione. Un nostro consulente ti contatterà entro 48 ore per supportarti."
    }
    
    messaggio = template_messaggi.get(ticket['sentiment'], 
        "Abbiamo registrato la tua segnalazione e provvederemo a gestirla.")
    
    return f"{messaggio} Numero ticket: {ticket['id'][:8]}"