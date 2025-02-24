import json
import os
import numpy as np
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


def openJson(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)

def jsonToText(json_data, indent=0):
    """
    Converte ricorsivamente un JSON in testo leggibile.
    """
    text = ""
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            text += "  " * indent + f"{key}:\n" + jsonToText(value, indent + 1)
    elif isinstance(json_data, list):
        for item in json_data:
            text += jsonToText(item, indent)
    else:
        text += "  " * indent + f"{json_data}\n"
    return text

def splitToChunks(json_path, chunk_size=250):
    """
    Legge un file JSON, lo converte in un unico testo e lo suddivide in porzioni di lunghezza definita da chunk_size.
    Ogni porzione di testo viene chiamata "chunk" e verrà utilizzata per generare gli embedding.
    """
    json_data = openJson(json_path)
    all_text = jsonToText(json_data)
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        separator="\n\n"
    )
    
    chunks = text_splitter.split_text(all_text)
    return chunks
def processJsonToEmbedding(json_path, output_path):
    """
    Carica un file JSON, genera gli embedding e salva sia i chunks che gli embedding nel file di output.
    """
    document_chunks = splitToChunks(json_path)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = embeddings_model.embed_documents(document_chunks)
    
    # Creiamo un dizionario con entrambi i dati
    output_data = {
        "chunks": document_chunks,
        "embeddings": embeddings
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"Chunks ed embeddings salvati con successo in {output_path}!")
    return document_chunks, np.array(embeddings)

# Caricamento API Key
env_loaded = load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

processJsonToEmbedding("data/faq.json", "data/embeddings_faq.json")
processJsonToEmbedding("data/knowledgeBase.json", "data/embeddings_kb.json")
