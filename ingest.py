import json
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


def openJson(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


# Caricamento API Key
env_loaded = load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


faq=openJson("data/faq.json")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
for f in faq:
    f["dVec"] = embeddings_model.embed_documents([f["domanda"]])[0]
with open("data/faq.json", "w", encoding="utf-8") as f:
    json.dump(faq, f, indent=4, ensure_ascii=False)   



kb=openJson("data/knowledgeBase.json")
for f in kb:
    # Unisci tutti i contenuti in una singola stringa
    content = " ".join(f"{key}: {value}," for key, value in f.items())
    # Genera l'embedding    
    f["vec"] = embeddings_model.embed_documents([content])[0]

# Salva il file aggiornato
with open("data/knowledgeBase.json", "w", encoding="utf-8") as file:
    json.dump(kb, file, indent=4, ensure_ascii=False)
