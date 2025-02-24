import json
import getpass
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


def openJson(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        dati = json.load(f)
    return dati

def spliAndCheckJsonToChunks(json_data,data, chunk_size=250):
    # Convertiamo ogni FAQ in un formato testo
    texts = []
    for qa in json_data[data]:
        text = f"{qa['domanda']}\n{qa['risposta']}\n\n"
        texts.append(text)
    
    # Uniamo tutti i testi
    all_text = "".join(texts)
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        separator="\n\n"  # Divide sui doppi newline per mantenere domande/risposte insieme
    )
    
    # Dividiamo il testo
    chunks = text_splitter.split_text(all_text)
    return chunks



faq=openJson("data/faq.json")
kBase=openJson("data/knowledgeBase.json")

faqChunks =spliAndCheckJsonToChunks(faq,"faq")


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass(openai_api_key)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = embeddings_model.embed_documents(faqChunks)

with open("data/embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=4)
