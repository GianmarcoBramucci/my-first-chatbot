from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODELLO")
chat = ChatOpenAI(model_name=model, api_key=openai_api_key)

query = "ciao,dimmi un curiosita"

response = chat.invoke([HumanMessage(query)])
 
print(response.content)