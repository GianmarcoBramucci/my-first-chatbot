from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODELLO")
chat = ChatOpenAI(model_name=model, api_key=openai_api_key)

query = "ciao,dimmi un curiosita"
prompt = PromptTemplate.from_template("Fornisci una risposta utilizzando un linguaggio per bambini alla seguente domanda {question}")

query = "dimmi una curiosita"

formatted_prompt = prompt.format(question=query)
response = chat.invoke([HumanMessage(content=formatted_prompt)])

print(response.content)