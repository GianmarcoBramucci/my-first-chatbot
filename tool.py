from googletrans import Translator
from langchain.tools import Tool
from googlesearch import search

translator = Translator()
def tool_rileva_lingua(testo):
    '''
    strumento per capire la lingua sfurttando i moduli di google
    '''
    lingua = translator.detect(testo)
    return lingua.lang  
lingua_tool = Tool(
    name="Language Detection",
    func=tool_rileva_lingua,
    description="Rileva la lingua del testo."
)

def cerca_su_internet(domanda):
    try:
        risultati = list(search(domanda, num=2, stop=2, lang="it"))
        if risultati:
            return risultati
        else:
            return "Non ho trovato risultati rilevanti su Internet."
    except Exception as e:
        return f"Errore nella ricerca online: {e}"

web_tool = Tool(
    name="Web Search",
    func=cerca_su_internet,
    description="Cerca informazioni su Internet se non presenti nelle FAQ o nel database o se richiesto dall utente."
)


