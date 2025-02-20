from googletrans import Translator
from langchain.tools import Tool

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
