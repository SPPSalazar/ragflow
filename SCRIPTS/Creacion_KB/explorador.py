import inspect
from dotenv import load_dotenv
from ragflow_sdk import RAGFlow
import os
load_dotenv()
RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY = os.getenv("RAGFLOW_API_KEY", "dev")
RF_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME")

rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)

#print(inspect.signature(rf.retrieve))
metodos = [m for m in dir(rf) if not m.startswith("_")]
print(metodos)

#print(rf.list_rerank_models())

#noticias = [
#  {
#    "title": "Hombre asesinado parque",
#    "date": "07/10/2025",
#    "content": "Un hombre fue encontrado muerto tras un ataque armado nocturno.",
#    "category": ["asesinato"]
#  },
#  {
#    "title": "Ataque mortal restaurante",
#    "date": "07/11/2025",
#    "content": "Sospechosos armados dispararon contra clientes, dejando un fallecido.",
#    "category": ["asesinato"]
#  }]
