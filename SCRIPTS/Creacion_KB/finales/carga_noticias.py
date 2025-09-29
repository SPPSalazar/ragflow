from dotenv import load_dotenv, find_dotenv
import funciones
import os
from ragflow_sdk import RAGFlow

# =========================
# Carga de entorno y logging a ragflow
# =========================
load_dotenv(find_dotenv(usecwd=True))
RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY = "ragflow-ZlMmJiYWE4OWQ1MTExZjBhZTkxYzJiMT"# os.getenv("RAGFLOW_API_KEY", "dev")
RF_DATASET_NAME = ["2020", "2021", "2022", "2023","2024"]#la cantidad de años que se necesiten
rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)
# --- dataset (get or create) ---
# =========================
# --- sample record and normalization
# # =========================
for i in RF_DATASET_NAME:
    dataset = rf.get_dataset(name=i)
    noticias = funciones.sample_noticias_structured(300, int(i))
    noticias = funciones.datetime_ISO8601(noticias)#noticias[0]["Fecha"] es datetime.datetime no soportado por json
    funciones.cargar_docs_v2(noticias,dataset)



#EN EL README#####################
#Para consultar métodos use:
#import inspect
#RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
#RF_API_KEY = os.getenv("RAGFLOW_API_KEY", "dev")
#RF_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "a")
#rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)
# Lista de métodos públicos
#metodos = [m for m in dir(rf) if not m.startswith("_")]
#print("Métodos públicos:", metodos)
# Documentación de un método específico
#print(inspect.getdoc(rf.list_datasets))



