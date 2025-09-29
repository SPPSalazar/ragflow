from dotenv import load_dotenv, find_dotenv
import funciones2 as funciones
import os
from ragflow_sdk import RAGFlow

# =========================
# Carga de entorno y logging a ragflow
# =========================
load_dotenv(find_dotenv(usecwd=True))
RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY = "ragflow-ZlMmJiYWE4OWQ1MTExZjBhZTkxYzJiMT"# os.getenv("RAGFLOW_API_KEY", "dev")
RF_DATASET_NAME = ["Last3months", "2025"]
rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)
# --- dataset (get or create) ---
# =========================
# --- sample record and normalization
# # =========================
for i in RF_DATASET_NAME:
    dataset = rf.get_dataset(name=i)
    if i != "2025":
        noticias = funciones.sample_last3months(10)
    else:
        noticias = funciones.sample_noticias_3months_complemento(1)
    noticias = funciones.datetime_ISO8601(noticias)#noticias[0]["Fecha"] es datetime.datetime no soportado por json
    funciones.cargar_docs_v2(noticias,dataset)
print("##################################################")
#dataset = rf.get_dataset(name="Last3months")
#c = funciones.delete_docs_older_than_3months(dataset=dataset)
