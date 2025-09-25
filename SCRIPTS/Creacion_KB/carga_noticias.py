from dotenv import load_dotenv, find_dotenv
import funciones, os
from ragflow_sdk import RAGFlow

# =========================
# Carga de entorno y logging a ragflow
# =========================
load_dotenv(find_dotenv(usecwd=True))
RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY="ragflow-gyZmU5NTA4OTk1ZjExZjBhZTQ1NGFiMj"
# os.getenv("RAGFLOW_API_KEY", "dev")
RF_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "Noticias")
rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)
# --- dataset (get or create) ---
dataset = rf.get_dataset(name=RF_DATASET_NAME)

# =========================
# --- sample record and normalization
# # =========================
noticias = funciones.sample_noticias_structured(n=200)
noticias = funciones.datetime_ISO8601(noticias)#noticias[0]["Fecha"] es datetime.datetime no soportado por json
funciones.cargar_docs(noticias,dataset)
