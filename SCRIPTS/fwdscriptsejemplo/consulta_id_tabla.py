import requests
from dotenv import load_dotenv, find_dotenv
import os
from ragflow_sdk import RAGFlow
import requests
# =========================
# Carga de entorno y logging a ragflow
# =========================
load_dotenv(find_dotenv(usecwd=True))
RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY="ragflow-gyZmU5NTA4OTk1ZjExZjBhZTQ1NGFiMj"
RF_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "Noticias")
rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)
# --- dataset (get or create) ---
dataset = rf.get_dataset(name=RF_DATASET_NAME)
url = "http://localhost:23820/databases/default_db/tables"
response = requests.get(url)

if response.status_code == 200:
    # Si la respuesta es JSON
    try:
        data = response.json()
        print("Listado de tablas:")
        for table in data["tables"]:
            print(table)
    except ValueError:
        # Si la respuesta no es JSON, imprime como texto plano
        print(response.text)
else:
    print(f"Error {response.status_code}: {response.text}")

###Otra opción:
# --- Listar datasets disponibles ---
datasets = rf.list_datasets()

print("Datasets encontrados:")
for ds in datasets:
    print(f"- {ds.id} :: {ds.name}")
    print(ds.tenant_id)
# --- Listar tablas de un dataset específico (ejemplo: default_db) ---
#tables = rf.list_datasets()
#print("\nTablas:")
#for t in tables:
#    print("-", t)
#print("#####################################################################")
