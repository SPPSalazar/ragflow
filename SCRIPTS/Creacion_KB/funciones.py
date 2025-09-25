import os
from datetime import datetime
from dateutil import parser
from ragflow_sdk import RAGFlow
import logging
from typing import Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote_plus
from pymongo import MongoClient
from bson.json_util import dumps
import json
import logging
import requests
from slugify import slugify
# =========================
# Carga de entorno y logging
# =========================
load_dotenv(find_dotenv(usecwd=True))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("mongo_sample_struct")

# =========================
# Parámetros MongoDB
# =========================
MONGO_HOST = os.getenv("MONGO_PRO_HOST")
MONGO_PORT = os.getenv("MONGO_PRO_PORT")
MONGO_USER = os.getenv("MONGO_PRO_USER")
MONGO_PASS = os.getenv("MONGO_PRO_PASSWORD")
MONGO_DB = os.getenv("MONGO_PRO_DB")
MONGO_COL = os.getenv("MONGO_PRO_COLLECTION")

def _get_mongo_client():
  if MONGO_USER and MONGO_PASS:
    uri = (
    f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}"
    f"@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource=admin"
    )
  else:
    uri = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"
  client = MongoClient(uri, serverSelectionTimeoutMS=5000)
  client.admin.command("ping") # Verificar conexión
  return client

_get_mongo_client()


# =========================
# Muestreo y transformación
# =========================
def sample_noticias_structured(n=3) -> List[Dict[str, Any]]:
  client = _get_mongo_client()
  db = client[MONGO_DB]
  col = db[MONGO_COL]
  pipeline = [{"$sample": {"size": int(n)}},
              {"$sort": {"Fecha": -1}}  # -1 = descendente, 1 = ascendente
              ] 
  docs = list(col.aggregate(pipeline))

  client.close()
  return docs

def datetime_ISO8601(coleccion):
    for documento in coleccion:
        documento["Fecha"] = documento["Fecha"].isoformat()
    return coleccion

def cargar_docs(coleccion,dataset):
    #Realiza la carga del chunk y de la metadata
    #el embedding lo realiza ragflow automaticamente
    #con el modelo configurado en la plataforma
    for documento in coleccion:
        titulo = slugify(documento.get("Titulo"))
        contenido = slugify(documento.get("Contenido"))
        filename = f"{titulo}.json"
        json_bytes = json.dumps(documento, ensure_ascii=False).encode("utf-8")

        #INGESTA DE DOCS
        doc = dataset.upload_documents([{
        "display_name": filename,
        "blob": json_bytes # <--- SOLO bytes
        }])[0]
        print("Documento subido:", doc)
        # ============================================================
        # --- funciones para parsear: (indicar el chunk y la metadata,
        #  el embedding lo realiza ragflow automaticamente)
        # ============================================================
        #add ONE chunk (only content)
        doc.add_chunk(content=contenido)
        #attach metadata at document-level ---
        doc.update({"meta_fields": {
        "id_original":documento["_id"],
        "fecha": documento["Fecha"],
        "titulo": titulo,
        "url":documento["URL"],
        "medio":documento["Medio"],
        "categoria": documento["Categoria"],
        "entidades": documento.get("entidades", ""),
        "personas": documento.get("personas", ""),
        "relaciones":documento.get("relaciones_directas", "")
        }})
#import ragflow_sdk
#print(ragflow_sdk.__version__) #verificar que sea la 20.05 para que acepte "meta_fields"

#print("##############################################################################################")


def get_columns(INF_HOST, DB_NAME, TABLE_NAME,TIMEOUT):
    url = f"{INF_HOST}/databases/{DB_NAME}/tables/{TABLE_NAME}/columns"
    r = requests.get(url, headers={"accept": "application/json"}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    cols = data.get("columns", data)  # algunas versiones devuelven {"columns": [...]} otras solo [...]
    print("Columnas de la tabla:")
    for col in cols:
        if isinstance(col, dict):
            print(f"- {col.get('name')} ({col.get('type')})")
        else:
            print(f"- {col}")
    return cols

def get_rows(INF_HOST, DB_NAME, TABLE_NAME,TIMEOUT,limit=1, offset=0):
    url = f"{INF_HOST}/databases/{DB_NAME}/tables/{TABLE_NAME}/docs"
    payload = {"limit": str(limit), "offset": str(offset)}
    # Nota: Infinity acepta GET con body
    r = requests.get(url, json=payload, headers={"accept": "application/json"}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # Formato típico: {"error_code":0, "output":[{...}], "total_hits_count":N}
    if "output" in data:
        return data["output"]
    elif "rows" in data and "columns" in data:
        return [dict(zip(data["columns"], row)) for row in data["rows"]]
    return data


if __name__=="__main__":
  print("Muestra aleatoria de dos documentos de prueba")
  out = sample_noticias_structured(n=2)
  print(out)


