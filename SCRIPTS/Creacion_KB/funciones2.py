import os
from datetime import datetime
from dateutil import parser
from ragflow_sdk import RAGFlow
import logging
from typing import Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote_plus
from pymongo import MongoClient
from dateutil.relativedelta import relativedelta
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
MONGO_USER = os.getenv("MONGO_PRO_USER", "")
MONGO_PASS = os.getenv("MONGO_PRO_PASSWORD", "")
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
def sample_noticias_structured(n,year) -> List[Dict[str, Any]]:
  client = _get_mongo_client()
  db = client[MONGO_DB]
  col = db[MONGO_COL]
  # --- Filtrar por año si se pasa el parámetro ---
  if year:
    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)  # exclusivo
    pipeline = [{"$match": {"Fecha": {"$gte": start_date, "$lt": end_date}}},
                {"$sample": {"size": int(n)}},
          {"$sort": {"Fecha": -1}}  # -1 = descendente, 1 = ascendente
          ]
  else:
     pipeline = [{"$sample": {"size": int(n)}},
              {"$sort": {"Fecha": -1}}  # -1 = descendente, 1 = ascendente
              ] 
  docs = list(col.aggregate(pipeline))

  client.close()
  return docs

# =========================
# Muestreo y transformación
# =========================
def sample_last3months(n) -> List[Dict[str, Any]]:
  client = _get_mongo_client()
  db = client[MONGO_DB]
  col = db[MONGO_COL]
  hoy = datetime.now()
  hace_tres_meses = hoy - relativedelta(months=3)

  pipeline = [
      {
          "$match": {
              "Fecha": {"$gte": hace_tres_meses, "$lte": hoy}
          }
      },
      {"$sample": {"size": int(n)}},
      {"$sort": {"Fecha": -1}}
  ]
  docs = list(col.aggregate(pipeline))

  client.close()
  return docs


def sample_noticias_3months_complemento(n=10) -> List[Dict[str, Any]]:
    client = _get_mongo_client()
    db = client[MONGO_DB]
    col = db[MONGO_COL]

    # Rango de fechas
    fecha_inicio = datetime(2025, 1, 1)   # 1 de enero de 2025 #CAMBIAR CADA AÑO
    fecha_fin = datetime.today() - relativedelta(months=3)  # hoy - 3 meses

    pipeline = [
        {
            "$match": {
                "Fecha": {
                    "$gte": fecha_inicio,
                    "$lte": fecha_fin
                }
            }
        },
        {"$sample": {"size": int(n)}},
        {"$sort": {"Fecha": -1}}  # Orden descendente
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
        id_original1 = documento.get("_id", "")
        filename = f"{id_original1}.json"
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
        contenido_limitado = contenido[:8000] #aprox 2000 tokens
        doc.add_chunk(content=contenido_limitado)
        #attach metadata at document-level ---
        doc.update({"meta_fields": {
        "id_original":documento.get("_id", ""),
        "fecha": documento.get("Fecha", ""),
        "titulo": titulo,
        "url":documento.get("URL", ""),
        "medio":documento.get("Medio", ""),
        "categoria": documento.get("Categoria", ""),
        "entidades": documento.get("entidades", ""),
        "personas": documento.get("personas", ""),
        "relaciones":documento.get("relaciones_directas", "")
        }})

def cargar_docs_v2(coleccion, dataset):
    """
    Realiza la carga del chunk y de la metadata.
    Evita duplicados usando el id_original en meta_fields.
    """
    # Traemos todos los docs ya existentes con su id_original
    existentes = dataset.list_documents()
    ids_existentes = set()
    for d in existentes:
        # algunos SDKs exponen .meta_fields como objeto, otros como None
        meta = getattr(d, "meta_fields", None)

        if meta:
            # aseguramos dict
            if not isinstance(meta, dict):
                try:
                    meta = meta.__dict__  # convierte Base -> dict
                except:
                    meta = dict(meta)

            id_orig = str(meta.get("id_original", "")) if meta else ""
            if id_orig:
                ids_existentes.add(id_orig)

    for documento in coleccion:
        id_original1 = str(documento.get("_id", ""))

        if id_original1 in ids_existentes:
            print(f"Documento {id_original1} ya existe en el dataset, se omite.")
            continue

        titulo = slugify(documento.get("Titulo"))
        contenido = slugify(documento.get("Contenido"))
        filename = f"{id_original1}.json"
        json_bytes = json.dumps(documento, ensure_ascii=False).encode("utf-8")

        # --- Ingesta de documento ---
        doc = dataset.upload_documents([{
            "display_name": filename,
            "blob": json_bytes
        }])[0]
        print("Documento subido:", doc)

        # --- chunk limitado a 2000 tokens aprox ---
        contenido_limitado = contenido[:8000]
        doc.add_chunk(content=contenido_limitado)

        # --- metadata ---
        doc.update({
            "meta_fields": {
                "id_original": id_original1,
                "fecha": documento.get("Fecha", ""),
                "titulo": titulo,
                "url": documento.get("URL", ""),
                "medio": documento.get("Medio", ""),
                "categoria": documento.get("Categoria", ""),
                "entidades": documento.get("entidades", ""),
                "personas": documento.get("personas", ""),
                "relaciones": documento.get("relaciones_directas", "")
            }
        })

def delete_docs_older_than_3months(dataset) -> int:
    """
    Elimina de Ragflow todos los documentos cuya metadata 'fecha'
    sea anterior a 3 meses. Solo actúa en Ragflow.
    Retorna: número de documentos eliminados
    """
    hoy = datetime.now()
    hace_tres_meses = hoy - relativedelta(months=3)

    # Listar documentos en el dataset
    docs = dataset.list_documents()
    to_delete = []

    for d in docs:
        meta = getattr(d, "meta_fields", None)

        if meta:
            # aseguramos dict (igual que en cargar_docs_v2)
            if not isinstance(meta, dict):
                try:
                    meta = meta.__dict__
                except:
                    meta = dict(meta)

            fecha_str = str(meta.get("fecha", ""))  # OJO: 'fecha' en minúscula

            if fecha_str:
                try:
                    fecha_doc = datetime.fromisoformat(fecha_str)
                    if fecha_doc < hace_tres_meses:
                        to_delete.append(d.id)
                except Exception as e:
                    print(f"No se pudo parsear fecha '{fecha_str}' para doc {d.id}: {e}")

    if to_delete:
        #print(to_delete)
        dataset.delete_documents(to_delete)
        print(f"Se eliminaron {len(to_delete)} documentos de Ragflow.")
    else:
        print("No se encontraron documentos para eliminar.")

    return len(to_delete)
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
  out = sample_noticias_structured(2,2024)
  print(out)


