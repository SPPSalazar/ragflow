# sample_random_noticias_structured.py
# ragflow_minimal_ingest_and_retrieve.py
# Requires: pip install ragflow-sdk python-dotenv

import os
from datetime import datetime
from dateutil import parser
from dotenv import load_dotenv
from ragflow_sdk import RAGFlow

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote_plus
from pymongo import MongoClient
from bson.json_util import dumps

# =========================
# Carga de entorno y logging
# =========================
load_dotenv(find_dotenv(usecwd=True))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("mongo_sample_struct")

# Si tu .env vive en el directorio padre del script:
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dotenv_path = os.path.join(parent_dir, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# =========================
# Parámetros MongoDB
# =========================
MONGO_HOST = os.getenv("MONGO_PRO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PRO_PORT", "27017")
MONGO_USER = os.getenv("MONGO_PRO_USER", "")
MONGO_PASS = os.getenv("MONGO_PRO_PASSWORD", "")
MONGO_DB   = os.getenv("MONGO_PRO_DB", "noticias_db")
MONGO_COL  = os.getenv("MONGO_PRO_COLLECTION", "Noticias")

# Parámetros de provenance
PROVENANCE_SOURCE = os.getenv("PROVENANCE_SOURCE", "unknown")
SCHEMA_VERSION = os.getenv("SCHEMA_VERSION", "1.0")

# =========================
# Utilidades
# =========================
def _extract_fecha_iso_and_epoch(fecha_val: Any):
    """
    Acepta:
      - {"$date": "2025-05-07T14:03:00.000Z"} (estilo Mongo export/extended JSON)
      - datetime nativo
      - string ISO
      - epoch ms (int/float)
    Devuelve: (iso_str, epoch_ms) o (None, None) si no pudo convertir.
    """
    try:
        # Caso dict estilo {"$date": "..."}
        if isinstance(fecha_val, dict) and "$date" in fecha_val:
            iso_text = str(fecha_val["$date"])
            # Asegura Z al final si falta zona
            dt = datetime.fromisoformat(
                iso_text.replace("Z", "+00:00")
            )
            epoch_ms = int(dt.timestamp() * 1000)
            iso_out = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            return iso_out, epoch_ms

        # datetime nativo
        if isinstance(fecha_val, datetime):
            dt = fecha_val.astimezone(timezone.utc)
            return dt.isoformat().replace("+00:00", "Z"), int(dt.timestamp() * 1000)

        # string ISO
        if isinstance(fecha_val, str):
            dt = datetime.fromisoformat(fecha_val.replace("Z", "+00:00"))
            epoch_ms = int(dt.timestamp() * 1000)
            iso_out = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            return iso_out, epoch_ms

        # epoch ms numérico
        if isinstance(fecha_val, (int, float)):
            dt = datetime.fromtimestamp(float(fecha_val) / 1000.0, tz=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z"), int(float(fecha_val))

    except Exception:
        pass

    return None, None


def _get_mongo_client():
    if MONGO_USER and MONGO_PASS:
        uri = (
            f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}"
            f"@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource=admin"
        )
    else:
        uri = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # Verificar conexión
    return client


# =========================
# Muestreo y transformación
# =========================
def sample_noticias_structured(n=3) -> List[Dict[str, Any]]:
    client = _get_mongo_client()
    db = client[MONGO_DB]
    col = db[MONGO_COL]

    pipeline = [{"$sample": {"size": int(n)}}]
    docs = list(col.aggregate(pipeline))
   
    client.close()
    return docs



log.info(
    f"Conectando a MongoDB {MONGO_HOST}:{MONGO_PORT} "
    f"DB={MONGO_DB}, Colección={MONGO_COL}"
)
out = sample_noticias_structured(n=3)

#        print(dumps(out, ensure_ascii=False, indent=2))
#        log.info(f"Total documentos devueltos: {len(out)}")
#    except Exception as e:
#        log.error(f"Error al muestrear/transformar documentos: {e}")
#        raise


load_dotenv()
RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY = os.getenv("RAGFLOW_API_KEY", "dev")
RF_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "a")

rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)

# --- dataset (get or create) ---
datasets = rf.list_datasets(name=RF_DATASET_NAME)
dataset = datasets[0] if datasets else rf.create_dataset(name=RF_DATASET_NAME)

# --- sample record ---
noticias = out
# --- 1) create document ---
import json
import re
def slugify(texto, maxlen=80):
    s = re.sub(r"[^\w\s-]", "", str(texto), flags=re.UNICODE).strip().lower()
    s = re.sub(r"[-\s]+", "-", s)
    return s[:maxlen] if maxlen else s

for noticia in noticias:
    titulo    = noticia.get("Titulo")   or noticia.get("title")    or "sin_titulo"
    contenido = noticia.get("Contenido") or noticia.get("content") or ""
    url       = noticia.get("URL")      or noticia.get("url")
    medio     = noticia.get("Medio")    or noticia.get("medio")
    categoria = noticia.get("Categoria") or noticia.get("category")

    fecha_iso, fecha_epoch = _extract_fecha_iso_and_epoch(noticia.get("Fecha"))
    date_month = fecha_iso[:7] if fecha_iso else None  # YYYY-MM

    # Archivo "dummy" (JSON) con la noticia serializada (maneja datetime con bson.json_util)
    filename   = f"{slugify(titulo)}.json"
    json_text  = bson_dumps(
        {
            "source": PROVENANCE_SOURCE,
            "schema_version": SCHEMA_VERSION,
            "mongo_doc": noticia
        },
        ensure_ascii=False
    )
    json_bytes = json_text.encode("utf-8")

    # Subir documento (SDK típico: displayed_name + blob como (name, bytes, mime))
    doc = dataset.upload_documents([{
        "displayed_name": filename,
        "blob": (filename, json_bytes, "application/json")
    }])[0]
    # --- 3) attach metadata at document-level ---
    doc.update({"meta_fields": {
        "title": noticia["title"],
        "date": noticia["Fecha"],        # original
#        "date_iso": date_iso,           # YYYY-MM-DD
#        "date_month": date_month,       # YYYY-MM
#        "category": noticia["category"]
    }})
###################
# --- retrieve with strict month filter---
chunks = rf.retrieve(
    question="Resumen de noticias",
    dataset_ids=[dataset.id],
    metadata_condition={
        "date_iso": {"$gte": "2025-07-01", "$lt": "2025-08-01"},
        "category": {"$in": ["category1"]}  # change as needed
    },
    rerank_id=None,#"rerank-multilingual-v3.0",
    top_k=50,          # initial candidates
    #rerank_top_k=10    # final reranked results
)

# --- print simple view ---
for i, ch in enumerate(chunks, 1):
    meta = ch.get("meta_fields", {})
    print(f"{i:02d}. {meta.get('title','(no title)')} | {meta.get('date_iso')} | score={ch.get('score')}")
    print(f"    {ch.get('content','')[:120]}{'...' if len(ch.get('content',''))>120 else ''}")
ds = rf.list_datasets(name="Noticias")
print(ds[0])
print("############################################################")
ch = doc.list_chunks()[0]
print(ch) 