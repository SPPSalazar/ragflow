# sample_random_noticias_structured_fixed.py
import os, json, re, logging
from typing import Any, Dict, List
from datetime import datetime, timezone
from urllib.parse import quote_plus

from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from bson.json_util import dumps as bson_dumps  # maneja datetime/BSON correctamente
from ragflow_sdk import RAGFlow

# ---------------------------
# Carga de entorno y logging
# ---------------------------
load_dotenv(find_dotenv(usecwd=True))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("mongo_sample_struct")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
dotenv_path = os.path.join(parent_dir, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# ---------------------------
# Parámetros MongoDB
# ---------------------------
MONGO_HOST = os.getenv("MONGO_PRO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PRO_PORT", "27017")
MONGO_USER = os.getenv("MONGO_PRO_USER", "")
MONGO_PASS = os.getenv("MONGO_PRO_PASSWORD", "")
MONGO_DB   = os.getenv("MONGO_PRO_DB", "noticias_db")
MONGO_COL  = os.getenv("MONGO_PRO_COLLECTION", "Noticias")

PROVENANCE_SOURCE = os.getenv("PROVENANCE_SOURCE", "unknown")
SCHEMA_VERSION    = os.getenv("SCHEMA_VERSION", "1.0")

# ---------------------------
# Utilidades
# ---------------------------
def _extract_fecha_iso_and_epoch(fecha_val: Any):
    try:
        if isinstance(fecha_val, dict) and "$date" in fecha_val:
            iso_text = str(fecha_val["$date"])
            dt = datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
        elif isinstance(fecha_val, datetime):
            dt = fecha_val
        elif isinstance(fecha_val, str):
            dt = datetime.fromisoformat(fecha_val.replace("Z", "+00:00"))
        elif isinstance(fecha_val, (int, float)):  # epoch ms
            dt = datetime.fromtimestamp(float(fecha_val)/1000.0, tz=timezone.utc)
        else:
            return None, None
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.isoformat().replace("+00:00", "Z"), int(dt_utc.timestamp()*1000)
    except Exception:
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
    client.admin.command("ping")
    return client

def sample_noticias_structured(n=3) -> List[Dict[str, Any]]:
    client = _get_mongo_client()
    col = client[MONGO_DB][MONGO_COL]
    docs = list(col.aggregate([{"$sample": {"size": int(n)}}]))
    client.close()
    return docs

def slugify(texto: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", str(texto), flags=re.UNICODE).strip().lower()
    s = re.sub(r"[-\s]+", "-", s)
    return s[:maxlen] if maxlen else s

# ---------------------------
# Muestreo desde Mongo
# ---------------------------
log.info(f"Conectando a MongoDB {MONGO_HOST}:{MONGO_PORT} DB={MONGO_DB}, Colección={MONGO_COL}")
noticias = sample_noticias_structured(n=3)

# ---------------------------
# RAGFlow
# ---------------------------
RF_URL          = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY      = os.getenv("RAGFLOW_API_KEY", "dev")
RF_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "a")

rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)

datasets = rf.list_datasets(name=RF_DATASET_NAME)
dataset  = datasets[0] if datasets else rf.create_dataset(name=RF_DATASET_NAME)

# ---------------------------
# Ingesta documento + chunk + metadatos
# ---------------------------
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
        {   "source": PROVENANCE_SOURCE,
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

    # Agregar el chunk con el contenido textual principal
    doc.add_chunk(content=contenido)

    # Metadatos útiles para filtrar posteriormente
    doc.update({"meta_fields": {
        "title": titulo,
        "url": url,
        "medio": medio,
        "category": categoria,
        "date_iso": fecha_iso,
        "date_month": date_month,
        "date_epoch_ms": fecha_epoch,
        "provenance": PROVENANCE_SOURCE
    }})

# ---------------------------
# Retrieve (sin metadata_condition si tu SDK no lo soporta)
# ---------------------------
chunks = rf.retrieve(
    question="Resumen de noticias",
    dataset_ids=[dataset.id],
    top_k=20,
    # rerank_id="rerank-multilingual-v3.0",  # actívalo si tu despliegue lo tiene
)

for i, ch in enumerate(chunks, 1):
    meta = ch.get("meta_fields", {}) or {}
    print(f"{i:02d}. {meta.get('title','(sin título)')} | {meta.get('date_iso')} | score={ch.get('score')}")
    content = ch.get("content", "") or ""
    print(f"    {content[:120]}{'...' if len(content)>120 else ''}")
