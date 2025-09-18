# sample_random_noticias_to_ragflow.py
import uuid
import re

import os
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, date
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from bson.json_util import dumps

from bson import ObjectId
import numpy as np
from decimal import Decimal
# ========= Logging & .env =========
load_dotenv(find_dotenv(usecwd=True))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("noticias->ragflow")

# Si el .env vive en el directorio padre del script (como en tu ejemplo)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    dotenv_path = os.path.join(parent_dir, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
except Exception:
    pass

# ========= Mongo =========
MONGO_HOST = os.getenv("MONGO_PRO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PRO_PORT", "27017")
MONGO_USER = os.getenv("MONGO_PRO_USER", "")
MONGO_PASS = os.getenv("MONGO_PRO_PASSWORD", "")
MONGO_DB   = os.getenv("MONGO_PRO_DB", "noticias_db")
MONGO_COL  = os.getenv("MONGO_PRO_COLLECTION", "Noticias")

# ========= RAGFlow (Infinity) =========
RAGFLOW_BASE_URL      = os.getenv("RAGFLOW_BASE_URL", "http://localhost:9380")
RAGFLOW_API_KEY       = os.getenv("RAGFLOW_API_KEY", "ragflow-Q1MWQ0ZjUyOTQxNzExZjBhNmQ0ZjI2Mj")
RAGFLOW_DATASET_ID    = os.getenv("RAGFLOW_DATASET_ID", "")  # si ya lo tienes
RAGFLOW_DATASET_NAME  = os.getenv("RAGFLOW_DATASET_NAME", "noticias_kb")  # si no tienes ID
RAGFLOW_CHUNK_METHOD  = os.getenv("RAGFLOW_CHUNK_METHOD", "naive")
#RAGFLOW_EMBED_MODEL   = os.getenv("RAGFLOW_EMBED_MODEL", "amazon.titan-embed-text-v2@AWS")  # etiqueta en RAGFlow (ajusta)
RAGFLOW_EMBED_MODEL="amazon.titan-embed-text-v2:0@Bedrock"

# ========= Bedrock (Titan) =========
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
TITAN_MODEL_ID = os.getenv("AWS_BEDROCK_EMBEDDINGS_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "256"))
TITAN_NORMALIZE = True  # Titan v2 puede normalizar L2

# ========= Provenance =========
PROVENANCE_SOURCE   = os.getenv("PROVENANCE_SOURCE", "noticias_mongo")
SCHEMA_VERSION      = os.getenv("SCHEMA_VERSION", "1.0")

# ========= Utilidades =========
def bedrock_client():
    retry_config = Config(
        retries={'max_attempts': 6, 'mode': 'adaptive'},
        read_timeout=300,
        connect_timeout=30
    )
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID or None,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
        region_name=AWS_REGION
    )
    return session.client("bedrock-runtime", config=retry_config, region_name=AWS_REGION)

def titan_text_embedding(bedrock, text: str, dimensions: int = EMBED_DIMS) -> List[float]:
    body = {
        "inputText": text,
        "dimensions": dimensions,
        "normalize": TITAN_NORMALIZE,
        "embeddingTypes": ["float"]
    }
    try:
        resp = bedrock.invoke_model(
            modelId=TITAN_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        payload = json.loads(resp["body"].read())
        emb = payload.get("embedding")
        if emb is None:
            raise ValueError(f"No 'embedding' in Titan response: {payload}")
        return emb
    except ClientError as e:
        raise RuntimeError(f"Bedrock invoke_model failed: {e}")

def mongo_client():
    if MONGO_USER and MONGO_PASS:
        uri = f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource=admin"
    else:
        uri = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client

def _extract_fecha_iso_and_epoch(fecha_val: Any):
    from datetime import datetime, timezone
    try:
        if isinstance(fecha_val, dict) and "$date" in fecha_val:
            iso_text = str(fecha_val["$date"])
            dt = datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), int(dt.timestamp() * 1000)
        if isinstance(fecha_val, datetime):
            dt = fecha_val.astimezone(timezone.utc)
            return dt.isoformat().replace("+00:00", "Z"), int(dt.timestamp() * 1000)
        if isinstance(fecha_val, str):
            dt = datetime.fromisoformat(fecha_val.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), int(dt.timestamp() * 1000)
        if isinstance(fecha_val, (int, float)):
            dt = datetime.fromtimestamp(float(fecha_val) / 1000.0, tz=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z"), int(float(fecha_val))
    except Exception:
        pass
    return None, None

def normalize_map(doc: Dict[str, Any]) -> Dict[str, Any]:
    # posibles grafías
    keymap = {
        "titulo": ["titulo","Titulo","TITULO"],
        "contenido": ["contenido","Contenido","CONTENIDO"],
        "fecha": ["fecha","Fecha","FECHA"],
        "url": ["url","URL","Url"],
        "medio": ["medio","Medio","MEDIO"],
        "categoria": ["categoria","Categoría","Categoria","CATEGORIA","categoría"],
        "entidades": ["entidades","Entidades","ENTIDADES"],
        "personas": ["personas","Personas","PERSONAS"],
        "relaciones_directas": ["relaciones_directas","Relaciones_directas","RELACIONES_DIRECTAS"],
    }
    def pick_any(d, keys):
        for k in keys:
            if k in d: return d[k], k
        return None, None

    contenido, contenido_k = pick_any(doc, keymap["contenido"])
    titulo,    titulo_k    = pick_any(doc, keymap["titulo"])
    fecha_v,   fecha_k     = pick_any(doc, keymap["fecha"])
    url,       url_k       = pick_any(doc, keymap["url"])
    medio,     medio_k     = pick_any(doc, keymap["medio"])
    categoria, categoria_k = pick_any(doc, keymap["categoria"])
    entidades, entidades_k = pick_any(doc, keymap["entidades"])
    personas,  personas_k  = pick_any(doc, keymap["personas"])
    rels,      rels_k      = pick_any(doc, keymap["relaciones_directas"])

    metadata: Dict[str, Any] = {
        "id_original": str(doc.get("_id")) if "_id" in doc else None,
        "provenance": {
            "source": PROVENANCE_SOURCE,
            "ingested_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "schema_version": SCHEMA_VERSION
        },
        "extras": {}
    }

    if titulo is not None:   metadata["titulo"] = titulo
    if url is not None:      metadata["url"] = url
    if medio is not None:    metadata["medio"] = medio
    if categoria is not None:metadata["categoria"] = categoria
    if entidades is not None:metadata["entidades"] = entidades
    if personas is not None: metadata["personas"] = personas
    if rels is not None:     metadata["relaciones_directas"] = rels

    fecha_iso, fecha_epoch = (None, None)
    if fecha_v is not None:
        fecha_iso, fecha_epoch = _extract_fecha_iso_and_epoch(fecha_v)
        if fecha_iso:  metadata["fecha_iso"] = fecha_iso
        if fecha_epoch is not None: metadata["fecha_epoch_ms"] = fecha_epoch

    # mover “todo lo demás” a extras (salvo las claves ya mapeadas)
    mapped = {
        "_id", contenido_k, titulo_k, fecha_k, url_k, medio_k, categoria_k,
        entidades_k, personas_k, rels_k
    }
    mapped = {k for k in mapped if k}
    for k, v in doc.items():
        if k not in mapped:
            metadata["extras"][k] = v
    if fecha_k and fecha_k in doc:
        metadata["extras"].setdefault("Fecha_original", doc[fecha_k])

    return {
        "content": {"contenido": contenido},
        "metadata": metadata
    }

# ========= RAGFlow HTTP helpers =========
def rf_headers():
    return {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

#para serializar las fechas (json.dumps no sabe serializar datetime.)
def json_default(o):
    if isinstance(o, (datetime, date)):
        if isinstance(o, datetime) and o.tzinfo is None:
            o = o.replace(tzinfo=timezone.utc)
        return o.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(o, ObjectId):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, Decimal)):
        return float(o)
    if isinstance(o, (set, bytes)):
        return list(o) if isinstance(o, set) else o.decode("utf-8", "ignore")
    raise TypeError(f"Not JSON serializable: {type(o)}")


def rf_get_or_create_dataset() -> str:
    """
    Devuelve dataset_id. Si no se pasa RAGFLOW_DATASET_ID,
    intenta crear (o usar) uno por nombre.
    """
    if RAGFLOW_DATASET_ID:
        return RAGFLOW_DATASET_ID

    # Crear dataset mínimo (nombre, método de chunk, modelo de embedding)
    payload = {
        "name": RAGFLOW_DATASET_NAME,
        "chunk_method": RAGFLOW_CHUNK_METHOD,
        "embedding_model": RAGFLOW_EMBED_MODEL
    }
    url = f"{RAGFLOW_BASE_URL}/api/v1/datasets"
    r = requests.post(url, headers=rf_headers(), data=json.dumps(payload), timeout=60)
    # Si ya existe, algunas versiones devuelven error 400/409; entonces lo listamos y buscamos
    if r.status_code == 200:
        data = r.json().get("data")
        # algunos builds devuelven dict, otros boolean; intentamos extraer id si existe
        if isinstance(data, dict) and data.get("id"):
            return data["id"]

    # fallback: listar y hallar por nombre
    url_list = f"{RAGFLOW_BASE_URL}/api/v1/datasets"
    rl = requests.get(url_list, headers=rf_headers(), timeout=60)
    if rl.status_code == 200:
        for d in rl.json().get("data", []):
            if d.get("name", "").lower() == RAGFLOW_DATASET_NAME.lower():
                return d.get("id")
    raise RuntimeError(f"No pude obtener/crear dataset '{RAGFLOW_DATASET_NAME}'. Resp: {r.status_code} {r.text}")

def _slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    return re.sub(r"_{2,}", "_", s).strip("_")[:80] or f"doc_{uuid.uuid4().hex[:8]}"

def rf_upload_text_document(dataset_id: str, name: str, text: str) -> str:
    url = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/documents"
    # Si tu build usa /files, cambia la ruta arriba.

    # Nombre de archivo único para evitar colisiones
    unique_name = _slug(name)

    files = {
        "file": (f"{unique_name}.txt", (text or unique_name).encode("utf-8"), "text/plain")
    }
    data = {
        "name": unique_name
        # agrega flags si tu build los soporta, p.ej. "override": "true"
    }
    headers = {"Authorization": f"Bearer {RAGFLOW_API_KEY}"}  # no fijes Content-Type en multipart

    r = requests.post(url, headers=headers, files=files, data=data, timeout=60)

    try:
        jr = r.json()
    except Exception:
        raise RuntimeError(f"upload_text_document bad JSON: {r.status_code} {r.text}")

    if r.status_code != 200 or jr.get("code") not in (0, None):
        raise RuntimeError(f"upload_text_document failed: HTTP {r.status_code} code={jr.get('code')} msg={jr.get('message')}")

    data_obj = jr.get("data")

    # Soporta ambos formatos: dict o list
    if isinstance(data_obj, dict):
        doc_id = data_obj.get("id") or (data_obj.get("document") or {}).get("id")
        if doc_id:
            return doc_id

    if isinstance(data_obj, list):
        # toma el primero con id válido
        for item in data_obj:
            if isinstance(item, dict):
                doc_id = item.get("id") or (item.get("document") or {}).get("id")
                if doc_id:
                    return doc_id

    raise RuntimeError(f"No document id in response: {jr}")

def rf_create_document(dataset_id: str, name: str, metadata: Dict[str, Any]) -> str:
    url = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/documents"
    body = {
        "name": name,
        "metadata": metadata  # opcional
    }
    r = requests.post(url, headers=rf_headers(),
                      data=json.dumps(body, default=json_default), timeout=60)
    jr = r.json()
    if r.status_code != 200 or jr.get("code") not in (0, None):
        raise RuntimeError(f"create_document failed: HTTP {r.status_code} code={jr.get('code')} msg={jr.get('message')}")
    return jr["data"]["id"]  # ajusta si tu payload difiere


def rf_add_chunk(dataset_id: str,
                 document_id: str,      # <-- nuevo
                 content_text: str,
                 metadata: Dict[str, Any],
                 embedding: List[float]) -> Dict[str, Any]:

    url = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/chunks"
    body = {
        "document_ids": [document_id],   # <-- requerido por tu build
        "text": content_text,
        "metadata": metadata,
        "embedding": embedding,
        "embedding_dimension": len(embedding),
        "embedding_model": RAGFLOW_EMBED_MODEL,
        "doc_type": "news",
        "url": metadata.get("url"),
    }
    r = requests.post(url, headers=rf_headers(),
                      data=json.dumps(body, default=json_default), timeout=60)
    jr = r.json()
    if r.status_code != 200 or jr.get("code") not in (0, None):
        raise RuntimeError(f"add_chunk failed: HTTP {r.status_code} code={jr.get('code')} msg={jr.get('message')}")
    return jr


# ========= Flujo principal =========
def main(n: int = 3):
    # 1) Mongo → sample
    mclient = mongo_client()
    col = mclient[MONGO_DB][MONGO_COL]
    pipeline = [{"$sample": {"size": int(n)}}]
    raw_docs = list(col.aggregate(pipeline))
    mclient.close()

    # 2) Normalización a {content, metadata}
    items = [normalize_map(d) for d in raw_docs]

    # 3) Embeddings con Bedrock
    br = bedrock_client()
    for x in items:
        text = (x["content"] or {}).get("contenido") or ""
        emb = titan_text_embedding(br, text, dimensions=EMBED_DIMS)
        x["embedding"] = emb  # guardamos también localmente por si quieres persistir

    # 4) RAGFlow: dataset & upsert chunks
    dataset_id = rf_get_or_create_dataset()
    log.info(f"Usando dataset_id = {dataset_id}")

    results = []
    for x in items:
        meta = x["metadata"].copy()
        # nombre del documento para RAGFlow (preferencia: título; si no, URL; si no, id)
        doc_name = meta.get("titulo") or meta.get("url") or meta.get("id_original") or "noticia"
        seed_text = (x["content"]["contenido"] or "")[:2000] or doc_name  # algo corto para el .txt
        doc_id = rf_upload_text_document(dataset_id, str(doc_name), seed_text)

        resp = rf_add_chunk(
            dataset_id=dataset_id,
            document_id=doc_id,
            content_text=x["content"]["contenido"] or "",
            metadata=meta,
            embedding=x["embedding"]
        )
        results.append(resp)



    # 5) Log/print
    #print(json.dumps({
    #    "inserted": results,
    #    "payload_preview": json.loads(dumps(items, ensure_ascii=False))  # para inspección local
    #}, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main(n=3)
