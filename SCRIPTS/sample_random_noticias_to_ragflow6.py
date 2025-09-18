# sample_random_noticias_to_ragflow.py
import uuid
import re
import os
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, date
from urllib.parse import quote_plus
import time
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
RAGFLOW_CHUNK_METHOD  = "manual"  # Cambiado a "manual" para control total sobre chunking
RAGFLOW_EMBED_MODEL="amazon.titan-embed-text-v2:0@Bedrock"

# ========= Bedrock (Titan) =========
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
TITAN_MODEL_ID = os.getenv("AWS_BEDROCK_EMBEDDINGS_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "1024"))
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

# ======== Chunking util ========
# Aproximación: ~4.7 chars por token (promedio), target 800 tokens/chunk, overlap 120 tokens
CH_TOKEN_RATIO = 4.7
TARGET_TOKENS = 1500
OVERLAP_TOKENS = 200
TARGET_CHARS = int(TARGET_TOKENS * CH_TOKEN_RATIO)       # ~3 760 chars
OVERLAP_CHARS = int(OVERLAP_TOKENS * CH_TOKEN_RATIO)     # ~560 chars

def split_into_paragraphs(text: str) -> list[str]:
    # Divide primero por párrafos dobles, con fallback
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    parts = [p for p in parts if p]
    return parts if parts else [text]

def chunk_text(text: str,
               max_chars: int = TARGET_CHARS,
               overlap_chars: int = OVERLAP_CHARS) -> list[str]:
    """
    Construye chunks cercanos a max_chars respetando párrafos,
    con solapamiento suave entre chunks.
    """
    if not text:
        return []
    paras = split_into_paragraphs(text)
    chunks = []
    buf = ""
    for p in paras:
        # Si el párrafo solo ya excede, lo cortamos en trozos duros
        if len(p) > max_chars:
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                piece = p[start:end]
                if piece.strip():
                    chunks.append(piece.strip())
                start = end - overlap_chars if end < len(p) else end
            continue

        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf.strip():
                chunks.append(buf.strip())
            # inicia siguiente buf con solapamiento desde el final del buf previo
            if overlap_chars > 0 and buf:
                tail = buf[-overlap_chars:]
                buf = (tail + "\n\n" + p).strip()
            else:
                buf = p

    if buf.strip():
        chunks.append(buf.strip())
    return chunks


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

    if titulo is not None:    metadata["titulo"] = titulo
    if url is not None:       metadata["url"] = url
    if medio is not None:     metadata["medio"] = medio
    if categoria is not None: metadata["categoria"] = categoria
    if entidades is not None: metadata["entidades"] = entidades
    if personas is not None:  metadata["personas"] = personas
    if rels is not None:      metadata["relaciones_directas"] = rels

    fecha_iso, fecha_epoch = (None, None)
    if fecha_v is not None:
        fecha_iso, fecha_epoch = _extract_fecha_iso_and_epoch(fecha_v)
        if fecha_iso:              metadata["fecha_iso"] = fecha_iso
        if fecha_epoch is not None:metadata["fecha_epoch_ms"] = fecha_epoch

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

def _norm_list(jr: dict) -> List[dict]:
    """
    Normaliza respuestas con shapes variables:
    - {"data": [...]} -> [...]
    - {"data": {"items":[...]}} -> [...]
    - {"data": {"list":[...]}} -> [...]
    - {"data": {"results":[...]}} -> [...]
    - {"data": {"datasets":[...]}} -> [...]
    - {"data": {"id":..., "name":...}} -> [data] (objeto único)
    - {"data": true/false/None} -> []  (nada iterable)
    """
    d = jr.get("data")
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for k in ("items", "list", "results", "datasets", "data"):
            v = d.get(k)
            if isinstance(v, list):
                return v
        # Si es un solo objeto con id/name, lo convertimos en lista de 1
        if any(k in d for k in ("id", "name", "dataset_id", "uuid")):
            return [d]
    # bool/None/otros -> lista vacía
    return []

def rf_get_or_create_dataset() -> str:
    if RAGFLOW_DATASET_ID:
        return RAGFLOW_DATASET_ID

    # 1) Intentar crear
    payload = {
        "name": RAGFLOW_DATASET_NAME,
        "chunk_method": RAGFLOW_CHUNK_METHOD,
        "embedding_model": RAGFLOW_EMBED_MODEL
    }
    url_create = f"{RAGFLOW_BASE_URL}/api/v1/datasets"
    r = requests.post(url_create, headers=rf_headers(), data=json.dumps(payload), timeout=60)
    try:
        jr = r.json()
    except Exception:
        jr = {}

    if r.status_code == 200:
        data = jr.get("data")
        # data puede ser dict con id, string, bool, etc.
        if isinstance(data, dict) and (data.get("id") or data.get("dataset_id") or data.get("uuid")):
            return data.get("id") or data.get("dataset_id") or data.get("uuid")
        if isinstance(data, str) and data.strip():
            return data  # algunas builds devuelven el id como string
        # si es True/False o no trae id, seguimos al listado

    # 2) Listar y encontrar por nombre (normalizando shapes)
    url_list = f"{RAGFLOW_BASE_URL}/api/v1/datasets"
    rl = requests.get(url_list, headers=rf_headers(), timeout=60)
    try:
        jr_list = rl.json()
    except Exception:
        jr_list = {}

    items = _norm_list(jr_list)
    for d in items:
        name = (d.get("name") or d.get("dataset_name") or "").strip()
        if name.lower() == RAGFLOW_DATASET_NAME.lower():
            return d.get("id") or d.get("dataset_id") or d.get("uuid")

    # 3) Si llegamos aquí, no pudimos extraer id
    raise RuntimeError(
        f"No pude obtener/crear dataset '{RAGFLOW_DATASET_NAME}'. "
        f"CreateResp={r.status_code} {jr}  ListResp={rl.status_code} {jr_list}"
    )


def _slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    return re.sub(r"_{2,}", "_", s).strip("_")[:80] or f"doc_{uuid.uuid4().hex[:8]}"

def rf_upload_json_document(dataset_id: str, name: str, content_text: str, metadata: Dict[str, Any]) -> str:
    """
    Sube un archivo .json (multipart) con:
      {
        "document_name": ...,
        "client_key": ...,
        "content": "...",
        "metadata": {...}
      }
    *El doc_id real lo devuelve el servidor en la respuesta.*
    """
    url = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/documents"
    unique_name = _slug(name)
    payload = {
        "document_name": unique_name,
        "client_key": metadata.get("id_original") or unique_name,
        "content": content_text,
        "metadata": {
            **metadata,
            "chunk_type": "complete_document",
            "is_complete": True,
            "content_length": len(content_text)
        }
    }
    body_bytes = json.dumps(payload, ensure_ascii=False, indent=2, default=json_default).encode("utf-8")

    files = {
        "file": (f"{unique_name}.json", body_bytes, "application/json")
    }
    data = { "name": unique_name }
    headers = {"Authorization": f"Bearer {RAGFLOW_API_KEY}"}

    r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    try:
        jr = r.json()
    except Exception:
        raise RuntimeError(f"upload_json_document bad JSON: {r.status_code} {r.text}")

    if r.status_code != 200 or jr.get("code") not in (0, None):
        raise RuntimeError(f"upload_json_document failed: HTTP {r.status_code} code={jr.get('code')} msg={jr.get('message')}")

    data_obj = jr.get("data")
    if isinstance(data_obj, dict):
        doc_id = data_obj.get("id") or (data_obj.get("document") or {}).get("id")
        if doc_id: return doc_id
    if isinstance(data_obj, list):
        for item in data_obj:
            if isinstance(item, dict):
                doc_id = item.get("id") or (item.get("document") or {}).get("id")
                if doc_id: return doc_id
    raise RuntimeError(f"No document id in response: {jr}")

def rf_add_chunk(dataset_id: str,
                 document_id: str,
                 chunk_text: str,
                 base_metadata: Dict[str, Any],
                 embedding: List[float],
                 chunk_index: int,
                 chunk_total: int) -> Dict[str, Any]:
    url = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/chunks"
    body = {
        "document_ids": [document_id],
        "text": chunk_text,
        "metadata": {
            **base_metadata,
            "chunk_type": "custom",
            "chunk_size": len(chunk_text),
            "is_complete": False,
            "document_complete": False,
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
        },
        "embedding": embedding,
        "embedding_dimension": len(embedding),
        "embedding_model": RAGFLOW_EMBED_MODEL,
        "doc_type": "news",
        "url": base_metadata.get("url"),
    }

    max_attempts = 10
    delay = 1.0  # segundos
    for attempt in range(1, max_attempts + 1):
        r = requests.post(url, headers=rf_headers(),
                          data=json.dumps(body, default=json_default), timeout=60)
        try:
            jr = r.json()
        except Exception:
            raise RuntimeError(f"add_chunk bad JSON: {r.status_code} {r.text}")

        code = jr.get("code")
        msg = (jr.get("message") or "").lower()

        # Éxito normal
        if r.status_code == 200 and code in (0, None):
            return jr

        # Documento aún procesándose -> reintentar con backoff
        if (r.status_code == 200 and code == 102) or "currently being processed" in msg:
            log.warning(f"Documento {document_id} aún en proceso. Reintento {attempt}/{max_attempts} en {delay:.1f}s…")
            time.sleep(delay)
            delay = min(delay * 1.8, 10.0)
            continue

        # Otros errores -> abortar
        raise RuntimeError(f"add_chunk failed: HTTP {r.status_code} code={code} msg={jr.get('message')}")

    raise RuntimeError(f"add_chunk failed after {max_attempts} retries: document stayed in processing state.")


def rf_add_complete_chunk(dataset_id: str,
                         document_id: str,
                         content_text: str,
                         metadata: Dict[str, Any],
                         embedding: List[float]) -> Dict[str, Any]:
    """
    Agrega un chunk que contiene el documento completo (sin dividir)
    """
    url = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/chunks"
    
    # El texto completo del documento será el contenido del chunk
    chunk_text = content_text or ""
    if not chunk_text.strip():
        chunk_text = metadata.get("titulo", "Documento sin contenido")
    
    body = {
        "document_ids": [document_id],  # Requerido por la API
        "text": chunk_text,  # Contenido completo sin dividir
        "metadata": {
            **metadata,  # Todos los metadatos originales
            "chunk_type": "complete_document",  # Marcador de que es documento completo
            "chunk_size": len(chunk_text),
            "is_complete": True,
            "document_complete": True
        },
        "embedding": embedding,
        "embedding_dimension": len(embedding),
        "embedding_model": RAGFLOW_EMBED_MODEL,
        "doc_type": "news",
        "url": metadata.get("url"),
    }
    
    r = requests.post(url, headers=rf_headers(),
                      data=json.dumps(body, default=json_default), timeout=60)
    try:
        jr = r.json()
    except Exception:
        raise RuntimeError(f"add_complete_chunk bad JSON: {r.status_code} {r.text}")

    if r.status_code != 200 or jr.get("code") not in (0, None):
        raise RuntimeError(f"add_complete_chunk failed: HTTP {r.status_code} code={jr.get('code')} msg={jr.get('message')}")
    return jr

# ========= Flujo principal =========
def main(n: int = 4):
    # 1) Mongo → sample
    mclient = mongo_client()
    col = mclient[MONGO_DB][MONGO_COL]
    pipeline = [{"$sample": {"size": int(n)}},
                {"$sort": {"Fecha": 1}}]
    raw_docs = list(col.aggregate(pipeline))
    mclient.close()

    log.info(f"Obtenidos {len(raw_docs)} documentos de MongoDB")

    # 2) Normalización a {content, metadata}
    items = [normalize_map(d) for d in raw_docs]

    # 3) Embeddings con Bedrock
    br = bedrock_client()
    

    # 4) Dataset
    dataset_id = rf_get_or_create_dataset()
    log.info(f"Usando dataset_id = {dataset_id}")

    # 5) Upload documentos con chunking custom
    uploaded_docs = []
    for i, x in enumerate(items):
        meta = x["metadata"].copy()
        doc_name = meta.get("titulo") or meta.get("url") or meta.get("id_original") or f"noticia_{i+1}"
        content_text = (x["content"]["contenido"] or "").strip()

        # 5.a) Crear el documento (una sola vez)
        log.info(f"Creando documento base {i+1}/{len(items)}: {doc_name[:60]}...")
        doc_id = rf_upload_json_document(
            dataset_id=dataset_id,
            name=str(doc_name),
            content_text="",       # opcional: puedes dejar vacío si subirás solo por chunks # dejamos vacío para minimizar parsing del server
            metadata=meta
        )
        log.info(f"  ✓ Documento creado con ID: {doc_id}")
        # Espera breve para que el backend registre el doc
        time.sleep(0.8) #evita el error (code=102: "Can't parse document that is currently being processed") que pasa cuando se intenta agregar chunks mientras
        #el backend de RAGFlow aún está “registrando / parsing” el documento recién creado.

        # 5.b) Generar chunks
        chunks = chunk_text(content_text, max_chars=TARGET_CHARS, overlap_chars=OVERLAP_CHARS)
        if not chunks:
            chunks = [content_text] if content_text else [meta.get("titulo", "Documento sin contenido")]

        # 5.c) Embedding y subida por chunk
        total = len(chunks)
        for ci, chunk in enumerate(chunks, start=1):
            log.info(f"   · Chunk {ci}/{total} ({len(chunk):,} chars) → embedding")
            emb = titan_text_embedding(br, chunk, dimensions=EMBED_DIMS)
            _ = rf_add_chunk(
                dataset_id=dataset_id,
                document_id=doc_id,
                chunk_text=chunk,
                base_metadata=meta,
                embedding=emb,
                chunk_index=ci,
                chunk_total=total
            )

        uploaded_docs.append({
            "document_id": doc_id,
            "name": doc_name,
            "content_length": len(content_text),
            "metadata": meta,
            "chunks": total
        })
        log.info(f"✓ Documento {i+1} subido con {total} chunks")
        
    
    # Resumen final
    total_chars = sum(doc["content_length"] for doc in uploaded_docs)
    log.info(f"📊 Resumen:")
    log.info(f"   - Total de documentos: {len(uploaded_docs)}")
    log.info(f"   - Total de caracteres: {total_chars:,}")
    log.info(f"   - Promedio por documento: {total_chars // len(uploaded_docs) if uploaded_docs else 0:,} caracteres")

if __name__ == "__main__":
    main(n=20)
    #en el retrieval testing parece que si se busca un nombre específico se ppone alto valor de similarity threshold y muy bajo de vector similarity.