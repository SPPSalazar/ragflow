#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingesta desde MongoDB → RAGFlow (doc engine: Infinity)
- Toma documentos aleatorios de Mongo y los ordena por fecha
- Crea 1 chunk por documento (solo 'contenido'), respetando presupuesto de tokens Titan v2
- Adjunta metadata (todo excepto Titulo/Contenido) a nivel documento
- Sube documentos/chunks a RAGFlow (Infinity) e indexa con embedding Titan v2
- Ejemplo de retrieve con rerank ColBERT

Requisitos:
  pip install ragflow-sdk python-dotenv boto3 pymongo

Variables .env relevantes (con defaults razonables):
  MONGO_PRO_HOST, MONGO_PRO_PORT, MONGO_PRO_USER, MONGO_PRO_PASSWORD, MONGO_PRO_DB, MONGO_PRO_COLLECTION
  RAGFLOW_BASE_URL, RAGFLOW_API_KEY, RAGFLOW_DATASET_ID, RAGFLOW_DATASET_NAME
  AWS_DEFAULT_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BEDROCK_EMBEDDINGS_ID, EMBED_DIMS
  SAMPLE_SIZE, TITAN_TOKEN_BUDGET
"""

import os
import json
import math
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from urllib.parse import quote_plus

from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from ragflow_sdk import RAGFlow

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# =========================
# Carga de entorno y logging
# =========================
load_dotenv(find_dotenv(usecwd=True))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("mongo→ragflow-infinity")

# ========= Mongo =========
MONGO_HOST = os.getenv("MONGO_PRO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PRO_PORT", "27017")
MONGO_USER = os.getenv("MONGO_PRO_USER", "")
MONGO_PASS = os.getenv("MONGO_PRO_PASSWORD", "")
MONGO_DB   = os.getenv("MONGO_PRO_DB", "noticias_db")
MONGO_COL  = os.getenv("MONGO_PRO_COLLECTION", "Noticias")

# ========= RAGFlow (Infinity) =========
RF_URL          = os.getenv("RAGFLOW_BASE_URL", "http://localhost:9380")

RAGFLOW_API_KEY       = os.getenv("RAGFLOW_API_KEY", "ragflow-Q1MWQ0ZjUyOTQxNzExZjBhNmQ0ZjI2Mj")

RF_DATASET_ID   = os.getenv("RAGFLOW_DATASET_ID", "")               # si ya existe
RF_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "noticias_kb")  # si no hay ID

# ========= Bedrock (Titan v2) =========
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
AWS_ACCESS   = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET   = os.getenv("AWS_SECRET_ACCESS_KEY", "")
TITAN_MODEL  = os.getenv("AWS_BEDROCK_EMBEDDINGS_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIMS   = int(os.getenv("EMBED_DIMS", "1024"))
TITAN_NORMALIZE = True

# ========= Control de lotes / tokens =========
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "10"))  # cuántos docs tomar aleatoriamente
# Titan v2 soporta ~8K tokens; usamos margen conservador (p.ej. 7000)
TITAN_TOKEN_BUDGET = int(os.getenv("TITAN_TOKEN_BUDGET", "7000"))

# ========= Utilidades =========
def mongo_client() -> MongoClient:
    if MONGO_USER and MONGO_PASS:
        uri = f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource=admin"
    else:
        uri = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client

def bedrock_client():
    retry_config = Config(
        retries={'max_attempts': 6, 'mode': 'adaptive'},
        read_timeout=300,
        connect_timeout=30
    )
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS or None,
        aws_secret_access_key=AWS_SECRET or None,
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
            modelId=TITAN_MODEL,
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

def approx_token_count(text: str) -> int:
    # Aproximación típica: ~4 caracteres por token en idiomas latinos
    # (ajustable si lo deseas)
    return math.ceil(len(text) / 4)

def safe_truncate_for_budget(text: str, token_budget: int) -> str:
    """
    Trunca con margen de seguridad para no exceder el budget.
    Conserva frases completas cuando es posible.
    """
    est_tokens = approx_token_count(text)
    if est_tokens <= token_budget:
        return text

    # Reducimos por longitud de caracteres proporcional al budget:
    # chars_per_token ~ 4 => max_chars = token_budget*4
    max_chars = token_budget * 4
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # intenta cortar en límite "limpio"
    last_break = max(truncated.rfind("\n"), truncated.rfind(". "), truncated.rfind(" "))
    if last_break > max_chars * 0.7:  # sólo si encontramos un corte razonable
        truncated = truncated[:last_break].rstrip()
    return truncated + "…"

def to_iso_date(val: Any) -> Optional[str]:
    """
    Convierte fechas variadas (Mongo Date, str, etc.) a 'YYYY-MM-DD'.
    Devuelve None si no se puede inferir.
    """
    if val is None:
        return None
    # Mongo Date puede venir como dict {"$date": "..."} o datetime
    if isinstance(val, dict) and "$date" in val:
        try:
            dt = datetime.fromisoformat(val["$date"].replace("Z","+00:00"))
            return dt.date().isoformat()
        except Exception:
            pass
    if isinstance(val, datetime):
        return val.date().isoformat()
    if isinstance(val, str):
        # intenta varios formatos comunes
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(val, fmt)
                return dt.date().isoformat()
            except Exception:
                continue
        # ISO flexible
        try:
            dt = datetime.fromisoformat(val.replace("Z","+00:00"))
            return dt.date().isoformat()
        except Exception:
            return None
    return None

def normalize_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Metadata = todo lo que NO es 'Titulo' ni 'Contenido'.
    Adicionalmente agrega campos derivados 'date_iso' y 'date_month' según 'Fecha' (si existe).
    """
    meta = {}
    for k, v in doc.items():
        if k.lower() in ("titulo", "contenido"):
            continue
        # convierte ObjectId/fechas raras a string legible
        if isinstance(v, datetime):
            meta[k] = v.isoformat()
        elif isinstance(v, dict) and "$date" in v:
            try:
                meta[k] = datetime.fromisoformat(v["$date"].replace("Z","+00:00")).isoformat()
            except Exception:
                meta[k] = str(v)
        else:
            meta[k] = str(v) if not isinstance(v, (int, float, bool, list, dict)) else v

    # Normaliza fechas útiles
    raw_fecha = doc.get("Fecha") or doc.get("fecha") or meta.get("Fecha") or meta.get("fecha")
    d_iso = to_iso_date(raw_fecha)
    if d_iso:
        meta["date_iso"] = d_iso                # YYYY-MM-DD
        meta["date_month"] = d_iso[:7]          # YYYY-MM
    return meta

def get_mongo_sample_sorted_by_date(client: MongoClient, size: int) -> List[Dict[str, Any]]:
    """
    1) $sample para aleatoriedad
    2) sort por fecha ascendente (si la hay)
    """
    coll = client[MONGO_DB][MONGO_COL]
    pipeline = [
        {"$sample": {"size": size}},
        # Fecha podría ser 'Fecha' o 'fecha' o un $date; ordenaremos luego en Python con robustez
    ]
    docs = list(coll.aggregate(pipeline))
    # Ordenar por fecha (usando to_iso_date robusto)
    def keyf(x):
        d = x.get("Fecha") or x.get("fecha")
        di = to_iso_date(d)
        return di or "9999-12-31"
    docs.sort(key=keyf)  # ascendente
    return docs

def ensure_dataset(rf: RAGFlow) -> Any:
    if RF_DATASET_ID:
        ds = rf.get_dataset(RF_DATASET_ID)
        if ds:
            return ds
        log.warning(f"No se encontró dataset por ID {RF_DATASET_ID}, se intentará por nombre…")
    datasets = rf.list_datasets(name=RF_DATASET_NAME)
    if datasets:
        return datasets[0]
    return rf.create_dataset(name=RF_DATASET_NAME)

def main():
    # ===== Conexiones =====
    try:
        client = mongo_client()
        log.info("✔ Conectado a MongoDB")
    except ServerSelectionTimeoutError as e:
        log.error(f"No se pudo conectar a MongoDB: {e}")
        return

    rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)
    dataset = ensure_dataset(rf)
    log.info(f"✔ Dataset RAGFlow listo: {dataset.id} ({dataset.name})")

    bedrock = bedrock_client()
    log.info(f"✔ Cliente Bedrock listo (region={AWS_REGION}, model={TITAN_MODEL})")

    # ===== 1) Muestreo aleatorio y orden por fecha =====
    docs = get_mongo_sample_sorted_by_date(client, SAMPLE_SIZE)
    if not docs:
        log.warning("No se obtuvieron documentos desde Mongo.")
        return
    log.info(f"→ Documentos obtenidos: {len(docs)} (aleatorios, ordenados por fecha ascendente)")

    uploaded = 0
    for i, d in enumerate(docs, 1):
        titulo = d.get("Titulo") or d.get("title") or d.get("Title") or "(sin título)"
        contenido = d.get("Contenido") or d.get("content") or d.get("texto") or ""

        if not contenido or not str(contenido).strip():
            log.warning(f"[{i}] Omitido por contenido vacío: {titulo}")
            continue

        content_str = str(contenido)
        content_trim = safe_truncate_for_budget(content_str, TITAN_TOKEN_BUDGET)

        # ===== 3) Metadata (todo excepto Titulo/Contenido) + normalizada =====
        meta = normalize_metadata(d)
        meta["title"] = titulo  # útil tenerlo también en meta
        # (si quieres, podrías también colocar el _id original)
        if "_id" in d:
            meta["id_original"] = str(d["_id"])

        # ===== 4) Crear documento y 1 chunk (solo contenido) =====
        # Nota: 'display_name' visible en UI; metadata va en 'meta_fields'
        doc = dataset.upload_documents([{"display_name": titulo}])[0]
        # Adjunta metadata a nivel documento
        doc.update({"meta_fields": meta})

        # ===== 5) Embedding explícito con Titan v2 =====
        try:
            vec = titan_text_embedding(bedrock, content_trim, EMBED_DIMS)
        except Exception as e:
            log.error(f"[{i}] Error embebiendo con Titan v2: {e}")
            # Fallback: deja que RAGFlow embeba (si está configurado un prov/model en el server)
            vec = None

        # ===== 6) Agregar el chunk y empujar a Infinity =====
        # Si vec != None, lo mandamos para forzar el embedding explícito;
        # si None, el servidor embeberá según su config.
        try:
            if vec is not None:
                # Algunos builds de ragflow_sdk aceptan 'vector' y 'embed_model' en add_chunk.
                ch = doc.add_chunk(
                    content=content_trim,
                    vector=vec,
                    embed_model=TITAN_MODEL  # opcional: anotación del modelo
                )
            else:
                ch = doc.add_chunk(
                    content=content_trim,
                    embed={"provider": "bedrock", "model": TITAN_MODEL}
                )
        except TypeError:
            # Compatibilidad si la firma difiere: usar forma básica y confiar en server-side embedding
            ch = doc.add_chunk(content=content_trim)

        uploaded += 1
        log.info(f"[{i}] OK: '{titulo}' | chunk_len={len(content_trim)} | meta_keys={len(meta)}")

    log.info(f"=======================================")
    log.info(f" Ingesta completada: {uploaded} documentos subidos")
    log.info(f" Doc engine: Infinity (vectores en índice Infinity; metadata en RAGFlow)")
    log.info(f"=======================================")

    # ===== 7) Ejemplo de Retrieve + Rerank ColBERT =====
    # Ajusta filtros si quieres (por ejemplo por mes o categoría, si existen en tu metadata)
    try:
        results = rf.retrieve(
            question="Resumen de las noticias más relevantes",
            dataset_ids=[dataset.id],
            metadata_condition={},  # e.g., {"date_month": {"$in": ["2025-05","2025-06"]}}
            top_k=50,
            rerank={"provider": "infinity", "model": "colbert"},
            rerank_top_k=10
        )
        log.info("— Resultados (top 10 tras rerank ColBERT) —")
        for j, ch in enumerate(results, 1):
            meta = ch.get("meta_fields", {})
            t = meta.get("title", "(sin título)")
            di = meta.get("date_iso", "")
            score = ch.get("score")
            preview = ch.get("content","")[:120].replace("\n"," ")
            log.info(f"{j:02d}. {t} | {di} | score={score:.4f} | {preview}{'…' if len(ch.get('content',''))>120 else ''}")
    except Exception as e:
        log.warning(f"No se pudo ejecutar retrieve con rerank ColBERT (revisa Infinity/ColBERT): {e}")

if __name__ == "__main__":
    main()
