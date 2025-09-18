#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, logging, sys
from typing import Any, Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv, find_dotenv

# ================== Config & logging ==================
load_dotenv(find_dotenv(usecwd=True))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ragflow-export")

RAGFLOW_BASE_URL     = os.getenv("RAGFLOW_BASE_URL", "http://localhost:9380")
RAGFLOW_API_KEY       = os.getenv("RAGFLOW_API_KEY", "ragflow-Q1MWQ0ZjUyOTQxNzExZjBhNmQ0ZjI2Mj")
RAGFLOW_DATASET_ID   = os.getenv("RAGFLOW_DATASET_ID", "")  # opcional
RAGFLOW_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "noticias_kb")  # fallback si no hay ID
OUT_PATH             = os.getenv("RAGFLOW_EXPORT_PATH", f"ragflow_export_{int(time.time())}.jsonl")
PAGE_SIZE            = int(os.getenv("RAGFLOW_PAGE_SIZE", "100"))  # tamaño de página sugerido

# ================== Helpers HTTP ==================
def rf_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

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
        if any(k in d for k in ("id", "name", "dataset_id", "uuid")):
            return [d]
    return []

def _get_next_token(jr: dict) -> Optional[str]:
    """
    Intenta encontrar un campo de paginación común:
    - jr['data']['next_page_token'] o jr['next_page_token']
    - jr['data']['next'] / jr['next']
    """
    for path in [
        ("data", "next_page_token"),
        ("next_page_token",),
        ("data", "next"),
        ("next",),
    ]:
        d = jr
        ok = True
        for k in path:
            if not isinstance(d, dict) or k not in d:
                ok = False
                break
            d = d[k]
        if ok and isinstance(d, str) and d.strip():
            return d.strip()
    return None

# ================== Dataset discovery ==================
def rf_get_dataset_id() -> str:
    """
    Si ya hay RAGFLOW_DATASET_ID, lo usa.
    Si no, lista datasets y busca por nombre (RAGFLOW_DATASET_NAME).
    """
    if RAGFLOW_DATASET_ID:
        return RAGFLOW_DATASET_ID

    url = f"{RAGFLOW_BASE_URL}/api/v1/datasets"
    params = {"page_size": PAGE_SIZE}
    token = None
    while True:
        if token:
            params["page_token"] = token
        r = requests.get(url, headers=rf_headers(), params=params, timeout=60)
        try:
            jr = r.json()
        except Exception:
            raise RuntimeError(f"datasets list bad JSON: {r.status_code} {r.text}")

        items = _norm_list(jr)
        for d in items:
            name = (d.get("name") or d.get("dataset_name") or "").strip()
            if name.lower() == (RAGFLOW_DATASET_NAME or "").lower():
                return d.get("id") or d.get("dataset_id") or d.get("uuid")

        token = _get_next_token(jr)
        if not token:
            break

    raise RuntimeError(
        f"No se encontró dataset con nombre '{RAGFLOW_DATASET_NAME}'. "
        f"Define RAGFLOW_DATASET_ID en tu .env o ajusta RAGFLOW_DATASET_NAME."
    )

# ================== List documents (multi-endpoint) ==================
def rf_list_documents(dataset_id: str) -> List[dict]:
    """
    Intenta listar documentos probando varias rutas/rúbricas de RAGFlow.
    Si todo falla, devuelve [] (y el export hará fallback por chunks).
    """
    headers = rf_headers()
    tries = [
        (f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/documents", {"page_size": PAGE_SIZE}),
        (f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/documents/list", {"page_size": PAGE_SIZE}),
        (f"{RAGFLOW_BASE_URL}/api/v1/documents", {"dataset_id": dataset_id, "page_size": PAGE_SIZE}),
        (f"{RAGFLOW_BASE_URL}/api/v1/documents/list", {"dataset_id": dataset_id, "page_size": PAGE_SIZE}),
    ]

    all_docs: List[dict] = []
    for url, base_params in tries:
        params = dict(base_params)
        token = None
        try:
            while True:
                if token:
                    params["page_token"] = token
                    params["next"] = token
                r = requests.get(url, headers=headers, params=params, timeout=60)
                jr = r.json()
                items = _norm_list(jr)
                if items:
                    all_docs.extend(items)
                token = _get_next_token(jr)
                if not token:
                    break
        except Exception as e:
            log.debug(f"[rf_list_documents] Falló {url} ({e}); probando siguiente…")
            continue

        if all_docs:
            break

    # dedup por id
    seen = set()
    uniq = []
    for d in all_docs:
        did = d.get("id") or d.get("document_id") or d.get("uuid")
        if did and did not in seen:
            seen.add(did)
            uniq.append(d)
    return uniq

# ================== List chunks ==================
def rf_list_chunks_for_document(dataset_id: str, document_id: str) -> List[dict]:
    """
    Devuelve todos los chunks de un documento.
    Algunas builds aceptan query param 'document_id', otros 'document_ids'.
    Pagina y filtra por si el backend ignora el filtro.
    """
    base_url = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/chunks"
    out: List[dict] = []

    def _fetch(param_name: str) -> List[dict]:
        params = {param_name: document_id, "page_size": PAGE_SIZE}
        token = None
        buff: List[dict] = []
        while True:
            if token:
                params["page_token"] = token
                params["next"] = token
            r = requests.get(base_url, headers=rf_headers(), params=params, timeout=120)
            try:
                jr = r.json()
            except Exception:
                raise RuntimeError(f"chunks list bad JSON: {r.status_code} {r.text}")
            items = _norm_list(jr)
            buff.extend(items)
            token = _get_next_token(jr)
            if not token:
                break
        return buff

    items = _fetch("document_id")
    if not items:
        items = _fetch("document_ids")

    filtered = []
    for it in items:
        did = (
            it.get("document_id")
            or (it.get("document") or {}).get("id")
            or (it.get("document") or {}).get("document_id")
        )
        if not did or str(did) == str(document_id):
            filtered.append(it)
    return filtered

def rf_list_all_chunks(dataset_id: str) -> List[dict]:
    """
    Lista TODOS los chunks de un dataset (sin filtrar por document_id), paginando.
    Útil como fallback cuando la API de documentos no responde.
    """
    headers = rf_headers()
    urls = [
        f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/chunks",
        f"{RAGFLOW_BASE_URL}/api/v1/chunks",  # por si el backend exige dataset_id como query
    ]
    out: List[dict] = []

    for url in urls:
        params = {"page_size": PAGE_SIZE, "dataset_id": dataset_id}
        token = None
        try:
            while True:
                if token:
                    params["page_token"] = token
                    params["next"] = token
                r = requests.get(url, headers=headers, params=params, timeout=120)
                jr = r.json()
                items = _norm_list(jr)
                out.extend(items)
                token = _get_next_token(jr)
                if not token:
                    break
        except Exception as e:
            log.debug(f"[rf_list_all_chunks] Falló {url} ({e}); probando siguiente…")
            continue

        if out:
            break
    return out

# ================== Export core ==================
def export_dataset_to_jsonl(dataset_id: str, out_path: str) -> Tuple[int, int]:
    """
    Exporta todos los chunks del dataset a un JSONL.
    Si la API no devuelve documentos, hace fallback listando todos los chunks
    y reconstruyendo document_ids únicos.
    """
    docs = rf_list_documents(dataset_id)
    if not docs:
        log.warning("No se pudieron listar documentos por la(s) ruta(s) estándar. "
                    "Haré fallback: listaré todos los chunks del dataset y reconstruiré los documentos.")
        chunks = rf_list_all_chunks(dataset_id)
        grouped: Dict[str, List[dict]] = {}
        for ch in chunks:
            did = (
                ch.get("document_id")
                or (ch.get("document") or {}).get("id")
                or (ch.get("document") or {}).get("document_id")
            )
            if not did and isinstance(ch.get("document"), dict):
                did = ch["document"].get("uuid") or ch["document"].get("id")
            did = str(did) if did is not None else None
            if not did:
                continue
            grouped.setdefault(did, []).append(ch)

        log.info(f"Dataset {dataset_id} (fallback): {len(grouped)} documentos reconstruidos desde chunks.")
        n_chunks_total = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (doc_id, doc_chunks) in enumerate(grouped.items(), start=1):
                # nombre/metadata del doc (best-effort)
                doc_name = None
                doc_meta = {}
                for ch in doc_chunks:
                    maybe = (ch.get("document") or {}).get("name") or (ch.get("document") or {}).get("title")
                    if maybe:
                        doc_name = maybe
                        break
                for ch in doc_chunks:
                    if ch.get("metadata"):
                        doc_meta = ch["metadata"]; break

                log.info(f"[{i}/{len(grouped)}] Doc {doc_id} ({(str(doc_name) or '')[:60]}...) → {len(doc_chunks)} chunks")
                for ch in doc_chunks:
                    rec = {
                        "dataset_id": dataset_id,
                        "document_id": doc_id,
                        "document_name": doc_name,
                        "text": ch.get("text") or ch.get("content") or ch.get("chunk_text"),
                        "metadata": ch.get("metadata") or doc_meta or {},
                        "embedding": ch.get("embedding") or ch.get("vector") or [],
                        "embedding_dimension": (
                            ch.get("embedding_dimension")
                            or (len(ch.get("embedding")) if isinstance(ch.get("embedding"), list) else None)
                        ),
                        "embedding_model": ch.get("embedding_model"),
                        "chunk_index": ch.get("metadata", {}).get("chunk_index") or ch.get("chunk_index"),
                        "chunk_total": ch.get("metadata", {}).get("chunk_total") or ch.get("chunk_total"),
                        "created_at": ch.get("created_at") or ch.get("create_time"),
                        "updated_at": ch.get("updated_at") or ch.get("update_time"),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_chunks_total += 1
        return len(grouped), n_chunks_total

    # Camino normal (sí hubo documentos)
    log.info(f"Dataset {dataset_id}: {len(docs)} documentos encontrados.")
    n_chunks_total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs, start=1):
            doc_id = d.get("id") or d.get("document_id") or d.get("uuid")
            doc_name = d.get("name") or d.get("document_name") or d.get("title")
            if not doc_id:
                log.warning(f"[Doc {i}] No encuentro id en {list(d.keys())}, lo salto.")
                continue

            log.info(f"[{i}/{len(docs)}] Doc {doc_id} ({(doc_name or '')[:60]}...) → list chunks")
            chunks = rf_list_chunks_for_document(dataset_id, doc_id)
            log.info(f"   · {len(chunks)} chunks")

            for ch in chunks:
                rec = {
                    "dataset_id": dataset_id,
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "text": ch.get("text") or ch.get("content") or ch.get("chunk_text"),
                    "metadata": ch.get("metadata") or (d.get("metadata") or {}),
                    "embedding": ch.get("embedding") or ch.get("vector") or [],
                    "embedding_dimension": (
                        ch.get("embedding_dimension")
                        or (len(ch.get("embedding")) if isinstance(ch.get("embedding"), list) else None)
                    ),
                    "embedding_model": ch.get("embedding_model"),
                    "chunk_index": ch.get("metadata", {}).get("chunk_index") or ch.get("chunk_index"),
                    "chunk_total": ch.get("metadata", {}).get("chunk_total") or ch.get("chunk_total"),
                    "created_at": ch.get("created_at") or ch.get("create_time"),
                    "updated_at": ch.get("updated_at") or ch.get("update_time"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks_total += 1
            time.sleep(0.05)
    return len(docs), n_chunks_total

# ================== Main ==================
def main():
    try:
        dataset_id = rf_get_dataset_id()
        log.info(f"Exportando dataset_id = {dataset_id} a {OUT_PATH} …")
        n_docs, n_chunks = export_dataset_to_jsonl(dataset_id, OUT_PATH)
        log.info("======================================")
        log.info(f"✅ Export completado")
        log.info(f"   • Documentos: {n_docs}")
        log.info(f"   • Chunks:     {n_chunks}")
        log.info(f"   • Archivo:    {OUT_PATH}")
    except Exception as e:
        log.error(f"Fallo en export: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
