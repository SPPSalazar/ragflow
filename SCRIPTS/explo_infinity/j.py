# ragflow_minimal_ingest_and_retrieve.py
# Requires: pip install ragflow-sdk python-dotenv

import os
from datetime import datetime
from dateutil import parser
from dotenv import load_dotenv
from ragflow_sdk import RAGFlow

load_dotenv()
RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY = os.getenv("RAGFLOW_API_KEY", "dev")
RF_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "a")

rf = RAGFlow(api_key=RF_API_KEY, base_url=RF_URL)

# --- dataset (get or create) ---
datasets = rf.list_datasets(name=RF_DATASET_NAME)
dataset = datasets[0] if datasets else rf.create_dataset(name=RF_DATASET_NAME)

# --- sample record ---
noticias = [{
  "_id": {
    "$oid": "507f1f77bcf86cd799439011"
  },
  "title": "Ataque en la noche",
  "date": "2025-10-07T00:00:00.000Z",
  "content": "Un hombre fue encontrado muerto tras un ataque armado nocturno.",
  "category": [
    "asesinato"
  ]
},
{
  "_id": {
    "$oid": "507f1f77bcf86cd799439012"
  },
  "title": "Hombre asesinado parque",
  "date": "2025-10-11T00:00:00.000Z",
  "content": "Un hombre fue encontrado muerto tras un ataque armado nocturno.",
  "category": [
    "asesinato"
  ]
}]

# --- 1) create document ---
import json
import re
def slugify(texto, maxlen=80):
    s = re.sub(r"[^\w\s-]", "", str(texto), flags=re.UNICODE).strip().lower()
    s = re.sub(r"[-\s]+", "-", s)
    return s[:maxlen] if maxlen else s

for noticia in noticias:
    # --- normalize dates for metadata filters ---
    #dt = datetime.strptime(noticia["date"], "%m/%d/%Y")
    dt = parser.isoparse(noticia["date"])
    date_iso = dt.strftime("%Y-%m-%d")
    date_month = dt.strftime("%Y-%m")

    titulo = noticia.get("Titulo") or noticia.get("title") or "sin_titulo"
    contenido = noticia.get("Contenido") or noticia.get("content") or ""

    doc_json = {
        "content": {"titulo": titulo, "contenido": contenido},
        "metadata": {k: v for k, v in noticia.items() if k not in {"Titulo", "title", "Contenido", "content"}}
    }

    filename = f"{slugify(titulo)}.json"
    json_bytes = json.dumps(doc_json, ensure_ascii=False).encode("utf-8")
###############################################
#INGESTA DE DOCS
############
    # 👇 Cambia el blob: antes era (filename, json_bytes, "application/json")
    doc = dataset.upload_documents([{
        "display_name": filename,
        "blob": json_bytes  # <--- SOLO bytes
    }])[0]

    print("Documento subido:", doc)
    print("#############################################################################")
    # --- 2) add ONE chunk (only content) ---
    doc.add_chunk(content=noticia["content"])

    # --- 3) attach metadata at document-level ---
    doc.update({"meta_fields": {
        "title": noticia["title"],
        "date": noticia["date"],        # original
        "date_iso": date_iso,           # YYYY-MM-DD
        "date_month": date_month,       # YYYY-MM
        "category": noticia["category"]
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