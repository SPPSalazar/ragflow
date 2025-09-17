# sample_random_noticias_structured.py
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


def _normalize_and_map(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplica la estructura:
    {
      "content": {"contenido": "..."},
      "metadata": {
        "id_original": "...",
        "fecha_iso": "RFC3339",
        "fecha_epoch_ms": 1715080980000,
        "url": "...",
        "medio": "...",
        "categoria": "...",
        "entidades": [...],
        "personas": [...],
        "relaciones_directas": [...],
        "extras": {...},
        "provenance": {...}
      }
    }
    Reglas:
    - _id -> metadata.id_original (string)
    - Fecha -> metadata.fecha_iso y metadata.fecha_epoch_ms; conservar original en extras["Fecha_original"]
    - Titulo -> metadata.titulo (normalizo a minúsculas la clave, no el contenido)
    - Contenido -> content.contenido (texto tal cual)
    - URL/Medio/Categoria (insensible a mayúsculas) -> metadata.url/medio/categoria
    - entidades/personas/relaciones_directas -> copiar a metadata
    - Todo campo distinto de Titulo/Contenido y no listado -> extras
    """
    # Posibles claves con distintas grafías
    keymap_candidates = {
        "titulo": ["titulo", "Titulo", "TITULO"],
        "contenido": ["contenido", "Contenido", "CONTENIDO"],
        "fecha": ["fecha", "Fecha", "FECHA"],
        "url": ["url", "URL", "Url"],
        "medio": ["medio", "Medio", "MEDIO"],
        "categoria": ["categoria", "Categoría", "Categoria", "CATEGORIA", "categoría"],
        "entidades": ["entidades", "Entidades", "ENTIDADES"],
        "personas": ["personas", "Personas", "PERSONAS"],
        "relaciones_directas": ["relaciones_directas", "Relaciones_directas", "RELACIONES_DIRECTAS"],
    }

    def pick_any(d: Dict[str, Any], keys: List[str]):
        for k in keys:
            if k in d:
                return d[k], k
        return None, None

    content_text = None
    content_key = None

    # Extrae contenido
    content_text, content_key = pick_any(doc, keymap_candidates["contenido"])

    # Extrae campos mapeados
    titulo_val, titulo_key = pick_any(doc, keymap_candidates["titulo"])
    fecha_val, fecha_key = pick_any(doc, keymap_candidates["fecha"])
    url_val, url_key = pick_any(doc, keymap_candidates["url"])
    medio_val, medio_key = pick_any(doc, keymap_candidates["medio"])
    categoria_val, categoria_key = pick_any(doc, keymap_candidates["categoria"])
    entidades_val, entidades_key = pick_any(doc, keymap_candidates["entidades"])
    personas_val, personas_key = pick_any(doc, keymap_candidates["personas"])
    rel_dir_val, rel_dir_key = pick_any(doc, keymap_candidates["relaciones_directas"])

    # Construye metadata básica
    metadata: Dict[str, Any] = {}
    metadata["id_original"] = str(doc.get("_id")) if "_id" in doc else None

    # Fecha ISO + epoch
    fecha_iso, fecha_epoch = (None, None)
    if fecha_val is not None:
        fecha_iso, fecha_epoch = _extract_fecha_iso_and_epoch(fecha_val)
    if fecha_iso:
        metadata["fecha_iso"] = fecha_iso
    if fecha_epoch is not None:
        metadata["fecha_epoch_ms"] = fecha_epoch

    # Campos directos
    if url_val is not None:
        metadata["url"] = url_val
    if medio_val is not None:
        metadata["medio"] = medio_val
    if categoria_val is not None:
        metadata["categoria"] = categoria_val
    if titulo_val is not None:
        metadata["titulo"] = titulo_val

    # Arrays
    if entidades_val is not None:
        metadata["entidades"] = entidades_val
    if personas_val is not None:
        metadata["personas"] = personas_val
    if rel_dir_val is not None:
        metadata["relaciones_directas"] = rel_dir_val

    # Extras: todo lo que no sea Contenido/Titulo y no esté mapeado arriba
    mapped_keys = {
        "_id",
        content_key, titulo_key, fecha_key, url_key, medio_key, categoria_key,
        entidades_key, personas_key, rel_dir_key
    }
    mapped_keys = {k for k in mapped_keys if k}  # quita None

    extras: Dict[str, Any] = {}
    for k, v in doc.items():
        if k not in mapped_keys:
            extras[k] = v

    # Conservar fecha original dentro de extras (si existía)
    if fecha_key and fecha_key in doc:
        extras.setdefault("Fecha_original", doc[fecha_key])

    # Provenance
    metadata["provenance"] = {
        "source": PROVENANCE_SOURCE,
        "ingested_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema_version": SCHEMA_VERSION
    }

    # Añade extras (aunque sea vacío, para esquema estable)
    metadata["extras"] = extras

    # Construye objeto final
    result = {
        "content": {
            # clave en minúsculas / normalizada
            "contenido": content_text
        },
        "metadata": metadata
    }
    return result


# =========================
# Muestreo y transformación
# =========================
def sample_noticias_structured(n=3) -> List[Dict[str, Any]]:
    client = _get_mongo_client()
    db = client[MONGO_DB]
    col = db[MONGO_COL]

    pipeline = [{"$sample": {"size": int(n)}}]
    docs = list(col.aggregate(pipeline))

    transformed = [_normalize_and_map(doc) for doc in docs]

    client.close()
    return transformed


if __name__ == "__main__":
    try:
        log.info(
            f"Conectando a MongoDB {MONGO_HOST}:{MONGO_PORT} "
            f"DB={MONGO_DB}, Colección={MONGO_COL}"
        )
        out = sample_noticias_structured(n=3)
        print(dumps(out, ensure_ascii=False, indent=2))
        log.info(f"Total documentos devueltos: {len(out)}")
    except Exception as e:
        log.error(f"Error al muestrear/transformar documentos: {e}")
        raise
