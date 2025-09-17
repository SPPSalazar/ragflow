# sample_random_noticias.py
import os
import logging
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote_plus
from pymongo import MongoClient
from bson.json_util import dumps

# =========================
# Carga de entorno y logging
# =========================
load_dotenv(find_dotenv(usecwd=True))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("mongo_sample")

# Si tu .env está en el directorio padre (como en tu ejemplo):
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dotenv_path = os.path.join(parent_dir, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# =========================
# Parámetros MongoDB por .env (con defaults sensatos)
# =========================
MONGO_HOST = os.getenv("MONGO_PRO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PRO_PORT", "27017")
MONGO_USER = os.getenv("MONGO_PRO_USER", "")
MONGO_PASS = os.getenv("MONGO_PRO_PASSWORD", "")
MONGO_DB   = os.getenv("MONGO_PRO_DB", "noticias_db")
MONGO_COL  = os.getenv("MONGO_PRO_COLLECTION", "Noticias")

def get_mongo_client():
    """Construye el URI y devuelve un cliente Mongo."""
    if MONGO_USER and MONGO_PASS:
        uri = (
            f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}"
            f"@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource=admin"
        )
    else:
        uri = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    # Verificar conexión
    client.admin.command("ping")
    return client

def sample_random_noticias(n=3, projection=None):
    """
    Retorna n documentos aleatorios de noticias_db.Noticias usando $sample.
    projection: dict de proyección opcional, p.ej. {"Titulo": 1, "Contenido": 1}
    """
    client = get_mongo_client()
    db = client[MONGO_DB]
    col = db[MONGO_COL]

    pipeline = [{"$sample": {"size": int(n)}}]
    if projection:
        pipeline.append({"$project": projection})

    docs = list(col.aggregate(pipeline))
    client.close()
    return docs

if __name__ == "__main__":
    try:
        log.info(
            f"Conectando a MongoDB {MONGO_HOST}:{MONGO_PORT} "
            f"DB={MONGO_DB}, Colección={MONGO_COL}"
        )
        documentos = sample_random_noticias(
            n=3,
            # Opcional: limita campos devueltos (1=incluir, 0=excluir)
            # projection={"Titulo": 1, "Contenido": 1, "Fecha": 1, "_id": 0}
        )

        # Imprime en JSON legible (maneja ObjectId/fechas)
        print(dumps(documentos, ensure_ascii=False, indent=2))
        log.info(f"Total documentos devueltos: {len(documentos)}")

    except Exception as e:
        log.error(f"Error al muestrear documentos: {e}")
        raise
