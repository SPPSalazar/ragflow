import json
import requests

url = "http://localhost:23820/databases/default_db/tables/ragflow_706d8ba6995f11f0ae9f4ab243a5b9e6_7b358bd8995f11f09f1c4ab243a5b9e6/docs"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
}

# Mismo payload que en curl
payload = {
    "output": ["id","doc_id", "kb_id", "docnm_kwd", "q_1024_vec"],
    "limit": "1",
    "offset": "0"
}

# GET con cuerpo (no estándar, pero requests lo permite con data=)
resp = requests.get(url, headers=headers, data=json.dumps(payload), timeout=30)
resp.raise_for_status()  # lanza excepción si el status no es 2xx

data = resp.json()
print("Respuesta completa:", json.dumps(data, ensure_ascii=False, indent=2))

# Si el API devuelve filas en algún campo típico (ajusta según tu respuesta real):
# Ejemplo: si viene como {"rows":[{"doc_id":"..."} , ...]}
#rows = data.get("rows") or data.get("data") or data.get("result") or []
#doc_ids = [r.get("doc_id") for r in rows if isinstance(r, dict)]
#print("doc_ids:", doc_ids)



