
# rf_search_with_filter.py
import os, json, requests

# inspect_ragflow_files_only.py
import os, json, requests

BASE   = os.getenv("RAGFLOW_RAGFLOW_BASE_URL_URL", "http://localhost:9380")
RAGFLOW_API_KEY       = os.getenv("RAGFLOW_API_KEY", "ragflow-Q1MWQ0ZjUyOTQxNzExZjBhNmQ0ZjI2Mj")
DSID   = os.getenv("RAGFLOW_DATASET_ID", "b")
RAGFLOW_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "noticias_kb")


def search(dataset_id: str, query: str, categoria: str | None = None, top_k: int = 5):
    body_variants = [
        # Variante 1 (la más común)
        {"query": query, "top_k": top_k, "filters": {"metadata.categoria": categoria}} if categoria else {"query": query, "top_k": top_k},
        # Variante 2 (algunas builds)
        {"dataset_id": dataset_id, "query": query, "top_k": top_k,
         "filters": {"metadata.categoria": categoria} if categoria else {}},
    ]
    urls = [f"{BASE}/api/v1/datasets/{dataset_id}/search", f"{BASE}/api/v1/search"]
    for url in urls:
        for body in body_variants:
            try:
                r = requests.post(url, headers=H(), data=json.dumps(body), timeout=60)
                if r.status_code != 200: continue
                jr = r.json()
                if isinstance(jr.get("data"), list) and jr["data"]:
                    # Muestra texto y metadata del primer hit
                    hit = jr["data"][0]
                    print(json.dumps({
                        "used_url": url,
                        "body": body,
                        "first_hit_text": hit.get("text") or hit.get("content"),
                        "first_hit_metadata": hit.get("metadata")
                    }, ensure_ascii=False, indent=2))
                    return
            except Exception:
                pass
    print("No hubo resultados o la API de búsqueda de tu build usa otra ruta/shape.")

if __name__ == "__main__":
    # Cambia 'salud' y 'politica' por algo que exista en tus datos
    search(DSID, query="familia", categoria="asesinato")
