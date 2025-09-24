from ragflow_sdk import RAGFlow

# Conexión
rf = RAGFlow(api_key="ragflow-MwYTg3MTYwOTdlNDExZjBhMjUyZDY4OT", base_url="http://127.0.0.1:9380")

datasets = rf.list_datasets()
for ds in datasets:
    print(f"\n📚 Dataset: {ds.name} | ID: {ds.id}")

    documents = ds.list_documents()
    for doc in documents:
        print(f"  📄 Documento: {doc.name} | ID: {doc.id}")

        chunks = doc.list_chunks()
        for chunk in chunks:
            print(f"     🔹 Chunk ID: {chunk.id}")
            print("        Texto:", chunk.content[:80], "..." if len(chunk.content) > 80 else "")
            
            # Acceder al embedding
            embedding = getattr(chunk, "embedding", None)
            if embedding:
                print("        🔢 Embedding (dimensiones):", len(embedding))
                print("        🔸 Vector:", embedding[:5], "...")  # Muestra primeros 5 valores
            else:
                print("        ⚠️ Embedding no disponible")
                print("####################################################################")
