from ragflow_sdk import RAGFlow

# Conecta a RAGFlow
rf = RAGFlow(api_key="ragflow-MwYTg3MTYwOTdlNDExZjBhMjUyZDY4OT", base_url="http://127.0.0.1:9380")

# Lista datasets
datasets = rf.list_datasets()
for ds in datasets:
    print("\n📚 Dataset:", ds.name, "| ID:", ds.id)

    # Lista documentos del dataset
    documents = ds.list_documents()
    
#    for doc in documents:
#        print(vars(doc)) 
    for doc in documents:
        print("  📄 Documento:", doc.name, "| ID:", doc.id)

        # Lista chunks (embeddings) del documento
        chunks = doc.list_chunks()

        for chunk in chunks:
            print("     🔹 Chunk ID:", chunk.id)
            print("        Texto:", chunk.content[:80], "..." if len(chunk.content) > 80 else "")


