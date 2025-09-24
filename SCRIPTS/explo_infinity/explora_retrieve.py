from ragflow_sdk import RAGFlow

rf = RAGFlow(api_key="ragflow-MwYTg3MTYwOTdlNDExZjBhMjUyZDY4OT", base_url="http://127.0.0.1:9380")

datasets = rf.list_datasets()
for ds in datasets:
    print("Dataset:", ds.name, ds,id)

