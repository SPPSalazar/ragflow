import funciones
INF_HOST = "http://localhost:23820"
DB_NAME = "default_db"
TABLE_NAME = "ragflow_706d8ba6995f11f0ae9f4ab243a5b9e6_7b358bd8995f11f09f1c4ab243a5b9e6"
TIMEOUT = 15

funciones.get_columns(INF_HOST, DB_NAME,TABLE_NAME,TIMEOUT)

#if __name__ == "__main__":
    # 1. Mostrar columnas
    #funciones.get_columns(INF_HOST, DB_NAME,TABLE_NAME)
    # 2. Traer 1 fila
    #rows = get_rows(limit=1)
    #print("\nPrimera fila:")
    #if rows:
    #    print(rows[0])
    #else:
    #    print("(sin resultados)")
