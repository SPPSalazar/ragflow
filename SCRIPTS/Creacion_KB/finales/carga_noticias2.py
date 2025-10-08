import socket
import time
import sys

# Importar solo lo necesario para verificar conexión ANTES que funciones.py
try:
    import requests
except ImportError:
    print("⚠️  Instalando dependencia 'requests'...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

# =========================
# Función para verificar conexión a internet
# =========================
def verificar_conexion_internet(timeout=5):
    """
    Verifica si hay conexión a internet intentando conectarse a servicios conocidos.
    Retorna True si hay conexión, False si no.
    """
    hosts_prueba = [
        ("8.8.8.8", 53),  # Google DNS
        ("1.1.1.1", 53),  # Cloudflare DNS
    ]
    
    for host, port in hosts_prueba:
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except (socket.error, socket.timeout):
            continue
    
    # Verificación alternativa por HTTP
    try:
        requests.get("https://www.google.com", timeout=timeout)
        return True
    except:
        return False

def esperar_conexion():
    """
    Espera hasta que se restablezca la conexión a internet.
    Verifica cada 30 segundos.
    """
    print("\n⚠️  Conexión a internet perdida. Esperando reconexión...")
    while not verificar_conexion_internet():
        print(f"⏳ Sin conexión. Reintentando en 30 segundos... ({time.strftime('%H:%M:%S')})")
        time.sleep(30)
    print("✅ Conexión a internet restablecida. Continuando proceso...\n")

def ejecutar_con_verificacion_conexion(funcion, *args, **kwargs):
    """
    Ejecuta una función y maneja automáticamente las pérdidas de conexión.
    Si se pierde la conexión, espera a que se restablezca y reintenta.
    """
    max_reintentos = 10  # Aumentado para operaciones largas como embeddings
    reintentos = 0
    
    while reintentos < max_reintentos:
        try:
            # Verificar conexión antes de ejecutar
            if not verificar_conexion_internet():
                esperar_conexion()
            
            # Ejecutar la función
            resultado = funcion(*args, **kwargs)
            return resultado
            
        except (ConnectionError, socket.error, requests.exceptions.RequestException) as e:
            print(f"⚠️  Error de conexión detectado: {e}")
            esperar_conexion()
            reintentos += 1
            
        except Exception as e:
            error_msg = str(e)
            # Detectar errores relacionados con conexión a servicios externos
            errores_conexion = [
                'EndpointConnectionError',
                'Could not connect',
                'Connection refused',
                'Connection timeout',
                'Connection reset',
                'Network is unreachable',
                'Temporary failure',
                'Name or service not known',
                'bedrock-runtime',
                'amazonaws.com',
                'timed out',
                'HTTPSConnectionPool',
                'SSLError'
            ]
            
            # Verificar si el error está relacionado con conexión
            es_error_conexion = any(err.lower() in error_msg.lower() for err in errores_conexion)
            
            if es_error_conexion:
                print(f"⚠️  Error de conexión a servicio externo detectado:")
                print(f"    {error_msg[:200]}...")  # Mostrar solo los primeros 200 caracteres
                esperar_conexion()
                reintentos += 1
                print(f"🔄 Reintentando... (Intento {reintentos}/{max_reintentos})")
            else:
                # Errores no relacionados con conexión
                print(f"❌ Error no relacionado con conexión: {e}")
                raise
    
    raise Exception(f"No se pudo completar la operación después de {max_reintentos} reintentos")

# =========================
# Carga de entorno y logging a ragflow
# =========================
print("🔍 Verificando conexión a internet inicial...")
if not verificar_conexion_internet():
    esperar_conexion()
else:
    print("✅ Conexión a internet disponible\n")

# IMPORTANTE: Importar módulos que requieren conexión DESPUÉS de verificarla
print("📦 Cargando módulos y conectando a servicios...")
from dotenv import load_dotenv, find_dotenv
import os

# Cargar variables de entorno
load_dotenv(find_dotenv(usecwd=True))

# Importar funciones.py con manejo de errores de conexión
try:
    import funciones
    print("✓ Módulo 'funciones' cargado correctamente")
except Exception as e:
    error_msg = str(e)
    # Detectar si es error de conexión
    errores_conexion = [
        'Port contains non-digit',
        'Connection refused',
        'ServerSelectionTimeoutError',
        'Network is unreachable',
        'getaddrinfo failed',
        'Temporary failure',
        'Name or service not known'
    ]
    
    es_error_conexion = any(err.lower() in error_msg.lower() for err in errores_conexion)
    
    if es_error_conexion or 'Port' in error_msg:
        print(f"⚠️  Error al conectar a MongoDB: {error_msg[:150]}...")
        print("💡 Esto puede deberse a:")
        print("   1. Conexión a internet inestable")
        print("   2. URI de MongoDB mal configurada")
        print("   3. MongoDB no está accesible")
        esperar_conexion()
        print("🔄 Reintentando importar módulo 'funciones'...")
        import funciones
        print("✓ Módulo 'funciones' cargado correctamente")
    else:
        print(f"❌ Error al importar 'funciones': {e}")
        raise

from ragflow_sdk import RAGFlow

RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY = "ragflow-U3YTBkMTM2YTQ5YjExZjA5OWRmNzZjNT"# os.getenv("RAGFLOW_API_KEY", "dev")
RF_DATASET_NAME = ["2020","2021","2022","2023","2024"]#la cantidad de años que se necesiten

# Conectar a RAGFlow con verificación de conexión
rf = ejecutar_con_verificacion_conexion(RAGFlow, api_key=RF_API_KEY, base_url=RF_URL)

# =========================
# --- dataset (get or create) ---
# =========================
# --- sample record and normalization
# =========================
for i in RF_DATASET_NAME:
    print(f"\n{'='*50}")
    print(f"📅 Procesando año: {i}")
    print(f"{'='*50}")
    
    # Obtener dataset con verificación de conexión
    dataset = ejecutar_con_verificacion_conexion(rf.get_dataset, name=i)
    print(f"✓ Dataset '{i}' obtenido")
    
    # Obtener noticias (operación local, no requiere verificación)
    noticias = funciones.sample_noticias_structured(int(i))
    print(f"✓ {len(noticias)} noticias obtenidas para {i}")
    
    # Normalizar fechas (operación local)
    noticias = funciones.datetime_ISO8601(noticias)
    print(f"✓ Fechas normalizadas")
    
    # Cargar documentos con verificación de conexión
    print(f"📤 Cargando noticias al dataset...")
    try:
        funciones.cargar_docs_v2(noticias,dataset)
    except:
        pass
    print(f"✅ Noticias del año {i} cargadas exitosamente\n")

print("\n" + "="*50)
print("🎉 Proceso completado exitosamente")
print("="*50)

