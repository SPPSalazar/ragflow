import socket
import time
import sys
import funciones

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
                'SSLError',
                'ReadTimeoutError',
                'Read timeout',
                'endpoint URL'
            ]
            
            # Verificar si el error está relacionado con conexión
            es_error_conexion = any(err.lower() in error_msg.lower() for err in errores_conexion)
            
            if es_error_conexion:
                print(f"⚠️  Error de conexión/timeout detectado:")
                print(f"    {error_msg[:200]}...")  # Mostrar solo los primeros 200 caracteres
                esperar_conexion()
                reintentos += 1
                print(f"🔄 Reintentando... (Intento {reintentos}/{max_reintentos})")
                time.sleep(5)  # Esperar 5 segundos adicionales antes de reintentar
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
        print("✓ Módulo 'funciones' cargado correctamente")
    else:
        print(f"❌ Error al importar 'funciones': {e}")
        raise

from ragflow_sdk import RAGFlow

RF_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380")
RF_API_KEY = "ragflow-ZlMmJiYWE4OWQ1MTExZjBhZTkxYzJiMT"
RF_DATASET_NAME = ["Last3months", "2025"]

# Validar configuración
print(f"\n🔧 Configuración RAGFlow:")
print(f"   URL: {RF_URL}")
print(f"   API Key: {RF_API_KEY[:20]}...")
print(f"   Datasets: {', '.join(RF_DATASET_NAME)}")

# Conectar a RAGFlow con verificación de conexión
print(f"\n🔌 Conectando a RAGFlow...")
rf = ejecutar_con_verificacion_conexion(RAGFlow, api_key=RF_API_KEY, base_url=RF_URL)
print(f"✅ Conectado a RAGFlow exitosamente\n")

# =========================
# Procesamiento de datasets
# =========================
# Estadísticas globales
estadisticas_globales = {
    "total_noticias_procesadas": 0,
    "total_noticias_cargadas": 0,
    "total_noticias_descartadas": 0,
    "errores_por_anio": {}
}

for i in RF_DATASET_NAME:
    print(f"\n{'='*70}")
    print(f"📅 Procesando año: {i}")
    print(f"{'='*70}")
    

    # Obtener dataset con verificación de conexión
    dataset = ejecutar_con_verificacion_conexion(rf.get_dataset, name=i)
    print(f"✓ Dataset '{i}' obtenido")
    
    # Obtener noticias (operación local, no requiere verificación)
    if i != "2025":
        noticias = funciones.sample_last3months()
    else:
        noticias = funciones.sample_noticias_3months_complemento()
    print(f"✓ {len(noticias)} noticias obtenidas para {i}")
    estadisticas_globales["total_noticias_procesadas"] += len(noticias)
    
    if not noticias:
        print(f"⚠️  No hay noticias para cargar en {i}")
        estadisticas_globales["errores_por_anio"][i] = "Sin noticias obtenidas"
        continue
    
    # Normalizar fechas (operación local)
    noticias = funciones.datetime_ISO8601(noticias)
    print(f"✓ Fechas normalizadas")
    
    # USAR LA FUNCIÓN SEGURA DE CARGA CON VALIDACIÓN
    print(f"📤 Cargando noticias al dataset con validación y reintentos...")
    funciones.cargar_docs_v2(noticias,dataset)
    print("##################Eliminando noticias con más de 3 meses de antigüedad##########################")
    dataset = rf.get_dataset(name="Last3months")
    c = funciones.delete_docs_older_than_3months(dataset=dataset)

    

# =========================
# Resumen final
# =========================
print("\n" + "="*70)
print("📈 RESUMEN FINAL DEL PROCESO")
print("="*70)
print(f"📊 Noticias procesadas: {estadisticas_globales['total_noticias_procesadas']}")
print(f"✅ Noticias cargadas exitosamente: {estadisticas_globales['total_noticias_cargadas']}")
print(f"⚠️  Noticias descartadas: {estadisticas_globales['total_noticias_descartadas']}")

if estadisticas_globales["errores_por_anio"]:
    print(f"\n⚠️  Errores por año:")
    for anio, error in estadisticas_globales["errores_por_anio"].items():
        print(f"   - {anio}: {error}")

if estadisticas_globales['total_noticias_cargadas'] > 0:
    porcentaje_exito = (estadisticas_globales['total_noticias_cargadas'] / 
                        estadisticas_globales['total_noticias_procesadas'] * 100)
    print(f"\n📈 Tasa de éxito: {porcentaje_exito:.2f}%")
    
    tiempo_estimado = estadisticas_globales['total_noticias_cargadas'] * 0.5  # 0.5 seg por noticia
    print(f"⏱️  Tiempo estimado invertido: {tiempo_estimado/60:.1f} minutos")
else:
    print(f"\n❌ No se cargaron noticias. Problemas detectados:")
    print(f"   1. El campo 'contenido' está vacío en las noticias")
    print(f"   2. Problemas de timeout/conexión con RAGFlow")
    print(f"   3. Verifica que RAGFlow esté funcionando correctamente")

print("="*70)
print("🎉 Proceso completado")
print("="*70)

