import os
import json
import logging
from typing import Optional, List, Dict, Any


# Solo le dipendenze per la logica di business.
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gcp-function")

# ==========================================
# CONFIGURAZIONE
# ==========================================
# In produzione su GCP, useresti l'IP interno di Milvus o Zilliz Cloud
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost") 
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_DEFAULT = 5

# ==========================================
# INIZIALIZZAZIONE GLOBALE (WARM START)
# ==========================================
logger.info("🚀 [GCP] Inizializzazione Ambiente...")

# 1. Caricamento Modello
try:
    logger.info(f"📦 Caricamento modello {EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("✅ Modello caricato in memoria globale")
except Exception as e:
    logger.error(f"❌ Errore critico modello: {e}")
    embedding_model = None

# 2. Connessione Milvus
MILVUS_READY = False
collection = None

try:
    logger.info(f"🔌 Connessione a Milvus {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        collection.load()
        MILVUS_READY = True
        logger.info(f"✅ Collection '{COLLECTION_NAME}' pronta")
    else:
        logger.warning(f"⚠️ Collection '{COLLECTION_NAME}' non trovata.")
except Exception as e:
    logger.warning(f"⚠️ Errore connessione Milvus: {e}")

# ==========================================
# LOGICA DI BUSINESS
# ==========================================
def search_products_logic(
    query: str,
    top_k: int = TOP_K_DEFAULT,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
    in_stock_only: bool = False,
    brand: Optional[str] = None
) -> Dict[str, Any]:
    
    if not MILVUS_READY or not embedding_model:
        return {
            "error": "Servizio non disponibile (DB o Modello offline)",
            "details": "Verifica la connessione a Milvus."
        }

    # Generazione Embedding
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).tolist()
    
    # Costruzione Filtri
    filter_parts = []
    if max_price is not None:
        filter_parts.append(f"price <= {max_price}")
    if category:
        filter_parts.append(f'category == "{category}"')
    if in_stock_only:
        filter_parts.append("in_stock == true")
    if brand:
        filter_parts.append(f'brand == "{brand}"')
    
    filter_expr = " && ".join(filter_parts) if filter_parts else ""
    
# Ricerca
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        expr=filter_expr,
        # RIMOSSO "relevance" da qui sotto
        output_fields=["product_id", "name", "category", "description", "price", "in_stock"]
    )

    products = []
    for hits in results:
        for hit in hits:
            # Calcolo score leggibile
            score_pct = f"{(1 / (1 + hit.distance)) * 100:.1f}%"
            products.append({
                "product_id": hit.entity.get("product_id"),
                "name": hit.entity.get("name"),
                "category": hit.entity.get("category"),
                "description": hit.entity.get("description"),
                "price": hit.entity.get("price"),
                "in_stock": hit.entity.get("in_stock"),
                "relevance": score_pct
            })
            
    return {
        "query": query,
        "total_results": len(products),
        "products": products,
        "filters_applied": {
            "max_price": max_price,
            "category": category
        }
    }

# ==========================================
# CLOUD FUNCTION ENTRY POINT
# ==========================================
def search_products_function(request):
    """
    HTTP Cloud Function Entry Point.
    Args:
        request (flask.Request): The request object.
    Returns:
        The response text, or a tuple of (response, status, headers).
    """
    
    # Gestione CORS (Pre-flight request)
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Parsing Request
    request_json = request.get_json(silent=True)
    
    if not request_json or 'query' not in request_json:
        return (json.dumps({"error": "Parametro 'query' mancante nel JSON body"}), 400, {'Content-Type': 'application/json'})

    try:
        # Esecuzione logica
        result = search_products_logic(
            query=request_json['query'],
            top_k=request_json.get('top_k', 5),
            max_price=request_json.get('max_price'),
            category=request_json.get('category'),
            in_stock_only=request_json.get('in_stock_only', False),
            brand=request_json.get('brand')
        )
        
        # Risposta Successo
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        }
        return (json.dumps(result), 200, headers)
        
    except Exception as e:
        logger.error(f"Runtime Error: {e}")
        return (json.dumps({"error": str(e)}), 500, {'Content-Type': 'application/json'})