"""
Server MCP Completo
- Gestisce ricerca vettoriale con Milvus + SentenceTransformers
- Supporta protocollo MCP JSON-RPC 2.0 (Initialize, List Tools, Call Tool)
- Include endpoint diretti per testing rapido
"""
import os
import json
import logging
from typing import Optional, List, Dict, Any

# Frameworks
from fastapi import FastAPI, Request, HTTPException
import uvicorn

# AI & DB
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server")

# ==========================================
# CONFIGURAZIONE
# ==========================================
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_DEFAULT = 5

# ==========================================
# INIZIALIZZAZIONE GLOBALE
# ==========================================
logger.info("🚀 Inizializzazione Risorse Server MCP...")

# 1. Caricamento Modello
try:
    logger.info(f"📦 Caricamento modello {EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("✅ Modello caricato")
except Exception as e:
    logger.error(f"❌ Errore caricamento modello: {e}")
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
    logger.warning(f"⚠️ Milvus non disponibile: {e}")

# ==========================================
# LOGICA DI BUSINESS (Completa)
# ==========================================

def generate_embedding(text: str) -> List[float]:
    if not embedding_model:
        raise Exception("Modello embedding non caricato")
    return embedding_model.encode(text, convert_to_tensor=False).tolist()

def search_products_logic(
    query: str,
    top_k: int = TOP_K_DEFAULT,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
    in_stock_only: bool = False,
    brand: Optional[str] = None
) -> Dict[str, Any]:
    """Logica di ricerca con supporto a tutti i filtri"""
    
    if not MILVUS_READY:
        return {"error": "Milvus non connesso", "query_received": query}

    logger.info(f"🔍 Ricerca: '{query}' (Filtri: price<{max_price}, cat={category}, stock={in_stock_only})")
    
    query_embedding = generate_embedding(query)
    
    # Costruzione filtri dinamici
    filter_parts = []
    if max_price is not None: filter_parts.append(f"price <= {max_price}")
    if category: filter_parts.append(f'category == "{category}"')
    if in_stock_only: filter_parts.append("in_stock == true")
    if brand: filter_parts.append(f'brand == "{brand}"')
    
    filter_expr = " && ".join(filter_parts) if filter_parts else ""
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        expr=filter_expr,
        output_fields=["product_id", "name", "category", "description", "price", "in_stock", "brand"]
    )

    products = []
    for hits in results:
        for hit in hits:
            products.append({
                "product_id": hit.entity.get("product_id"),
                "name": hit.entity.get("name"),
                "category": hit.entity.get("category"),
                "description": hit.entity.get("description"),
                "price": hit.entity.get("price"),
                "in_stock": hit.entity.get("in_stock"),
                "brand": hit.entity.get("brand"),
                "relevance": f"{(1 / (1 + hit.distance)) * 100:.1f}%"
            })
            
    return {
        "query": query,
        "total_results": len(products),
        "products": products
    }

# ==========================================
# FASTAPI APP & ENDPOINTS
# ==========================================
app = FastAPI(title="Ecommerce MCP Server")

@app.get("/")
async def root():
    return {
        "name": "ecommerce-mcp-server",
        "status": "online",
        "protocol": "mcp-http",
        "milvus_connected": MILVUS_READY
    }

# ==========================================
# PROTOCOLLO MCP (JSON-RPC 2.0)
# ==========================================
@app.post("/mcp")
async def handle_mcp(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    jsonrpc_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params", {})

    logger.info(f"📨 MCP Method: {method}")

    # 1. Initialize
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": jsonrpc_id, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "ecommerce-mcp", "version": "1.0.0"}
        }}
    
    # 2. Notifications
    elif method == "notifications/initialized":
        return {"jsonrpc": "2.0", "id": jsonrpc_id, "result": True}

    # 3. List Tools (Espone tutti i filtri all'Agente)
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": jsonrpc_id, "result": {
            "tools": [{
                "name": "search_products",
                "description": "Semantic search for e-commerce products. Understands natural language queries (e.g., 'waterproof running shoes under 100 euros') and returns ranked results with relevance scores. Supports optional filtering by price, category, brand, and stock availability. Use this tool when customers want to find, search, browse, or get recommendations for products.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of the desired product"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results to return (1-20)",
                            "default": 5
                        },
                        "max_price": {
                            "type": "number",
                            "description": "Maximum price in EUR (optional filter)"
                        },
                        "category": {
                            "type": "string",
                            "description": "Product category to filter by",
                            "enum": ["Footwear", "Clothing", "Electronics", "Accessories", "Outdoor"]
                        },
                        "in_stock_only": {
                            "type": "boolean",
                            "description": "If true, return only products currently in stock",
                            "default": False
                        },
                        "brand": {
                            "type": "string",
                            "description": "Brand name to filter by (e.g., 'ActiveGear')"
                        }
                    },
                    "required": ["query"]
                }
            }]
        }}

    # 4. Call Tool
    elif method == "tools/call":
        args = params.get("arguments", {})
        try:
            result = search_products_logic(**args)
            return {"jsonrpc": "2.0", "id": jsonrpc_id, "result": {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
            }}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": jsonrpc_id, "error": {"code": -32000, "message": str(e)}}

    # 5. Ping
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": jsonrpc_id, "result": {}}

    return {"jsonrpc": "2.0", "id": jsonrpc_id, "error": {"code": -32601, "message": "Method not found"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)