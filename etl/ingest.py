"""
ingest.py - ETL Pipeline for Product Discovery Agent

This script:
1. Reads products from products.json
2. Generates embeddings for product descriptions
3. Creates a Milvus collection with proper schema
4. Inserts all products with embeddings into the vector database
"""

import json
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "products"
EMBEDDING_DIM = 384  # Dimensione per all-MiniLM-L6-v2
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PRODUCTS_FILE = "products.json"


def connect_to_milvus():
    """Connect to Milvus server"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        logger.info(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Milvus: {e}")
        raise


def create_collection():
    """Create Milvus collection with proper schema"""

    # Drop existing collection if exists
    if utility.has_collection(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' already exists. Dropping it...")
        utility.drop_collection(COLLECTION_NAME)

    # Define schema
    fields = [
        FieldSchema(name="product_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="price", dtype=DataType.FLOAT),
        FieldSchema(name="in_stock", dtype=DataType.BOOL),
        FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=100)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Product catalog with embeddings for semantic search"
    )

    # Create collection
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    logger.info(f"✅ Collection '{COLLECTION_NAME}' created successfully")

    # Create index on vector field for efficient search
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    logger.info("✅ Index created on embedding field")

    return collection


def load_products(filename: str) -> List[Dict]:
    """Load products from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            products = json.load(f)
        logger.info(f"✅ Loaded {len(products)} products from {filename}")
        return products
    except FileNotFoundError:
        logger.error(f"❌ File {filename} not found!")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in {filename}: {e}")
        raise


def generate_embeddings(texts: List[str], model: SentenceTransformer) -> List[List[float]]:
    """Generate embeddings for a list of texts"""
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def insert_products(collection: Collection, products: List[Dict], embeddings: List[List[float]]):
    """Insert products with embeddings into Milvus collection"""

    # Prepare data for insertion
    data = [
        [p["product_id"] for p in products],  # product_id
        embeddings,                            # embedding
        [p["name"] for p in products],        # name
        [p["description"] for p in products], # description
        [p["category"] for p in products],    # category
        [p["price"] for p in products],       # price
        [p["in_stock"] for p in products],    # in_stock
        [p["brand"] for p in products]        # brand
    ]

    # Insert data
    mr = collection.insert(data)
    collection.flush()

    logger.info(f"✅ Inserted {len(products)} products into collection")
    logger.info(f"   Insert count: {mr.insert_count}")


def main():
    """Main ETL pipeline"""
    logger.info("🚀 Starting ETL pipeline...")

    # Step 1: Connect to Milvus
    logger.info("\n📡 Step 1: Connecting to Milvus...")
    connect_to_milvus()

    # Step 2: Create collection
    logger.info("\n📊 Step 2: Creating collection...")
    collection = create_collection()

    # Step 3: Load products
    logger.info(f"\n📂 Step 3: Loading products from {PRODUCTS_FILE}...")
    products = load_products(PRODUCTS_FILE)

    # Step 4: Load embedding model
    logger.info(f"\n🤖 Step 4: Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"✅ Model loaded (embedding dimension: {EMBEDDING_DIM})")

    # Step 5: Generate embeddings
    logger.info("\n🔢 Step 5: Generating embeddings for product descriptions...")
    descriptions = [p["description"] for p in products]
    embeddings = generate_embeddings(descriptions, model)
    logger.info(f"✅ Generated {len(embeddings)} embeddings")

    # Step 6: Insert into database
    logger.info("\n💾 Step 6: Inserting products into Milvus...")
    insert_products(collection, products, embeddings)

    # Step 7: Load collection into memory
    logger.info("\n⚡ Step 7: Loading collection into memory...")
    collection.load()
    logger.info("✅ Collection loaded and ready for queries")

    # Verify insertion
    logger.info(f"\n✅ ETL Pipeline completed successfully!")
    logger.info(f"   Total products in collection: {collection.num_entities}")

    # Disconnect
    connections.disconnect("default")
    logger.info("👋 Disconnected from Milvus")


if __name__ == "__main__":
    main()