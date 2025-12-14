# 🛍️ Intelligent Product Discovery Agent

An AI-powered e-commerce search system that uses **semantic vector search** combined with **scalar filtering** to understand natural language queries and find relevant products. Built with **Google ADK**, **Milvus vector database**, and **MCP (Model Context Protocol)**.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [Features](#-features)
4. [Prerequisites](#-prerequisites)
5. [Quick Start](#-quick-start)
6. [Database Schema](#-database-schema)
7. [MCP Tool Interface](#-mcp-tool-interface)
8. [Step-by-Step Setup](#-step-by-step-setup)
9. [Testing the Agent](#-testing-the-agent)
10. [Example Interactions](#-example-interactions)
11. [GCP Simulation](#-gcp-simulation)
12. [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

Traditional keyword-based search fails when customers use natural language like *"I need running shoes good for rainy days under 100 euros"*. This project solves that problem by:

- **Understanding Intent**: Uses sentence embeddings to capture semantic meaning
- **Smart Filtering**: Combines vector similarity with price, category, brand, and stock filters
- **Natural Interaction**: Accepts conversational queries through an AI agent powered by Gemini
- **Scalable Architecture**: Built with production-ready components (Milvus, FastAPI, MCP)

### Key Technologies

- **Google ADK (Agent Development Kit)**: Framework for building AI agents
- **Milvus**: Open-source vector database for semantic search
- **SentenceTransformers**: Generates embeddings from product descriptions
- **FastAPI**: High-performance HTTP server for the MCP interface
- **MCP (Model Context Protocol)**: Standardized protocol for AI-tool communication

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER QUERY                              │
│          "Find waterproof running shoes under 90 euros"          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI AGENT (Google ADK)                        │
│  • Powered by Gemini 2.5 Flash                                   │
│  • Extracts semantic intent: "waterproof running shoes"          │
│  • Extracts constraints: max_price = 90                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ MCP Protocol (JSON-RPC 2.0)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MCP SERVER (FastAPI)                           │
│  Endpoint: http://localhost:8002/mcp                             │
│  • Receives: search_products(query, filters...)                  │
│  • Generates: 384-dim embedding via SentenceTransformer          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MILVUS VECTOR DATABASE                         │
│  • Collection: "products"                                        │
│  • Performs: L2 similarity search on embeddings                  │
│  • Applies: Scalar filters (price <= 90, category, stock)        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RESULTS                                   │
│  [                                                               │
│    {                                                             │
│      "name": "StormRunner X5",                                   │
│      "price": 89.99,                                             │
│      "relevance": "92.3%",                                       │
│      "description": "Durable trail shoe with GORE-TEX..."        │
│    }                                                             │
│  ]                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Core Capabilities

- ✅ **Semantic Search**: Understands "waterproof running shoes" without exact keyword matches
- ✅ **Advanced Filtering**: Price range, category, brand, stock availability
- ✅ **Natural Language Processing**: Interprets conversational queries

### Technical Features

- ✅ **Vector Database**: Milvus with IVF_FLAT indexing for efficient similarity search
- ✅ **MCP Protocol**: JSON-RPC 2.0 compliant tool interface
- ✅ **Docker Support**: Containerized Milvus deployment
- ✅ **GCP Ready**: Cloud Functions simulation included

---

## 📦 Prerequisites

### Required Software

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop/))
- **Git** (for cloning the repository)

### Python Dependencies

```bash
# Install all project dependencies
pip install -r requirements.txt
```

The project includes two requirements files:
- **`requirements.txt`** - Full project dependencies (ADK, FastAPI, Milvus, etc.)
- **`gcp_simulation/requirements.txt`** - Minimal dependencies for Cloud Functions only

### API Keys

- **Google AI Studio API Key**: Free at [aistudio.google.com](https://aistudio.google.com/)
    - Click "Get API Key" → Generate → Copy to `.env` file

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone <your-repo-url>
cd repo-folder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Milvus database
docker-compose up -d

# 4. Ingest sample data
python ingest.py

# 5. Start MCP server
python mcp_server_http.py

# 6. (New terminal) Start ADK web interface
adk web

# 7. Open browser at http://127.0.0.1:8000
```

---

## 🗄️ Database Schema

### Milvus Collection: `products`

| Field Name    | Data Type     | Description                                     | Indexed |
|---------------|---------------|-------------------------------------------------|---------|
| `product_id`  | INT64         | Primary key (unique identifier)                 | ✅       |
| `embedding`   | FLOAT_VECTOR  | 384-dimensional semantic vector                 | ✅       |
| `name`        | VARCHAR(256)  | Product name                                    | ❌       |
| `description` | VARCHAR(2000) | Full product description (used for embedding)   | ❌       |
| `category`    | VARCHAR(100)  | Product category (e.g., "Footwear", "Clothing") | ❌       |
| `price`       | FLOAT         | Price in EUR                                    | ❌       |
| `in_stock`    | BOOL          | Availability status                             | ❌       |
| `brand`       | VARCHAR(100)  | Brand name (e.g., "ActiveGear", "Nike")         | ❌       |

### Index Configuration

```python
index_params = {
    "metric_type": "L2",        # Euclidean distance
    "index_type": "IVF_FLAT",   # Inverted File with Flat quantization
    "params": {"nlist": 128}    # Number of cluster units
}
```

**Why L2 Distance?**
- L2 (Euclidean distance) measures the straight-line distance between embedding vectors
- Lower L2 distance = Higher similarity
- Converted to relevance percentage: `(1 / (1 + distance)) * 100`

### Sample Product Data

```json
{
  "product_id": 101,
  "name": "StormRunner X5",
  "category": "Footwear",
  "description": "A durable trail running shoe designed for wet conditions. Features GORE-TEX lining and high-traction soles.",
  "price": 89.99,
  "in_stock": true,
  "brand": "ActiveGear"
}
```

The dataset includes **20 diverse products** across 5 categories: Footwear, Clothing, Electronics, Accessories, and Outdoor.

---

## 🔧 MCP Tool Interface

### Protocol: JSON-RPC 2.0

The MCP server exposes tools following the Model Context Protocol standard, enabling seamless communication between AI agents and external services.

### Available Tool: `search_products`

#### Tool Definition (JSON Schema)

```json
{
  "name": "search_products",
  "description": "Semantic search for e-commerce products using natural language queries. Supports filters for price, category, brand, and stock availability.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language query describing the desired product"
      },
      "top_k": {
        "type": "integer",
        "description": "Maximum number of results to return",
        "default": 5
      },
      "max_price": {
        "type": "number",
        "description": "Maximum price in EUR (optional filter)"
      },
      "category": {
        "type": "string",
        "description": "Filter by category"
      },
      "in_stock_only": {
        "type": "boolean",
        "description": "Return only products currently in stock",
        "default": false
      },
      "brand": {
        "type": "string",
        "description": "Filter by brand name"
      }
    },
    "required": ["query"]
  }
}
```

#### Example Tool Call (from ADK Agent)

```json
{
  "jsonrpc": "2.0",
  "id": "adk-8b13a931-60bd-4237-8c43-94f646075c76",
  "method": "tools/call",
  "params": {
    "name": "search_products",
    "arguments": {
      "query": "scarpe",
      "max_price": 50
    }
  }
}
```

#### Example Tool Response

```json
{
  "jsonrpc": "2.0",
  "id": "adk-8b13a931-60bd-4237-8c43-94f646075c76",
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"query\":\"scarpe\",\"total_results\":5,\"products\":[{\"product_id\":103,\"name\":\"UrbanWalker Sneakers\",\"price\":45.0,\"relevance\":\"92.3%\"}]}"
    }]
  }
}
```

### MCP Endpoints

| Endpoint | Method | Description                    |
|----------|--------|--------------------------------|
| `/mcp`   | POST   | Main MCP JSON-RPC 2.0 endpoint |
| `/`      | GET    | Server info and status         |

---

## 📚 Step-by-Step Setup

### Step 1: Environment Setup

```bash
# Clone and open project root directory

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

Create a `.env` file in the p_discovery_agent directory:

```properties
# Google API Configuration
GOOGLE_API_KEY=your_actual_api_key_here

# Milvus Configuration (optional, defaults shown)
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

**Get Google API Key:**
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key" (top right)
3. Generate key and copy into `.env` file

### Step 3: Start Milvus Database

```bash
# Ensure Docker Desktop is running

# Start Milvus services
docker-compose up -d

# Verify containers are running
docker ps
```

### Step 4: Ingest Product Data

```bash
# Run ETL pipeline
cd etl
python ingest.py
```

**Expected Output:**
```
🚀 Starting ETL pipeline...
✅ Connected to Milvus at localhost:19530
✅ Collection 'products' created successfully
✅ Loaded 20 products from products.json
✅ Model loaded (embedding dimension: 384)
✅ Generated 20 embeddings
✅ Inserted 20 products into collection
✅ ETL Pipeline completed successfully!
```

### Step 5: Start MCP Server

```bash
# In p_discovery_agent directory
python mcp_server_http.py
```

**Expected Output:**
```
🚀 Inizializzazione Risorse Server MCP...
✅ Modello caricato
✅ Collection 'products' pronta
INFO:     Uvicorn running on http://0.0.0.0:8002
```

**Test Server:**
```bash
curl http://localhost:8002/
```

### Step 6: Start ADK Web Interface

```bash
# Open new terminal in project root (keep MCP server running)
adk web
```

**Expected Output:**
```
| ADK Web Server started                                |
| For local testing, access at http://127.0.0.1:8000.   |
```

**Open Browser:**
- Navigate to `http://127.0.0.1:8000`
- Select your project folder
- Start chatting with the agent!

---

## 🧪 Testing the Agent


### Direct API Testing

```bash
# Test MCP endpoint directly
curl -X POST http://localhost:8002/mcp -H "Content-Type: application/json" -d "{\"jsonrpc\":\"2.0\",\"id\":\"test-1\",\"method\":\"tools/call\",\"params\":{\"name\":\"search_products\",\"arguments\":{\"query\":\"waterproof shoes\",\"max_price\":100}}}"
```

---

## 💬 Web Interface Interactions

### Example 1: Natural Language Search

**User Query:**
> "I'm looking for shoes good for rainy weather, preferably under 100 euros"

**Agent Response:**
```
🔍 I found the perfect option for you!

**StormRunner X5** - €89.99 ⭐ 92.3% match
Category: Footwear | Brand: ActiveGear
✅ In Stock

Description: A durable trail running shoe designed for wet conditions. 
Features GORE-TEX lining and high-traction soles.

Would you like to see more options?
```

---

## ☁️ GCP Simulation

The `gcp_simulation/` folder demonstrates serverless deployment readiness using **Google Cloud Functions Framework**.

### Quick Test

```bash
#if you have already installed all the dependencies int the requirements.txt file in the the parent folder
#directly pass to bash simulation_test.sh
#if you just want to use only the gcp simulation then follow the steps below
cd gcp_simulation
python -m venv .venv
.venv/Scripts/activate      
pip install -r .\requirements.txt 

#Common commands for both the previos steps
# On Windows (Git Bash)
bash simulation_test.sh

# On Windows (PowerShell - manual)
functions-framework --target=search_products_function --source=main.py --port=8080
```

**Expected Output:**
```
============================================
☁️  GCP CLOUD FUNCTION SIMULATION TEST
============================================
🚀 Server avviato con PID: 12345
✅ Server pronto! Invio richiesta...

📄 Risposta ricevuta:
{
  "query": "scarpe running",
  "total_results": 5,
  "products": [...]
}

✅ Test completato.
```

**Note:** The `gcp_simulation/requirements.txt` contains only minimal dependencies (no ADK) needed for Cloud Functions deployment.

---

## 🐛 Troubleshooting

### Agent Can't Find Tools

```bash
# 1. Verify MCP server is running
curl http://localhost:8002/

# 2. Test MCP endpoint
curl -X POST http://localhost:8002/mcp -H "Content-Type: application/json" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}"

# 3. Check agent.py has correct URL
# MCP_SERVER_URL = "http://localhost:8002/mcp"
```

### Empty Search Results

```bash
# Check if containers are running
docker ps

# Restart Milvus
docker-compose down
docker-compose up -d
# Re-run data ingestion
python ingest.py
```

### GCP Simulation Fails

```bash
# Check error logs
type gcp_simulation\server.log

# Install dependencies
cd gcp_simulation
pip install -r requirements.txt

# Test manually
functions-framework --target=search_products_function --source=main.py --port=8080
```

### API Key Issues

1. Verify `.env` file exists in project root
2. Check key has no extra spaces or quotes
3. Restart ADK web server after changing `.env`
4. Verify key is active at [aistudio.google.com](https://aistudio.google.com/)
