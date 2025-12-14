import os
import logging
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

# Configura logs per vedere cosa succede
logging.basicConfig(level=logging.INFO)

# URL del server locale
# Assicurati che mcp_server_http.py stia girando su 8002
MCP_SERVER_URL = "http://localhost:8002/mcp"

print(f"🔗 Configurazione Agente su: {MCP_SERVER_URL}")

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='ecommerce_search_agent',
    instruction="""You are an expert e-commerce shopping assistant helping customers find products and get information.

## Your Approach
Understand customer needs, use available tools intelligently, and provide helpful recommendations.

## Domain Knowledge

### Price Interpretation
When customers use budget terms without numbers:
- "cheap/affordable/economico" → ≤50€
- "moderate/medio prezzo" → 50-100€  
- "premium/luxury" → no constraint

Adjust contextually (electronics vs accessories have different scales).

### Product Categories
Footwear, Clothing, Electronics, Accessories, Outdoor

## Best Practices
- Ask clarifying questions for ambiguous requests
- Explain why recommendations fit their needs
- Suggest alternatives if initial results don't match
- Always check availability before promising items

Be conversational, proactive, and customer-focused.""",
    tools=[
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=MCP_SERVER_URL
            )
        )
    ],
)