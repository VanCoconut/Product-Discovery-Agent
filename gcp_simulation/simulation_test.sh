#!/bin/bash
# simulation_test.sh - Versione Auto-Venv

PORT=8080
TARGET="search_products_function"

echo "============================================"
echo "☁️  GCP CLOUD FUNCTION SIMULATION TEST"
echo "============================================"

# --- LOGICA DI ATTIVAZIONE VENV ---
# Cerca il venv due livelli sopra (se sei in gcp_simulation/) o nella root
if [ -f "../.venv/Scripts/activate" ]; then
    echo "📦 Trovato Venv nella root (../.venv). Attivazione..."
    source "../.venv/Scripts/activate"
elif [ -f ".venv/Scripts/activate" ]; then
    echo "📦 Trovato Venv locale (.venv). Attivazione..."
    source ".venv/Scripts/activate"
else
    echo "⚠️  ATTENZIONE: Nessun .venv trovato! Sto usando il Python Globale (RISCHIO ERRORI)."
fi
# ----------------------------------

# Ora eseguiamo il resto...
functions-framework --target=$TARGET --source=main.py --port=$PORT > server.log 2>&1 &
SERVER_PID=$!

echo "🚀 Server avviato con PID: $SERVER_PID"
echo "⏳ In attesa disponibilità servizio..."

# 2. Loop di controllo (Max 30 secondi)
MAX_RETRIES=30
COUNT=0
SERVER_CRASHED=0

while ! curl -s "http://localhost:$PORT" > /dev/null; do
    # Controllo se il processo esiste ancora
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "❌ ERRORE CRITICO: Il server è crashato durante l'avvio!"
        SERVER_CRASHED=1
        break
    fi

    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo ""
        echo "❌ TIMEOUT: Il server è attivo ma non risponde."
        kill $SERVER_PID
        exit 1
    fi
    printf "."
done

# 3. Gestione esito
if [ $SERVER_CRASHED -eq 1 ]; then
    echo "--- LOG DEGLI ERRORI ---"
    cat server.log
    rm server.log
    exit 1
fi

echo -e "\n✅ Server pronto! Invio richiesta..."

# 4. Test CURL
RESPONSE=$(curl -s -X POST "http://localhost:$PORT" \
  -H "Content-Type: application/json" \
  -d '{"query": "scarpe running", "max_price": 150}')

# 5. Output Risultati
if [ -z "$RESPONSE" ]; then
    echo "❌ ERRORE: Risposta vuota."
else
    echo "📄 Risposta ricevuta:"
    # Tenta di formattare il JSON, altrimenti stampa raw
    echo "$RESPONSE" | python -m json.tool || echo "$RESPONSE"
fi

# 6. Pulizia
kill $SERVER_PID
# rm server.log  <-- Commenta questa riga con un # per NON cancellare il log
echo "--- Log del server salvato in gcp_simulation/server.log ---"
echo "============================================"
echo "✅ Test completato."