#!/usr/bin/env bash
# Starts the API against the locally-isolated stack (see local/docker-compose.override.yml).
# Model/URL settings are exported here rather than in a .env file so this
# deployment can never be confused with another stack's own settings; anything
# that is a credential lives in the two gitignored env files sourced below and
# is deliberately NOT in this script.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f local/infra.env ]; then
  echo "local/infra.env is missing — copy local/infra.env.example and fill it in" >&2
  exit 1
fi

# Postgres/Qdrant credentials, shared with docker compose (which reads the same
# file via --env-file), so the API and the containers can never drift apart.
set -a
. ./local/infra.env
set +a

export POSTGRES_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_HOST_PORT:-55433}/${POSTGRES_DB}"
export QDRANT_URL="http://localhost:${QDRANT_HOST_PORT:-6335}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-knowledge_base}"
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
export LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
export QUERY_EXPANDER_MODEL="${QUERY_EXPANDER_MODEL:-qwen3:8b}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
export RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
export PDF_OCR_LANGUAGE="${PDF_OCR_LANGUAGE:-rus+eng}"
export API_KEY="${API_KEY:-}"

# Confluence credentials, if configured. Kept in their own file (gitignored,
# never read by anything but this line) so they are not part of this script
# and never reach shell history.
if [ -f local/confluence.env ]; then
  set -a
  . ./local/confluence.env
  set +a
fi

exec ./venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 "$@"
