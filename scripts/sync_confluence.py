"""Trigger a Confluence space sync against a RUNNING API instance.

Deliberately an HTTP client, not a standalone ingestion script. Two reasons:

  - api/main.py holds a Postgres advisory lock and refuses to start a second
    instance (see lock.py). A script that did its own ingestion would either
    be refused that lock or, worse, bypass it and mutate the same registries
    the live API keeps in memory — which is exactly the desync the guard
    exists to prevent.
  - the models are already loaded in the API process; loading a second copy
    of bge-m3 + bge-reranker-v2-m3 to index a few dozen pages would double
    VRAM for no reason.

    ./venv/bin/python scripts/sync_confluence.py --space AISMT
"""
import argparse
import json
import os
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync a Confluence space into the knowledge base")
    parser.add_argument("--space", default=os.getenv("CONFLUENCE_SPACE", ""),
                        help="Confluence space key (default: $CONFLUENCE_SPACE)")
    parser.add_argument("--folder", default="", help="Folder to file the pages under (default: the space key)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only sync the first N pages. NOTE: with a limit, pages deleted upstream are "
                             "NOT removed locally — a partial listing can't prove a page is gone.")
    parser.add_argument("--api", default=os.getenv("RAG_API_URL", "http://localhost:8000"))
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    if not args.space:
        print("error: --space is required (or set CONFLUENCE_SPACE)", file=sys.stderr)
        return 2

    headers = {}
    api_key = os.getenv("API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {"space_key": args.space, "folder": args.folder}
    if args.limit is not None:
        payload["limit"] = args.limit

    try:
        response = httpx.post(f"{args.api}/confluence/sync", json=payload,
                              headers=headers, timeout=args.timeout)
    except httpx.HTTPError as e:
        print(f"error: could not reach the API at {args.api}: {e}", file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(f"error: sync failed ({response.status_code}): {response.text}", file=sys.stderr)
        return 1

    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
