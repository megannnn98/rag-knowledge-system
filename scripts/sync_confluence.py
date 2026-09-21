"""Trigger a Confluence sync against a RUNNING API instance.

The documented workflow is a TREE sync: one root page id, everything under
it, at any depth. `--space` is the older flat single-space mode, kept
working but without project/path context or comments.

Deliberately an HTTP client, not a standalone ingestion script. Two reasons:

  - api/main.py holds a Postgres advisory lock and refuses to start a second
    instance (see lock.py). A script that did its own ingestion would either
    be refused that lock or, worse, bypass it and mutate the same registries
    the live API keeps in memory — which is exactly the desync the guard
    exists to prevent.
  - the models are already loaded in the API process; loading a second copy
    of bge-m3 + bge-reranker-v2-m3 to index a few dozen pages would double
    VRAM for no reason.

    ./venv/bin/python scripts/sync_confluence.py --root-page-id 44302866
    CONFLUENCE_ROOT_PAGE_ID=44302866 ./venv/bin/python scripts/sync_confluence.py
"""
import argparse
import json
import os
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Confluence content into the knowledge base")
    parser.add_argument("--root-page-id", default=os.getenv("CONFLUENCE_ROOT_PAGE_ID", ""),
                        help="Root page of the tree to sync, recursively (default: $CONFLUENCE_ROOT_PAGE_ID). "
                             "A page id, not a title — titles get renamed.")
    parser.add_argument("--space", default="",
                        help="Legacy flat single-space sync; no tree context and no comments. "
                             "Ignored when --root-page-id is given.")
    parser.add_argument("--folder", default="", help="Folder to file the documents under")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only sync the first N pages. NOTE: with a limit, content deleted upstream is "
                             "NOT removed locally — a partial walk can't prove anything is gone.")
    parser.add_argument("--api", default=os.getenv("RAG_API_URL", "http://localhost:8000"))
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    if not args.root_page_id and not args.space:
        print("error: --root-page-id is required (or set CONFLUENCE_ROOT_PAGE_ID)", file=sys.stderr)
        return 2

    headers = {}
    api_key = os.getenv("API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {"root_page_id": args.root_page_id, "space_key": "" if args.root_page_id else args.space,
               "folder": args.folder}
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
