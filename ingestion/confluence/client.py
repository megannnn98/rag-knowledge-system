"""Confluence REST client — Basic Auth, pagination, retry on transient errors.

Ported from the reference implementation in the integra-rag project
(confluence_rag/loader/client.py), which had already been validated against
a real Confluence Server instance. Two deliberate differences from that
original:

  - auth is passed per REQUEST (httpx.BasicAuth) rather than baked into the
    AsyncClient constructor, so an injected test client authenticates the
    same way the production one does — in the original, injecting a client
    silently bypassed auth entirely, which is exactly the property a test
    should be able to assert on.
  - list_pages() asks for `expand=version,space` so a sync can tell an
    unchanged page from a changed one WITHOUT downloading every body first
    (see api/confluence.py). The original always re-fetched every page.

Credentials never appear in log records or exception messages — see
_request_json(), which reports the URL and status code only. tests/
test_confluence_client.py asserts this.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self
from urllib.parse import urljoin

import httpx

from ingestion.confluence.parser import ConfluenceHtmlParser

logger = logging.getLogger(__name__)

# Status codes worth retrying: rate limiting plus the transient 5xx family.
# A 4xx other than 429 means the request itself is wrong (bad space key, bad
# credentials) — retrying it just repeats the same failure more slowly.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ConfluenceClientError(Exception):
    pass


class ConfluenceConfigError(ConfluenceClientError):
    pass


class ConfluenceNotFoundError(ConfluenceClientError):
    pass


@dataclass(frozen=True)
class ConfluenceComment:
    """One comment on a page. `url` is Confluence's own permalink, which
    carries focusedCommentId so a citation lands on the comment itself rather
    than the top of a long page."""
    id: str
    page_id: str
    version: int
    url: str
    html_content: str
    text_content: str
    author: str | None
    created_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True)
class ConfluencePage:
    """One Confluence page as this codebase needs it. `html_content` is the
    raw Confluence Storage Format body; `text_content` is that body after
    ConfluenceHtmlParser. Both are kept because the parser is the part most
    likely to need re-running/debugging against real pages."""
    id: str
    title: str
    space_key: str
    url: str
    version: int
    html_content: str
    text_content: str
    created_at: datetime | None
    updated_at: datetime | None


class ConfluenceClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        api_token: str | None = None,
        api_page_size: int = 50,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_sleep_seconds: float = 0.5,
        http_client: httpx.AsyncClient | None = None,
        parser: ConfluenceHtmlParser | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("CONFLUENCE_URL") or "").rstrip("/")
        configured_username = username or os.getenv("CONFLUENCE_USERNAME")
        configured_password = password or os.getenv("CONFLUENCE_PASSWORD")
        configured_api_token = api_token or os.getenv("CONFLUENCE_API_TOKEN")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._api_page_size = api_page_size
        self._max_retries = max_retries
        self._retry_sleep_seconds = retry_sleep_seconds
        self._external_client = http_client is not None
        self._client = http_client
        self._parser = parser or ConfluenceHtmlParser()

        if not self._base_url:
            raise ConfluenceConfigError("CONFLUENCE_URL is required")
        if not configured_username:
            raise ConfluenceConfigError("CONFLUENCE_USERNAME is required")
        secret = configured_password or configured_api_token
        if not secret:
            raise ConfluenceConfigError("CONFLUENCE_PASSWORD or CONFLUENCE_API_TOKEN is required")
        self._username = configured_username
        self._auth = httpx.BasicAuth(configured_username, secret)

    async def __aenter__(self) -> Self:
        self._ensure_client()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and not self._external_client:
            await self._client.aclose()
        self._client = None

    async def get_page(self, page_id: str) -> ConfluencePage:
        data = await self._request_json(
            "GET",
            f"/rest/api/content/{page_id}",
            params={"expand": "body.storage,space,version,history"},
        )
        return self._page_from_api(data)

    async def list_pages(self, space_key: str, limit: int | None = None) -> list[ConfluencePage]:
        """Every page of a space (or the first `limit`), following the API's
        own `_links.next` cursor. Bodies are NOT expanded here — only
        version/space metadata — so a sync can decide what actually needs
        downloading before paying for it."""
        pages: list[ConfluencePage] = []
        start = 0
        while limit is None or len(pages) < limit:
            page_size = self._api_page_size if limit is None else min(self._api_page_size, limit - len(pages))
            data = await self._request_json(
                "GET",
                "/rest/api/content",
                params={
                    "spaceKey": space_key,
                    "type": "page",
                    "expand": "version,space",
                    "limit": str(page_size),
                    "start": str(start),
                },
            )
            results = data.get("results", [])
            if not isinstance(results, list) or not results:
                break
            pages.extend(self._page_from_api(page) for page in results if isinstance(page, dict))
            links = data.get("_links")
            if not isinstance(links, dict) or "next" not in links:
                break
            start += len(results)
        return pages if limit is None else pages[:limit]

    async def list_child_pages(self, page_id: str) -> list[ConfluencePage]:
        """Direct children of a page, in Confluence's own order.

        This instance's /descendant/page endpoint answers HTTP 500, so a tree
        sync recurses through this one level at a time (see
        api/confluence.py::walk_tree). Bodies are not expanded — same reason
        list_pages() doesn't expand them: the version alone decides whether a
        page needs downloading at all."""
        return [
            self._page_from_api(item)
            for item in await self._paginate(f"/rest/api/content/{page_id}/child/page",
                                             {"expand": "version,space"})
        ]

    async def list_comments(self, page_id: str) -> list[ConfluenceComment]:
        """Comments on a page. Bodies ARE expanded here: a comment is small,
        and unlike a page there is no cheaper signal that would tell us
        whether it changed without reading it."""
        return [
            self._comment_from_api(item, page_id)
            for item in await self._paginate(f"/rest/api/content/{page_id}/child/comment",
                                             {"expand": "body.storage,version,history.createdBy"})
        ]

    async def _paginate(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        """Follows `_links.next` until the server stops offering one.

        Trusting `_links.next` rather than comparing `size` to `limit` is
        deliberate: Confluence omits results the caller cannot view, so a page
        of fewer items than requested does NOT mean the listing is finished."""
        results: list[dict[str, Any]] = []
        start = 0
        while True:
            page_params = dict(params)
            page_params.update({"limit": str(self._api_page_size), "start": str(start)})
            data = await self._request_json("GET", path, params=page_params)
            batch = data.get("results", [])
            if not isinstance(batch, list) or not batch:
                return results
            results.extend(item for item in batch if isinstance(item, dict))
            links = data.get("_links")
            if not isinstance(links, dict) or "next" not in links:
                return results
            start += len(batch)

    async def _request_json(self, method: str, path: str, *, params: dict[str, str]) -> dict[str, Any]:
        client = self._ensure_client()
        url = urljoin(f"{self._base_url}/", path.lstrip("/"))
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.request(
                    method, url, params=params, timeout=self._timeout, auth=self._auth
                )
                if response.status_code == httpx.codes.NOT_FOUND:
                    raise ConfluenceNotFoundError(f"Confluence page not found: {url}")
                if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                    logger.warning(
                        f"Confluence {response.status_code} on {path} — retry {attempt + 1}/{self._max_retries}"
                    )
                    await self._sleep_before_retry()
                    continue
                response.raise_for_status()
                json_data = response.json()
                if not isinstance(json_data, dict):
                    raise ConfluenceClientError("Confluence API returned non-object JSON")
                return json_data
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                if attempt >= self._max_retries:
                    # str(error) only — never the request object, whose repr
                    # would carry the Authorization header.
                    raise ConfluenceClientError(f"Confluence request failed after retries: {url}") from None
                logger.warning(f"Confluence transport error on {path} ({type(error).__name__}) — retrying")
                await self._sleep_before_retry()
            except httpx.HTTPStatusError as error:
                raise ConfluenceClientError(
                    f"Confluence API error: {error.response.status_code}"
                ) from None

        raise ConfluenceClientError(f"Confluence request failed after retries: {url}")

    async def _sleep_before_retry(self) -> None:
        if self._retry_sleep_seconds > 0:
            await asyncio.sleep(self._retry_sleep_seconds)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url)
        return self._client

    def _page_from_api(self, data: dict[str, Any]) -> ConfluencePage:
        page_id = str(data.get("id", ""))
        title = str(data.get("title", ""))
        space = data.get("space")
        version = data.get("version")
        history = data.get("history")
        body = data.get("body")
        links = data.get("_links")

        html_content = self._extract_html(body)
        version_number = int(version.get("number", 0)) if isinstance(version, dict) else 0
        updated_at = self._parse_datetime(version.get("when")) if isinstance(version, dict) else None
        created_at = self._parse_datetime(history.get("createdDate")) if isinstance(history, dict) else None
        space_key = str(space.get("key", "")) if isinstance(space, dict) else ""
        webui = str(links.get("webui", "")) if isinstance(links, dict) else ""
        url = urljoin(f"{self._base_url}/", webui.lstrip("/")) if webui else ""

        return ConfluencePage(
            id=page_id,
            title=title,
            space_key=space_key,
            url=url,
            version=version_number,
            html_content=html_content,
            text_content=self._parser.parse(html_content) if html_content else "",
            created_at=created_at,
            updated_at=updated_at,
        )

    def _comment_from_api(self, data: dict[str, Any], page_id: str) -> ConfluenceComment:
        comment_id = str(data.get("id", ""))
        version = data.get("version")
        history = data.get("history")
        links = data.get("_links")

        html_content = self._extract_html(data.get("body"))
        version_number = int(version.get("number", 0)) if isinstance(version, dict) else 0
        updated_at = self._parse_datetime(version.get("when")) if isinstance(version, dict) else None
        created_by = (history or {}).get("createdBy") if isinstance(history, dict) else None
        author = str(created_by.get("displayName", "")) or None if isinstance(created_by, dict) else None
        created_at = self._parse_datetime(history.get("createdDate")) if isinstance(history, dict) else None
        webui = str(links.get("webui", "")) if isinstance(links, dict) else ""
        url = urljoin(f"{self._base_url}/", webui.lstrip("/")) if webui else ""

        return ConfluenceComment(
            id=comment_id,
            page_id=page_id,
            version=version_number,
            url=url,
            html_content=html_content,
            text_content=self._parser.parse(html_content) if html_content else "",
            author=author,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _extract_html(self, body: object) -> str:
        if not isinstance(body, dict):
            return ""
        storage = body.get("storage")
        if not isinstance(storage, dict):
            return ""
        return str(storage.get("value", ""))

    def _parse_datetime(self, value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
