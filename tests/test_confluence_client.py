"""Tests for ingestion/confluence/client.py — auth, pagination, retry, and
the property that credentials never reach a log record or an exception."""
import logging

import httpx
import pytest

from ingestion.confluence.client import (
    ConfluenceClient,
    ConfluenceClientError,
    ConfluenceConfigError,
    ConfluenceNotFoundError,
)

SECRET = "sup3r-secret-password"


def _api_page(page_id: str, title: str = "Глоссарий основных сущностей ISMT", version: int = 1) -> dict:
    return {
        "id": page_id,
        "title": title,
        "space": {"key": "AISMT"},
        "_links": {"webui": f"/display/AISMT/{page_id}"},
        "body": {"storage": {"value": "<h1>Title</h1><p>Body text.</p>"}},
        "history": {"createdDate": "2026-09-20T10:00:00.000+00:00"},
        "version": {"number": version, "when": "2026-09-21T10:00:00.000+00:00"},
    }


def _client(handler, **kwargs) -> ConfluenceClient:
    return ConfluenceClient(
        base_url="https://confluence.example.com",
        username="user@example.com",
        password=SECRET,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_sleep_seconds=0,
        **kwargs,
    )


async def test_get_page_maps_every_field_we_index():
    client = _client(lambda _r: httpx.Response(200, json=_api_page("4242", version=7)))

    page = await client.get_page("4242")

    assert page.id == "4242"
    assert page.title == "Глоссарий основных сущностей ISMT"
    assert page.space_key == "AISMT"
    assert page.version == 7
    assert page.url == "https://confluence.example.com/display/AISMT/4242"
    assert page.text_content == "Title\n\nBody text."
    assert page.updated_at is not None


async def test_requests_carry_basic_auth():
    """The injected-client path must authenticate too — otherwise a test can
    never prove the production path does."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json=_api_page("1"))

    await _client(handler).get_page("1")

    assert seen["auth"].startswith("Basic ")
    # Base64 of "user@example.com:sup3r-secret-password"
    assert seen["auth"] == "Basic dXNlckBleGFtcGxlLmNvbTpzdXAzci1zZWNyZXQtcGFzc3dvcmQ="


async def test_api_token_is_accepted_instead_of_password():
    client = ConfluenceClient(
        base_url="https://confluence.example.com",
        username="user@example.com",
        api_token="a-token",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=_api_page("1")))),
        retry_sleep_seconds=0,
    )

    assert (await client.get_page("1")).id == "1"


def test_missing_credentials_are_rejected_at_construction(monkeypatch):
    for var in ("CONFLUENCE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_PASSWORD", "CONFLUENCE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ConfluenceConfigError, match="CONFLUENCE_URL"):
        ConfluenceClient()
    with pytest.raises(ConfluenceConfigError, match="CONFLUENCE_USERNAME"):
        ConfluenceClient(base_url="https://confluence.example.com")
    with pytest.raises(ConfluenceConfigError, match="CONFLUENCE_PASSWORD"):
        ConfluenceClient(base_url="https://confluence.example.com", username="u")


async def test_list_pages_follows_the_next_cursor_until_exhausted():
    seen_starts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        start = request.url.params.get("start", "0")
        seen_starts.append(start)
        if start == "0":
            return httpx.Response(200, json={
                "results": [_api_page("1"), _api_page("2")],
                "_links": {"next": "/rest/api/content?start=2"},
            })
        return httpx.Response(200, json={"results": [_api_page("3")], "_links": {}})

    pages = await _client(handler, api_page_size=2).list_pages("AISMT")

    assert [p.id for p in pages] == ["1", "2", "3"]
    assert seen_starts == ["0", "2"]


async def test_list_pages_requests_version_without_bodies():
    """Unchanged-page detection needs the version; downloading every body to
    get it would defeat the point of skipping unchanged pages."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["expand"] = request.url.params.get("expand", "")
        seen["spaceKey"] = request.url.params.get("spaceKey", "")
        return httpx.Response(200, json={"results": [_api_page("1")], "_links": {}})

    await _client(handler).list_pages("AISMT")

    assert "version" in seen["expand"]
    assert "body" not in seen["expand"]
    assert seen["spaceKey"] == "AISMT"


async def test_list_pages_honours_limit():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "results": [_api_page("1"), _api_page("2"), _api_page("3")],
            "_links": {"next": "/rest/api/content?start=3"},
        })

    pages = await _client(handler).list_pages("AISMT", limit=2)

    assert [p.id for p in pages] == ["1", "2"]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_transient_statuses_are_retried_then_succeed(status):
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(status)
        return httpx.Response(200, json=_api_page("1"))

    page = await _client(handler).get_page("1")

    assert calls["n"] == 2
    assert page.id == "1"


async def test_retries_are_bounded():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    with pytest.raises(ConfluenceClientError):
        await _client(handler, max_retries=2).get_page("1")

    assert calls["n"] == 3  # initial attempt + 2 retries, then give up


async def test_not_found_is_not_retried():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    with pytest.raises(ConfluenceNotFoundError):
        await _client(handler).get_page("nope")

    assert calls["n"] == 1


async def test_credentials_never_reach_logs_or_error_messages(caplog):
    """A Confluence failure gets logged and re-raised on a path that has the
    password in scope — the request object's repr carries the Authorization
    header, so anything that formats it leaks the credential into logs."""
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ConfluenceClientError) as exc_info:
            await _client(handler, max_retries=1).get_page("1")

    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)
    # Also nothing chained underneath it (raise ... from None).
    assert exc_info.value.__cause__ is None
    assert SECRET not in caplog.text
    assert "dXNlckBleGFtcGxlLmNvbTpz" not in caplog.text  # nor its base64 form


async def test_http_error_message_carries_status_only(caplog):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "unauthorized"})

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ConfluenceClientError) as exc_info:
            await _client(handler).get_page("1")

    assert "401" in str(exc_info.value)
    assert SECRET not in str(exc_info.value)
    assert SECRET not in caplog.text
