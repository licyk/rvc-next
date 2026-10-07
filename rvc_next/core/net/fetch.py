"""Fetch a file from a URL a user gave, guarded: the model import's "Add from a link".

Links people share for voices point at Hugging Face (a file, or a repository or folder to list), at
Google Drive, or at any web server. ``resolve`` turns a link into the files to fetch; ``fetch``
downloads one into a folder.

Every request is checked before and after connecting (server-side request forgery): only http and
https, and, unless ``allow_private`` (a request from the server's own machine, or the command line),
neither the host's addresses nor the address actually connected to may be loopback, private,
link-local, multicast or reserved. Redirects are followed one hop at a time, each hop checked. A
download stops past ``max_bytes``.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import httpx

from rvc_next.core.errors import ValidationError

MAX_REDIRECTS = 8
MAX_BYTES = 8 * 1024**3
CHUNK = 1024 * 1024
HF_HOSTS = {"huggingface.co", "hf.co", "www.huggingface.co", "hf-mirror.com"}
_HF_FILE = re.compile(r"^/(?P<repo>(?:(?:datasets|spaces)/)?[^/]+/[^/]+)/(?:blob|resolve)/(?P<rev>[^/]+)/(?P<path>.+)$")
_HF_TREE = re.compile(r"^/(?P<repo>(?:(?:datasets|spaces)/)?[^/]+/[^/]+)(?:/tree/(?P<rev>[^/]+)(?:/(?P<path>.*))?)?/?$")
_DRIVE = re.compile(r"/file/d/(?P<id>[A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class RemoteFile:
    url: str
    name: str
    size: int | None = None


def _blocked(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%", 1)[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_multicast or addr.is_reserved or addr.is_unspecified


def check_url(url: str, allow_private: bool) -> None:
    """Refuse a URL that is not http(s), or whose host resolves to a non-public address."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValidationError(f"Not an http or https link: {url}", {"reason": "url"})
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValidationError(f"Cannot resolve {parts.hostname}: {e}", {"reason": "url"}) from e
    if any(_blocked(str(info[4][0])) for info in infos):
        raise ValidationError(f"{parts.hostname} is a local or private address; links must point at the internet", {"reason": "url_blocked"})


def _check_peer(response: httpx.Response, allow_private: bool, proxied: bool) -> None:
    if allow_private or proxied:
        return
    stream = response.extensions.get("network_stream")
    peer = stream.get_extra_info("server_addr") if stream is not None else None
    if peer and _blocked(str(peer[0])):
        raise ValidationError("The link led to a local or private address", {"reason": "url_blocked"})


def _proxied(client: httpx.Client) -> bool:
    """Whether requests go through a proxy (from the environment): then the peer is the proxy."""
    return any(t is not None for t in (getattr(client, "_mounts", None) or {}).values())


def _open(client: httpx.Client, url: str, allow_private: bool, headers: dict[str, str] | None = None) -> httpx.Response:
    """GET ``url`` as a stream, following redirects one checked hop at a time. The caller closes it."""
    proxied = _proxied(client)
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url, allow_private)
        response = client.send(client.build_request("GET", url, headers=headers), stream=True, follow_redirects=False)
        try:
            _check_peer(response, allow_private, proxied)
        except BaseException:
            response.close()
            raise
        if response.is_redirect and "location" in response.headers:
            url = urljoin(str(response.url), response.headers["location"])
            response.close()
            continue
        return response
    raise ValidationError("Too many redirects", {"reason": "url"})


def _filename(response: httpx.Response, fallback: str) -> str:
    header = response.headers.get("content-disposition")
    if header:
        message = Message()
        message["content-disposition"] = header
        name = message.get_param("filename*", header="content-disposition") or message.get_param("filename", header="content-disposition")
        if isinstance(name, tuple):
            name = name[2]
        if name:
            name = unquote(str(name).removeprefix("UTF-8''").removeprefix("utf-8''"))
            if name:
                return Path(name.replace("\\", "/")).name
    return fallback


def _hf_endpoint(endpoint: str, host: str) -> str:
    """A Hugging Face link goes through the configured endpoint (a mirror), unless it names one."""
    return endpoint.rstrip("/") if host in ("huggingface.co", "hf.co", "www.huggingface.co") else f"https://{host}"


def resolve(url: str, client: httpx.Client, endpoint: str, allow_private: bool, suffixes: set[str]) -> list[RemoteFile]:
    """The files a link stands for: a Hugging Face file, every file with one of ``suffixes`` in a
    Hugging Face repository or folder, a Google Drive file, or the link itself."""
    url = url.strip()
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    if host in HF_HOSTS:
        base = _hf_endpoint(endpoint, host)
        path = unquote(parts.path)
        m = _HF_FILE.match(path)
        if m:
            name = m["path"].rsplit("/", 1)[-1]
            return [RemoteFile(f"{base}/{m['repo']}/resolve/{m['rev']}/{quote(m['path'])}", name)]
        m = _HF_TREE.match(path)
        if m:
            repo, rev, folder = m["repo"], m["rev"] or "main", (m["path"] or "").strip("/")
            kind = "datasets" if repo.startswith("datasets/") else "spaces" if repo.startswith("spaces/") else "models"
            repo_id = repo.split("/", 1)[1] if kind != "models" else repo
            api = f"{base}/api/{kind}/{repo_id}/tree/{quote(rev)}" + (f"/{quote(folder)}" if folder else "") + "?recursive=true"
            response = _open(client, api, allow_private)
            try:
                response.read()
                if response.status_code != 200:
                    raise ValidationError(f"Cannot list {repo} ({response.status_code})", {"reason": "url"})
                entries = response.json()
            finally:
                response.close()
            prefix = "" if kind == "models" else f"{kind}/"
            files = [
                RemoteFile(f"{base}/{prefix}{repo_id}/resolve/{quote(rev)}/{quote(e['path'])}", e["path"].rsplit("/", 1)[-1], e.get("size"))
                for e in entries
                if e.get("type") == "file" and Path(e["path"]).suffix.lower() in suffixes
            ]
            if not files:
                raise ValidationError(f"{repo} has no model, index or archive files", {"reason": "url_empty"})
            return files
    if host in ("drive.google.com", "docs.google.com"):
        m = _DRIVE.search(parts.path)
        file_id = m["id"] if m else (parse_qs(parts.query).get("id") or [None])[0]
        if file_id:
            # The direct download endpoint; ``confirm=t`` skips the "cannot scan for viruses" page of large files.
            return [RemoteFile(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t", f"{file_id}.bin")]
    return [RemoteFile(url, Path(unquote(parts.path)).name or "download")]


def fetch(
    client: httpx.Client,
    file: RemoteFile,
    folder: Path,
    allow_private: bool,
    on_bytes: Callable[[int, int | None], None] | None = None,
    check_cancel: Callable[[], Any] | None = None,
    max_bytes: int = MAX_BYTES,
) -> Path:
    """Download ``file`` into ``folder`` under its served name; returns the path."""
    folder.mkdir(parents=True, exist_ok=True)
    response = _open(client, file.url, allow_private)
    try:
        if response.status_code != 200:
            raise ValidationError(f"HTTP {response.status_code} for {file.url}", {"reason": "url"})
        if "text/html" in response.headers.get("content-type", "") and not file.name.lower().endswith((".htm", ".html")):
            raise ValidationError(f"{file.url} is a web page, not a file (a link that needs a sign-in or a confirmation?)", {"reason": "url"})
        total = int(response.headers["content-length"]) if response.headers.get("content-length", "").isdigit() else file.size
        if total and total > max_bytes:
            raise ValidationError(f"{file.name} is larger than {max_bytes // 1024**3} GiB", {"reason": "too_large"})
        name = _filename(response, file.name) or "download"
        target = folder / name
        part = target.with_name(target.name + ".part")
        done = 0
        try:
            with open(part, "wb") as out:
                for chunk in response.iter_bytes(CHUNK):
                    if check_cancel:
                        check_cancel()
                    done += len(chunk)
                    if done > max_bytes:
                        raise ValidationError(f"{name} is larger than {max_bytes // 1024**3} GiB", {"reason": "too_large"})
                    out.write(chunk)
                    if on_bytes:
                        on_bytes(done, total)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(target)
        return target
    finally:
        response.close()
