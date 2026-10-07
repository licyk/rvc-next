"""Links in the model import: what a link stands for, the guard against local addresses, the job."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest

from rvc_next.core.errors import ValidationError
from rvc_next.core.net.fetch import RemoteFile, check_url, fetch, resolve

PUBLIC = "http://93.184.216.34"  # an address literal: checked without DNS


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://x/y", "http://127.0.0.1/a", "http://10.1.2.3/a", "http://[::1]/a", "http://169.254.169.254/latest", "http://0.0.0.0/"]
)
def test_local_and_odd_links_are_refused(url: str) -> None:
    with pytest.raises(ValidationError):
        check_url(url, allow_private=False)


def test_public_and_allowed_links_pass() -> None:
    check_url(f"{PUBLIC}/a.pth", allow_private=False)
    check_url("http://127.0.0.1/a", allow_private=True)


def test_what_a_link_stands_for() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/models/owner/repo/tree/main/sub" and request.url.params["recursive"] == "true"
        return httpx.Response(200, json=[{"type": "file", "path": "sub/v.pth", "size": 5}, {"type": "file", "path": "sub/readme.md"}, {"type": "directory", "path": "sub/x"}])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    file = resolve("https://huggingface.co/owner/repo/blob/main/dir/v.pth", client, "https://hf-mirror.com", True, {".pth"})
    assert file == [RemoteFile("https://hf-mirror.com/owner/repo/resolve/main/dir/v.pth", "v.pth")]
    listed = resolve("https://huggingface.co/owner/repo/tree/main/sub", client, "https://huggingface.co", True, {".pth"})
    assert listed == [RemoteFile("https://huggingface.co/owner/repo/resolve/main/sub/v.pth", "v.pth", 5)]
    drive = resolve("https://drive.google.com/file/d/AbC_123/view?usp=sharing", client, "", True, {".pth"})
    assert drive[0].url == "https://drive.usercontent.google.com/download?id=AbC_123&export=download&confirm=t"
    assert resolve(f"{PUBLIC}/files/My%20Voice.zip", client, "", True, {".zip"})[0].name == "My Voice.zip"


def test_a_redirect_to_a_local_address_is_refused(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8080/secret"})
        return httpx.Response(200, content=b"secret")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValidationError) as e:
        fetch(client, RemoteFile(f"{PUBLIC}/a.pth", "a.pth"), tmp_path, allow_private=False)
    assert e.value.detail["reason"] == "url_blocked"
    assert not list(tmp_path.iterdir())


def test_fetch_names_limits_and_pages(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/page":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>")
        return httpx.Response(200, headers={"content-disposition": "attachment; filename*=UTF-8''Alto%20v2.pth"}, content=b"x" * 100)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = fetch(client, RemoteFile(f"{PUBLIC}/dl?id=1", "1.bin"), tmp_path, allow_private=False)
    assert path.name == "Alto v2.pth" and path.read_bytes() == b"x" * 100
    with pytest.raises(ValidationError):
        fetch(client, RemoteFile(f"{PUBLIC}/big", "big.pth"), tmp_path / "b", allow_private=False, max_bytes=10)
    with pytest.raises(ValidationError):
        fetch(client, RemoteFile(f"{PUBLIC}/page", "p.pth"), tmp_path / "c", allow_private=False)


def test_links_stage_into_an_import(tmp_path: Path, tiny_assets_dir: Path) -> None:
    from rvc_next.core.context import build_services
    from tests.tiny import make_tiny_index, make_tiny_voice

    voice = make_tiny_voice(tmp_path / "alto.pth").read_bytes()
    index = make_tiny_index(tmp_path / "added_IVF4_Flat_nprobe_1_alto_v2.index", 768).read_bytes()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("alto/alto.pth", voice)
        zf.writestr("alto/added_IVF4_Flat_nprobe_1_alto_v2.index", index)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-disposition": 'attachment; filename="alto.zip"'}, content=archive.getvalue())

    services = build_services(data_dir=tmp_path / "data", environ={"RVC_NEXT_PATHS__ASSETS_DIR": str(tiny_assets_dir)}, transport=httpx.MockTransport(handler))
    try:
        sid = services.imports.create()
        job = services.imports.add_urls(sid, [f"{PUBLIC}/share/alto.zip"], foreground=True)
        assert job.state == "completed", job.error
        plan = services.imports.plan(sid)
        assert [f.kind for f in plan.files] == ["voice", "index"] and plan.confident
        assert {f.group for f in plan.files} == {"alto.zip/alto"}
        with pytest.raises(ValidationError):
            services.imports.add_urls(sid, ["file:///etc/passwd"])
    finally:
        services.close()
