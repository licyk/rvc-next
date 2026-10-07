"""The API over TestClient: error mapping, security, raw uploads, Range, browse roots, jobs."""

import pytest
from fastapi.testclient import TestClient

from rvc_next.api.app import create_app

ORIGIN = {"origin": "http://localhost"}


@pytest.fixture
def app(services):
    return create_app(services, bound_host="127.0.0.1", serve_ui=False)


@pytest.fixture
def client(app):
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def test_health_meta_version(client):
    assert client.get("/api/v1/app/health").json() == {"status": "ok", "auth_required": False}
    meta = client.get("/api/v1/app/meta").json()
    assert meta["local"] is True and meta["trusted"] is True and "convert" in meta["features"]
    assert client.get("/api/v1/app/version").json()["version"]


def test_error_shape(client):
    r = client.get("/api/v1/models/nope")
    assert r.status_code == 404 and r.json()["code"] == "not_found" and "nope" in r.json()["message"]
    r = client.post("/api/v1/presets", json={"name": " ", "params": {}}, headers=ORIGIN)
    assert r.status_code == 400 and r.json()["code"] == "invalid_input"
    r = client.patch("/api/v1/settings", json={"live": {"stream": {"block_ms": 1}}}, headers=ORIGIN)
    assert r.status_code == 400


def test_host_origin_and_token(app, services):
    with TestClient(app, base_url="http://evil.example") as c:
        assert c.get("/api/v1/app/health").status_code == 400
    with TestClient(app, base_url="http://localhost") as c:
        r = c.post("/api/v1/compute/release", headers={"origin": "http://evil.example"})
        assert r.status_code == 403
        services.settings.update({"server": {"access_token": "s3cret"}})
        assert c.get("/api/v1/models").status_code == 401
        assert c.get("/api/v1/app/health").status_code == 200
        assert c.get("/api/v1/models", headers={"authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/v1/models?token=s3cret").status_code == 200


def test_upload_import_and_presets(client, services, tiny_voice_file):
    r = client.put("/api/v1/models/import?filename=tiny.pth&name=Tiny", content=tiny_voice_file.read_bytes(), headers=ORIGIN)
    assert r.status_code == 200, r.text
    voice = r.json()["voices"][0]
    assert voice["name"] == "Tiny" and voice["sample_rate"] == 40000
    presets = client.get(f"/api/v1/presets?voice_id={voice['id']}").json()
    assert presets[0]["is_default"] and presets[0]["params"]["f0_method"] == "rmvpe"
    r = client.post("/api/v1/presets", json={"name": "Up", "voice_id": voice["id"], "params": {"pitch": 5}}, headers=ORIGIN)
    assert r.status_code == 200 and r.json()["params"]["pitch"] == 5
    r = client.patch(f"/api/v1/models/{voice['id']}", json={"description": "hello"}, headers=ORIGIN)
    assert r.json()["description"] == "hello"


def test_audio_upload_range_and_peaks(client, wav_file):
    data = wav_file.read_bytes()
    r = client.put("/api/v1/audio/uploads?name=speech.wav", content=data, headers=ORIGIN)
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["sample_rate"] == 22050 and f["channels"] == 1 and abs(f["duration"] - 1.5) < 0.01
    r = client.get(f"/api/v1/audio/files/{f['id']}/content", headers={"range": "bytes=0-99"})
    assert r.status_code == 206 and r.content == data[:100]
    peaks = client.get(f"/api/v1/audio/files/{f['id']}/peaks?points=64").json()
    assert peaks["points"] == 64 and len(peaks["data"]) == 128


def test_browse_roots_are_enforced(client, services, tmp_path, wav_file):
    services.settings.update({"paths": {"browse_roots": [str(wav_file.parent)]}})
    listing = client.get("/api/v1/audio/browse", params={"path": str(wav_file.parent)}).json()
    assert [e["name"] for e in listing["entries"]] == ["speech.wav"]
    r = client.get("/api/v1/audio/browse", params={"path": str(tmp_path)})
    assert r.status_code == 400 and r.json()["code"] == "invalid_path"
    r = client.post("/api/v1/convert", json={"inputs": [{"kind": "path", "path": "/etc/passwd"}], "voice_id": "x"}, headers=ORIGIN)
    assert r.status_code == 400 and r.json()["code"] == "invalid_path"


def test_convert_job_over_api(client, services, tiny_voice_file, wav_file, no_asset_checks):
    import time

    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    up = client.put("/api/v1/audio/uploads?name=a.wav", content=wav_file.read_bytes(), headers=ORIGIN).json()
    services.jobs.start()
    r = client.post("/api/v1/convert", json={"inputs": [{"kind": "upload", "id": up["id"]}], "voice_id": voice.id, "params": {"f0_method": "pm"}}, headers=ORIGIN)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    for _ in range(300):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["state"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job["state"] == "completed", job
    outs = client.get(f"/api/v1/outputs?job_id={job_id}").json()["items"]
    assert len(outs) == 1 and outs[0]["voice"]["f0_method"] == "pm"
    assert client.get(f"/api/v1/outputs/{outs[0]['id']}/source").status_code == 200
    z = client.post("/api/v1/outputs/zip", json={"ids": [outs[0]["id"]]}, headers=ORIGIN)
    assert z.status_code == 200 and z.content[:2] == b"PK"
    assert client.get("/api/v1/jobs?state=finished").json()["items"][0]["id"] == job_id


def test_openapi_lists_events(client):
    schema = client.get("/openapi.json").json()
    events = schema["components"]["schemas"]["ServerEvents"]["properties"]
    assert {"job_updated", "live_stats", "devices_changed", "train_metrics"} <= set(events)


def test_tts_over_api(client, services, wav_file):
    from tests.core.test_tts import FakeTts

    services.tts.backend = FakeTts(wav_file.read_bytes())
    assert client.get("/api/v1/tts/voices").json()["voices"][0]["id"] == "zh-CN-XiaoxiaoNeural"
    r = client.post("/api/v1/tts", json={"text": "你好", "voice": "zh-CN-XiaoxiaoNeural"}, headers=ORIGIN)
    assert r.status_code == 200 and r.json()["name"] == "你好.mp3"
    assert client.get(f"/api/v1/audio/files/{r.json()['id']}").status_code in (200, 206)
    assert client.post("/api/v1/tts", json={"text": "", "voice": "x"}, headers=ORIGIN).status_code == 422


LOCAL = {"host": "localhost"}  # the test client's sockets say "testserver", which the host check refuses


def test_browser_audio_socket(client, app, services):
    from starlette.websockets import WebSocketDisconnect

    with client.websocket_connect("/api/v1/live/browser-audio", headers=LOCAL) as ws:
        assert wait_until(lambda: services.live._browser is not None)
        link = services.live._browser
        ws.send_bytes(b"\x00\x00" * 160)
        assert wait_until(lambda: link.received == 1)
        link.send(b"\x01\x00" * 4)  # what the worker would send back
        assert ws.receive_bytes() == b"\x01\x00" * 4
    assert wait_until(lambda: services.live._browser is None)
    services.settings.update({"server": {"access_token": "s3cret"}})
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/v1/live/browser-audio", headers=LOCAL) as ws:
        ws.receive_bytes()
    with client.websocket_connect("/api/v1/live/browser-audio?token=s3cret", headers=LOCAL):
        assert wait_until(lambda: services.live._browser is not None)


def wait_until(predicate, timeout=5.0):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False
