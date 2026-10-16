import hashlib

import httpx
import pytest

from rvc_next.core.assets.models import AssetFile, AssetSpec
from rvc_next.core.errors import AssetMissingError


def _service_with(make_services, files: dict[str, bytes], fail_first: bool = False):
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.split("/resolve/main/", 1)[1].removeprefix("up/")
        body = files[name]
        calls.append({"name": name, "range": request.headers.get("range")})
        if fail_first and len(calls) == 1:
            # A dropped connection after half the file: the retry resumes with Range.
            return httpx.Response(200, content=body[: len(body) // 2], headers={"content-length": str(len(body))})
        rng = request.headers.get("range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            return httpx.Response(206, content=body[start:])
        return httpx.Response(200, content=body)

    from rvc_next.core.context import build_services

    services = make_services()
    services.close()
    services = build_services(data_dir=services.settings.data_dir, environ={}, transport=httpx.MockTransport(handler))
    spec = AssetSpec(
        id="demo",
        title="Demo",
        description="",
        group="inference",
        files=[AssetFile(path=f"demo/{n}", sources={"rvc-model": n, "official": f"up/{n}"}, size=len(b), sha256=hashlib.sha256(b).hexdigest()) for n, b in files.items()],
    )
    services.assets.add_specs([spec])
    return services, calls


def test_download_resumes_and_verifies(make_services):
    data = {"a.bin": b"x" * 5000, "b.bin": b"y" * 300}
    services, calls = _service_with(make_services, data, fail_first=True)
    try:
        assert services.assets.status("demo").state == "missing"
        with pytest.raises(AssetMissingError) as e:
            services.assets.require(["demo"])
        assert e.value.detail["assets"] == ["demo"]
        job = services.assets.download(["demo"], foreground=True)
        assert job.state == "completed", job.error
        status = services.assets.status("demo")
        assert status.state == "installed" and status.verified is True
        assert any(c["range"] for c in calls)
    finally:
        services.close()


def test_corrupt_file_is_reported(make_services):
    services, _ = _service_with(make_services, {"c.bin": b"z" * 100})
    try:
        target = services.assets.root / "demo" / "c.bin"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"q" * 100)
        job = services.assets.download(["demo"], verify_only=True, foreground=True)
        assert job.state == "failed"
        assert services.assets.status("demo").state == "corrupt"
    finally:
        services.close()


def test_repository_choice_and_fallback(services):
    rmvpe = services.assets.spec("rmvpe").files[0]
    assert services.assets.url(rmvpe) == "https://huggingface.co/licyk/rvc-model/resolve/main/rmvpe/rmvpe.pt"
    services.settings.update({"downloads": {"repository": "official", "source": "hf-mirror"}})
    assert services.assets.url(rmvpe) == "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
    # The demo voices exist only in rvc-model; they come from there whatever is chosen.
    kiki = services.assets.catalog_voice("kikiv1")
    assert kiki.repositories == ["rvc-model"] and kiki.has_index
    assert services.assets.url(kiki.files[0]).startswith("https://hf-mirror.com/licyk/rvc-model/")
    assert [r.id for r in services.assets.repositories() if r.selected] == ["official"]


def test_modelscope_source(services):
    """ModelScope reaches rvc-model's copy there, whichever repository is chosen; a file only on Hugging Face
    (a community model's own repository) comes from there; the other sources follow as fallbacks."""
    rmvpe = services.assets.spec("rmvpe").files[0]
    services.settings.update({"downloads": {"source": "modelscope"}})
    assert services.assets.urls(rmvpe) == [
        "https://www.modelscope.cn/models/licyks/rvc-model/resolve/master/rmvpe/rmvpe.pt",
        "https://huggingface.co/licyk/rvc-model/resolve/main/rmvpe/rmvpe.pt",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "https://hf-mirror.com/licyk/rvc-model/resolve/main/rmvpe/rmvpe.pt",
        "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    ]
    services.settings.update({"downloads": {"repository": "official"}})
    assert services.assets.url(rmvpe) == "https://www.modelscope.cn/models/licyks/rvc-model/resolve/master/rmvpe/rmvpe.pt"
    assert services.assets.urls(rmvpe)[1] == "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
    snowie = services.assets.spec("community-snowie-40k").files[0]
    assert services.assets.url(snowie).startswith("https://huggingface.co/MUSTAR/SnowieV3.1-40k/resolve/")
    rvc_model = next(r for r in services.assets.repositories() if r.id == "rvc-model")
    assert rvc_model.modelscope == "licyks/rvc-model" and rvc_model.modelscope_revision == "master"


def test_download_falls_back_to_the_next_source(make_services):
    """A file a source lacks (404) or serves with other content comes from the next one; a file stored with
    LF line endings (ModelScope's small text files) is accepted as one of the file's alternates."""
    from rvc_next.core.assets.models import FileContent
    from rvc_next.core.context import build_services

    good = {"a.bin": b"a" * 3000, "b.bin": b"b" * 2000, "c.txt": b"x\r\ny\r\n"}
    lf = b"x\ny\n"
    served: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(f"{request.url.host}{request.url.path}")
        name = request.url.path.rsplit("/", 1)[1]
        if request.url.host == "www.modelscope.cn":
            if name == "a.bin":
                return httpx.Response(404)
            if name == "b.bin":
                return httpx.Response(200, content=b"B" * 2000)
            if name == "c.txt":
                return httpx.Response(200, content=lf)
        return httpx.Response(200, content=good[name])

    data_dir = make_services().settings.data_dir
    services = build_services(data_dir=data_dir, environ={}, transport=httpx.MockTransport(handler))
    try:
        services.settings.update({"downloads": {"source": "modelscope"}})
        files = [
            AssetFile(
                path=f"fb/{n}",
                sources={"rvc-model": n},
                size=len(b),
                sha256=hashlib.sha256(b).hexdigest(),
                alternates=[FileContent(size=len(lf), sha256=hashlib.sha256(lf).hexdigest())] if n == "c.txt" else [],
            )
            for n, b in good.items()
        ]
        services.assets.add_specs([AssetSpec(id="fb", title="Fallback", description="", group="inference", files=files)])
        job = services.assets.download(["fb"], foreground=True)
        assert job.state == "completed", job.error
        root = services.assets.root / "fb"
        assert (root / "a.bin").read_bytes() == good["a.bin"] and (root / "b.bin").read_bytes() == good["b.bin"]
        assert (root / "c.txt").read_bytes() == lf
        assert served == [
            "www.modelscope.cn/models/licyks/rvc-model/resolve/master/a.bin",
            "huggingface.co/licyk/rvc-model/resolve/main/a.bin",
            "www.modelscope.cn/models/licyks/rvc-model/resolve/master/b.bin",
            "huggingface.co/licyk/rvc-model/resolve/main/b.bin",
            "www.modelscope.cn/models/licyks/rvc-model/resolve/master/c.txt",
        ]
        assert not list(root.glob("*.part"))
        status = services.assets.status("fb")
        assert status.state == "installed" and status.verified is True
        assert services.assets.download(["fb"], verify_only=True, foreground=True).state == "completed"
    finally:
        services.close()


def test_download_fails_when_no_source_has_the_file(make_services):
    from rvc_next.core.context import build_services

    data_dir = make_services().settings.data_dir
    services = build_services(data_dir=data_dir, environ={}, transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"short")))
    try:
        services.assets.add_specs(
            [AssetSpec(id="nf", title="Nowhere", description="", group="inference", files=[AssetFile(path="nf/x.bin", sources={"rvc-model": "x.bin"}, size=100, sha256="0" * 64)])]
        )
        job = services.assets.download(["nf"], foreground=True)
        assert job.state == "failed" and job.error is not None
        assert len(job.error.detail["errors"]) == 3  # huggingface, hf-mirror, modelscope
    finally:
        services.close()


def test_voice_download_adds_to_library(make_services, tmp_path):
    from rvc_next.core.assets.models import CatalogVoice
    from rvc_next.core.context import build_services
    from tests.tiny import make_tiny_voice

    pth = make_tiny_voice(tmp_path / "v.pth").read_bytes()
    index = b"fake index bytes"
    served: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(request.url.path)
        return httpx.Response(200, content=pth if request.url.path.endswith(".pth") else index)

    data_dir = make_services().settings.data_dir
    services = build_services(data_dir=data_dir, environ={}, transport=httpx.MockTransport(handler))
    try:
        voice = CatalogVoice.model_validate(
            {
                "id": "tiny",
                "name": "Tiny demo",
                "description": "test",
                "sample_rate": 40000,
                "version": "v2",
                "pitch_guidance": True,
                "has_index": True,
                "files": [
                    {"role": "model", "size": len(pth), "sha256": hashlib.sha256(pth).hexdigest(), "sources": {"rvc-model": "voices/tiny/tiny.pth"}},
                    {"role": "index", "size": len(index), "sha256": hashlib.sha256(index).hexdigest(), "sources": {"rvc-model": "voices/tiny/tiny.index"}},
                ],
            }
        )
        services.assets._voices = {"tiny": voice}
        job = services.assets.download_voices(["tiny"], foreground=True)
        assert job.state == "completed", job.error
        assert served == ["/licyk/rvc-model/resolve/main/voices/tiny/tiny.pth", "/licyk/rvc-model/resolve/main/voices/tiny/tiny.index"]
        assert job.result is not None
        added = services.models.get(job.result["voices"]["tiny"])
        assert added.name == "Tiny demo" and added.has_index and added.catalog_id == "tiny"
        assert services.assets.voice_catalog()[0].installed_voice_id == added.id
        with pytest.raises(Exception):
            services.assets.download_voices([], all_missing=True)
    finally:
        services.close()


def test_delete_an_asset_then_download_it_again(make_services):
    services, _ = _service_with(make_services, {"d.bin": b"d" * 400})
    try:
        assert services.assets.download(["demo"], foreground=True).state == "completed"
        assert services.assets.status("demo").state == "installed"
        status = services.assets.delete("demo")
        assert status.state == "missing" and not (services.assets.root / "demo" / "d.bin").exists()
        assert services.assets.download(["demo"], foreground=True).state == "completed"
        assert services.assets.status("demo").state == "installed"
    finally:
        services.close()


def test_official_base_model_deletes_only_its_own_files(services):
    root = services.assets.root / "pretrained_v2"
    root.mkdir(parents=True)
    for name in ("f0G40k.pth", "f0D40k.pth", "G40k.pth", "D40k.pth"):
        (root / name).write_bytes(b"x")
    services.base_models.delete("official-v2-40k-f0")
    assert sorted(p.name for p in root.iterdir()) == ["D40k.pth", "G40k.pth"]
    assert not services.base_models.get("official-v2-40k-f0").installed
    assert services.base_models.get("official-v2-40k").installed


def test_download_reports_speed_and_eta(services):
    """Bytes over time give a rate and the time left; a file starting over resets the window."""
    from rvc_next.core.jobs.service import JobContext, JobSpec

    seen = []

    def run(ctx: JobContext):
        import time

        for done in (0, 100, 200):
            ctx.transfer(done, 1000)
            time.sleep(0.3)
        seen.append(services.jobs._active[ctx.job_id].job.transfer)
        ctx.transfer(50, 1000)
        seen.append(services.jobs._active[ctx.job_id].job.transfer)
        return {}

    services.jobs.register("download", lambda req: JobSpec(title="d", run=run, resources=frozenset({"io"})))
    job = services.jobs.run_now("download", {})
    measured, restarted = seen
    assert measured.done_bytes == 200 and measured.total_bytes == 1000
    assert 250 < measured.bytes_per_second < 400 and 2 < measured.eta_seconds < 4
    assert restarted.bytes_per_second is None and restarted.eta_seconds is None
    assert job.transfer is None


def test_community_base_models(services):
    from rvc_next.core.errors import AssetMissingError

    community = [b for b in services.base_models.list_base(sample_rate="40k", version="v2", pitch_guidance=True) if b.source == "community"]
    assert community and all(not b.installed and b.has_discriminator for b in community)
    snowie = next(b for b in community if b.id == "community-snowie-40k")
    spec = services.assets.spec(snowie.id)
    assert spec.base_model and spec.base_model.homepage.startswith("https://huggingface.co/MUSTAR/")
    # Fetched from the author's repository, pinned to a revision; never offered as a download mirror.
    assert services.assets.url(spec.files[0]).startswith("https://huggingface.co/MUSTAR/SnowieV3.1-40k/resolve/")
    assert {r.id for r in services.assets.repositories()} == {"rvc-model", "official"}
    assert snowie.id not in services.assets.ids_for_group("all")
    with pytest.raises(AssetMissingError):
        services.base_models.paths(snowie.id)
