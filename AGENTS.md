# rvc-next — notes for agents

The design record: what the code does, the rules it keeps and why. The original implementation
plan has been folded into the developer documentation ([site/content/docs/development*.mdx](site/content/docs/development.mdx),
Chinese with English twins, on the project site); this file records how it was built and where the build departs
from the plan. Add durable findings here, briefly.

## 1. Overview

A rewrite of Retrieval-based-Voice-Conversion-WebUI (RVC, `~/code_workspace/Retrieval-based-Voice-Conversion-WebUI`
at `81eed5e`) as one application: **Convert**, **Live**, **Separate**, **Train** and **Models**
behind one FastAPI server, one Vue 3 web UI and one Typer command line.

- `rvc-next` everywhere: package `rvc_next`, command `rvc-next`, env prefix `RVC_NEXT_`, data
  directory `rvc-next`, port **7868** (clear of RVC/SD Model Hub 7865, IIB 7866, Hanaikada 7867),
  GPL-3.0 (the skeleton is SD Model Hub's, via Hanaikada; RVC and PyMSS notices in `NOTICE`).
- The numerical code is **ported**, not reinvented; golden tests hold it to the original (§5).
- Skeleton from Hanaikada: settings, events, db, security, ports, CLI factory, the UI's theme,
  `ui/`, motion, and `embed.py` (hosting inside another app, §2).

## 2. Layout and layers

```
rvc_next/
  engine/     ML and DSP, no app code: runtime (GPU rule, model cache), graph (CUDA Graphs),
              audio/ (io via PyAV, resample, sinc_resample, dsp, peaks, denoise), audio_io/ (devices, streams,
              rings, drift, tone, fake backend), features/hubert, f0/ (pm, rmvpe, fcpe),
              index/ (retrieval, build), models/ (ported nets, checkpoint, merge, extract, loader),
              convert/ (VoiceParams, OfflineConverter), stream/ (StreamParams, StreamEngine),
              separate/ (presets, pymss runner), train/ (every stage, hparams, data/mute)
  protocol/   worker events (JSON lines) and the live control messages; standard library only
  workers/    python -m rvc_next.workers.{train,separate,live,devices} <request.json>
  core/       services: settings events db net jobs compute assets audio models presets
              conversion separation training live; context.py builds them; errors.py
  api/        FastAPI routers (one per resource), security, sockets, static. Thin.
  cli/        Typer: app.py registers everything; commands/ are plain functions. Thin.
  webui/      the Vue project; dist/ is git-ignored and shipped as package data
scripts/      dev.py, build_wheel.py, generate_openapi.py, golden.py
tests/        engine/ core/ api/ cli/ workers/ golden/, tiny.py (tiny models), conftest.py
rvc_next/embed.py  RvcNextServer / serve(): the server inside another Python program (lazy from `rvc_next`)
site/         the website: home page and docs (Fumadocs on TanStack Start, from Hanafubuki's site)
```

- **The site** (`site/`, its own bun project): `content/docs/<page>.mdx` (Chinese) and
  `<page>.en.mdx` (English) for every page, sidebar order in `meta.json` / `meta.en.json`; the home
  page's copy is `src/lib/site-copy.ts`. A test fails when a page lacks its English twin. Links are
  `/docs/<page>`; text `<`, `{`, `}` must be escaped (MDX). `dev.py site-dev|site|test-site`;
  `.github/workflows/site.yml` publishes to gh-pages (base path `/rvc-next/`), and
  `netlify.toml`, `wrangler.toml` and `site/vercel.json` deploy the same build at the root.

- **Embedding** (`embed.py`, from Hanaikada's): `RvcNextServer(data_dir, config_dir, settings_path,
  host, port, strict_port, api_prefix, open_browser, access_token, settings, log_level)` builds the
  services with `start_background=True`, binds, runs uvicorn on a thread and writes `server.json`
  like `webui`. `config_dir` (also `RVC_NEXT_CONFIG_DIR`, `webui --config-dir`) holds `settings.toml`
  apart from the data; `settings_path` names the file. `settings` are overrides: never written, shown
  in `SettingsView.pinned` (the UI labels them "set by the application"). `api_prefix`
  (`create_app`, `webui --api-prefix`) moves API, socket, docs and UI under one path; the UI derives
  its base from its script URL. `tests/api/test_embed.py`; docs `advanced-embed`.
- **Layer rule** (`tests/test_architecture.py`): `engine` imports no `rvc_next.core/api/cli/workers`
  and no web/CLI framework or pydantic; `sounddevice` and `soundcard` only under `engine/audio_io/`, `pymss` only
  under `engine/separate/`; `protocol` is standard library only; `workers` import `engine` and
  `protocol`; `core` imports neither a framework nor torch directly (torch only through engine
  functions, lazily); `api`/`cli` never import `engine`. A second test fails if importing
  `rvc_next.core.context` or `rvc_next.cli.app` loads torch (or FastAPI for the CLI).
- **One operation, one core method** taking and returning `Record` models; a route or command
  parses, calls it, and shapes the output, so the API's JSON equals the CLI's `--json`
  (`test_model_json_matches_api_shape`).
- `core/context.py:build_services()` builds everything once; the API keeps it on
  `app.state.services` (`ServicesDep`), the CLI opens it per command (`open_services()`), tests pass
  `environ=` and `transport=`. `start_background=True` (the server) starts the job scheduler and
  idle unloading; the CLI runs one job in the foreground (`jobs.run_now`).
- **Errors** are `core/errors.py` only: `NotFound` 404/2, `Conflict`/`Busy` 409/3,
  `InvalidPath`, `Validation`, `ModelFormat` 400/4, `AssetMissing` 409/7 (`detail.assets`),
  `Device` 409/8 (`detail.reason`), `Compute` 503/9, base 500/1. Engine exceptions
  (`engine/errors.py`) are mapped in `core/engine_errors.py`; a worker's `error` event is rebuilt
  with `worker_error`. Commands map them to exit codes inside `DebugTyperCommand.invoke`.

## 3. Commands and workflow

```bash
python scripts/dev.py check          # what CI runs: ruff, ty, pytest, vitest + vue-tsc, the site, generated types
python scripts/dev.py dev            # API + Vite with hot reload (Vite proxies to RVC_NEXT_BACKEND, default :7868)
python scripts/dev.py typegen        # regenerate webui/src/api/schema.d.ts after an API change
python scripts/build_wheel.py        # web UI, then wheel and sdist
python scripts/golden.py --original <RVC checkout> --assets <assets> --voice a.pth --clip x.wav [--f0 pm --f0 rmvpe --f0 fcpe]
```

- **rvc-next runs in its own environment** (a venv or a standalone Python), never a shared one.
  Development uses an isolated `.venv` (git-ignored, no system site packages) built from the py311
  interpreter: `uv venv .venv --python <py311>`, then torch, then
  `uv pip install -e . --group dev`. Run everything with `.venv/bin/python`. PortAudio
  (`libportaudio2`) is installed system-wide.
- **Dependency constraints** (`pyproject.toml`): lower bounds are what the code needs or a
  dependency forces; upper bounds stop at the next untested major (torch excepted). Two are load-bearing:
    - `torch>=2.7.1`, no upper bound: pymss needs 2.7.1; the user installs the torch build for the
      hardware first, and since torchaudio and torchfcpe are gone nothing pins torch to a release.
    - **No torchaudio, no torchfcpe.** Their code rvc-next used is ported: `Resample` in
      `engine/audio/sinc_resample.py`, FCPE (the conv-only model torchfcpe bundles, mel front end,
      decoders, loader) in `engine/f0/fcpe_model.py`, its weights the `fcpe` asset. Both ports are
      bit-identical (tests compare against the packages when they are installed); the layer test
      forbids importing either. The golden tests import them only for the original's side.
    - `transformers>=5,<6`: 4.x needs `huggingface_hub` < 1. Its `tokenizers` dependency (every
      release, 1.0 pre-releases included) needs `huggingface_hub` < 2, which is why rvc-next
      cannot share an environment with projects on `huggingface_hub` 2.x.
  Tested at both ends, from the built wheel (smoke install, then the test suite): Python 3.10 with
  torch 2.7.1 and every direct dependency at its floor (`uv pip install --resolution lowest-direct`),
  and Python 3.12–3.14 with torch 2.11, numpy 2.5, scipy 1.18, librosa 1.0, av 19, send2trash 2
  (those need Python 3.12+; 3.11, the dev env, gets numpy 2.4, librosa 0.11, av 18); torch 2.13 and
  2.14 (the versions the install docs name) on Python 3.12. The floor
  run found `fastapi` 0.115.0's Starlette ignoring Range requests, hence `>=0.115.3`. Standalone
  Pythons for such runs: `sd-webui-all-in-one self-manager python-standalone list`.
- Assets for real runs: a scratch `assets/` in the original's layout; point
  `RVC_NEXT_PATHS__ASSETS_DIR` at it. Tests marked `assets` read `RVC_NEXT_TEST_ASSETS`; `golden`
  tests also need the original checkout. Both are deselected by default.
- **A CLI command** is a plain function in `cli/commands/`, registered with name and help in
  `cli/app.py:get_app()` and added to `EXPECTED_TREE` in `tests/cli/test_cli.py`. Required options
  have no default (order parameters so they come first); never `= ...`.
- **API change → `typegen`** and commit `schema.d.ts`; `typegen-check` fails until it matches.
- Commit and open PRs only when the user asks.

## 4. Conventions

- **Python ≥ 3.10**, ruff and ty unpinned (latest), `lint.select` written out, line length 180.
  Every function in `rvc_next/` is annotated (ruff `ANN`, `Any` allowed); the ported files, tests
  and scripts are exempt.
  The ported files (`engine/models/{attentions,commons,modules,transforms,synthesizer}.py`,
  `engine/f0/rmvpe.py`, `engine/audio/denoise.py`, some `engine/train/*`) keep the original's
  shape: excluded from `ruff format` and ty, with per-file lint ignores, so they diff cleanly
  against the original. Suppressions always name a rule.
- **Pydantic v2 only.** `Record` sets `json_schema_serialization_defaults_required` (the generated
  TypeScript types match what the server sends); the v1 shim of SD Model Hub was dropped
  (FastAPI ≥ 0.100 needs v2 anyway).
- Engine parameters are frozen dataclasses (`VoiceParams`, `StreamParams`); the core mirrors them
  as `VoiceParamsModel`/`StreamParamsModel` with the ranges as field constraints (`core/params.py`).
- **Heavy imports** (torch, transformers, faiss, av, pymss, sounddevice) sit inside functions.
- Methods are never named `list` on a class that also annotates with `list[...]` (ty resolves the
  name to the method): `list_assets`, `list_jobs`, `list_voices`, `list_presets`, `list_experiments`.
- **Settings:** defaults < `settings.toml` < env `RVC_NEXT_<GROUP>__<FIELD>` < overrides; groups
  `server paths compute downloads convert separation training live`. Empty folder
  settings mean "inside the data directory" (`SettingsService.models_dir` etc.). Another process
  writing `settings.toml` (`rvc-next config set`) is picked up within a second and merged into, never
  overwritten, by the next save; every change publishes `settings_changed` with the changed keys
  (`core/context.py`), and a `paths.*` change rescans the library.
- **Events** carry snapshots; job progress is throttled to 4/s and job logs batched (≤ 50 lines).
- **Database** (`rvc-next.db`): three migrations (1: `voice_models presets jobs outputs audio_files
  client_state`; 2: `voice_models.catalog_id`; 3: `voice_models.meta`, JSON with the file's
  `provenance`, and every row re-read). It is a cache for voices (rebuilt from `models/` and legacy roots by
  `models.rescan()`), and the record of jobs and outputs. Experiments are folders, not rows.

## 5. Engine

- **Porting rule.** The nets, RMVPE, TorchGate and the CUDA Graph cache are copied with imports
  fixed. The pipeline maths (offline, streaming, training) is ported line for line around new
  interfaces. Golden runs (`scripts/golden.py`, `tests/golden/`) on CPU fp32, same seed, same
  16 kHz input:
    - offline, pm and RMVPE: **bit-identical** for v1 40k, v2 40k, v2 40k without pitch and
      v2 48k, on speech, silence and a 70 s clip that crosses the chunking threshold, with pitch
      shift and a speaker id; FCPE bit-identical too;
    - streaming: `StreamEngine` bit-identical, block by block, to `rtrvc.RVC.infer` plus the
      realtime GUI's callback maths (gate, TorchGate in and out, formant, RMS mix, SOLA).
  The models must be built **before** seeding: HuBERT, RMVPE and FCPE weight initialisation draws
  from the RNG, and the original builds RMVPE/FCPE inside the pipeline.
- **Runtime** (`engine/runtime.py`): the original's GPU rule as `choose_device(requested,
  precision)`; `ChunkConfig` (x_pad/x_query/x_center/x_max) from the choice; caches HuBERT, F0
  providers, the last 3 voices and indexes keyed by path and mtime; `release_idle`.
- **ROCm and XPU** (after ComfyUI's `model_management`): a ROCm torch shows AMD cards as `cuda:N`
  (`is_rocm()` = `torch.version.hip`); `cuda_profile` skips the SM rule there (the capability is
  the gfx version): ≥ 4 GiB, fp16. Intel cards are `xpu:N` (`xpu_profile`: ≥ 4 GiB, fp16 when
  `has_fp16`). Training's integer GPU ids index `gpu_profiles()` of `gpu_backend()` (`cuda` or
  `xpu`, one per torch build); children see their card through `visible_devices_env()`
  (`CUDA_VISIBLE_DEVICES`, which HIP reads, or `ZE_AFFINITY_MASK`). XPU training: autocast and
  `GradScaler("xpu")`, DDP over `xccl`. CUDA Graphs never on ROCm. pymss has no XPU: separation
  loads on the CPU, moves the net, runs fp32. Decide fp16 with `supports_half(device)`, never
  `startswith("cuda")`. `ComputeInfo.backend` = `torch_backend()` (`CUDA 13.0`, `ROCm 10.0`, `XPU`).
- **HuBERT normalisation:** the real `preprocessor_config.json` has `do_normalize: false`, so
  neither training nor inference normalises (one function, `features/hubert.normalize_input`).
- **Behaviour changes**, all in place: slicing keeps every tail; Live honours the
  speaker; the index is rebuilt when the features' fingerprint changes and loaded once; pm, rmvpe
  and fcpe everywhere for conversion; formant shift offline (the realtime method: pitch at
  `key − formant`, generate `ceil(n·2^(f/12))` frames with NSF pitch scaled, resample back);
  any path; inference off the audio callback; block/crossfade/context changes rebuffer; the
  training batch size is per GPU (the original passed `batch_size × n_gpus` to a sampler that
  gives each rank batches of that size).
- **Protect and unvoiced frames** (`VoiceParams.unvoiced`): the 2026 package interpolates unvoiced
  F0 before the protect mask (`pitchf < 1`) is built, so protect never applied (and its realtime
  path has none). `"original"` keeps that arithmetic (engine default, the golden runs);
  `"protect"` (the core's default) builds the mask from the detector's voicing (`offline_f0`
  returns it) and still feeds the filled F0; `"zero"` keeps 0 Hz, as classic RVC and Applio
  (voices trained on zeros). `StreamEngine` blends the same way from `cache_voiced`. Protect only
  holds back the index: with no index it changes nothing.
- **Pitch methods** (`engine/f0/METHODS`): pm, rmvpe, fcpe (the original's) plus `crepe`/`crepe-tiny`
  (`crepe_model.py`, torchcrepe ported bit-identically without torchaudio; weights are the
  `crepe`/`crepe-tiny` assets in rvc-model) and `swift` (`swift_model.py`, SwiftF0 0.3.0's ONNX
  graph in torch with onnxruntime's own trig tables; weights are package data, no asset) with
  Applio's subharmonic repair. `core/params.f0_assets` maps a method to its assets. Live calls the
  new ones through `compute()` on numpy.
- **Pitch extras** (offline, all off by default): `f0_high_register` (`engine/f0/high_register.py`,
  Applio's corrector: a second RMVPE pass on audio upsampled ×2 fixes octave errors above
  ~1040 Hz; never touches frames whose guide is below 460 Hz); RMVPE runs in overlapping
  320 s chunks past `CHUNK_FRAMES` (shorter audio, the golden clips included, in one go).
- **Untrusted files:** every `.pth` loads with `weights_only=True` (`checkpoint.torch_load`, the
  training loop); an imported separation checkpoint must load that way too
  (`separate/runner.ensure_plain_checkpoint`, demucs class names allowed as pymss's stand-ins)
  before pymss, which reads demucs/apollo with `weights_only=False`, sees it.
- Offline output is up to two 10 ms frames shorter than the input — the original's flooring.
- **Late blocks** (`AudioSession`): a block slower than the block length used to leave its backlog
  in the output buffer for good (duplex has no drift correction; split's settle target adopted it).
  The processing thread now converts only the newest input block when a whole block more waits,
  trims the output ring to its prefill when its after-write level stays above prefill + ½ block
  (duplex) or + 1½ blocks (split) for 3 writes (`trimmed`, `skipped`), and the drift corrector
  re-settles after an underrun or a trim. `prewarm()` runs `_process` itself (gate off/on, denoise,
  loudness match) so the first block is not late; the worker prewarms again before reopening after
  a rebuffer or a new rate. A hot voice swap cannot be prewarmed beside the running block (a CUDA
  Graph capture on another thread breaks its GPU work), so its first block is left out of the
  timings (`skip_timing`). Load is p95 of 50 blocks, the median until 10; the UI shows ">100%".
- **Channel counts:** PortAudio reports only a device's widest count, and some drivers refuse
  narrower streams (WDM-KS pins, stereo-only cards: `PaErrorCode -9998` on the mono input).
  `devices.probe_device` records `channel_counts` when a device refuses some, `Endpoint.open_channels`
  opens the next accepted count up, and rates are probed at it. WASAPI shared streams use
  `auto_convert`. `-9998` is the `channels` reason, never `format` (whose fix is "Use 48 kHz").
- **Recording an output device** (`engine/audio_io/capture.py`): PortAudio cannot open an output as
  an input (-9998, it has no input channels), and the 19.7.0 DLL sounddevice bundles predates
  PortAudio's WASAPI `[Loopback]` devices. `SoundcardLoopback` (SoundCard ≥ 0.4.3, Windows and
  Linux only; `soundcard` only under `engine/audio_io/`) records WASAPI loopback / Pulse monitors
  on a `CaptureStream` thread; PortAudio's own loopback devices win when present. Listed only with
  `live.show_all_devices` (keys `loopback:<output>`, negative indexes, `loopback_source`,
  `loopback_of`); a loopback of the output or monitor is refused as `feedback`. That setting no
  longer cross-lists devices (an input can never be played into).
- **Live extras** (all off/neutral by default): `StreamParams.phase_vocoder` (`audio/sola.phase_vocoder`,
  RVC's later `gui_v1` crossfade as Applio uses it), `denoise_strength` (TorchGate's
  `prop_decrease`, the original's 0.9), `LiveDevices.input_gain_db`; gains and the monitor source
  change in place (`SessionConfig.same_endpoints`). Recording: `AudioSession.tap` feeds
  `audio_io/recorder.Recorder` (its own writer thread), the worker sends `Recorded`, the core adds
  an output of kind `recording`.
- `StreamEngine` calls `remove_weight_norm()` on its voice's net, as the original does; it runs
  in the live worker's own runtime, never the server's.

## 6. Core services

- **Jobs** (`core/jobs/service.py`): a kind registers a factory `request dict → JobSpec(title, run,
  resources, steps, priority)`; the request is stored, so **Retry** works after a restart.
  Resources: `gpu` (capacity `compute.gpu_jobs`), `cpu` (half the cores), `io` (4). GPU jobs also
  need the GPU lease (holder `jobs`, shared by concurrent GPU jobs); Live takes it (`live`), and a
  waiting job gets `can_run_anyway`. Runners: in-process on a thread with a `CancelToken`, or
  `ctx.run_worker(module, request)` (subprocess, JSON lines, cancel = process-tree kill). A job
  still active at start-up becomes `interrupted` unless another live server owns the data dir.
  Logs are `jobs/<id>.log`, read by byte offset. Kinds: convert separate train index download
  fetch merge extract export. Downloads call `ctx.transfer(done, total)`: `Job.transfer` carries the bytes,
  the speed over a sliding 5 s window (reset when a file restarts) and the ETA; cleared at the end.
- **GPU memory** (`core/compute/service.py`, after ComfyUI's `comfy/model_management.py`): the
  server's runtime plus `MemoryHolder`s — Jobs and Live register one each (`busy`, `cached`,
  `release`). **Free GPU memory** (`POST /compute/release`) is allowed whenever nothing is busy (no
  job queued or running, Live not active) — not only when the server's own cache lists models,
  since the idle live worker keeps HuBERT, the voice and F0 in its own process
  (`P.State.cached`, `P.ReleaseMemory`). While busy it is refused (409, `detail.busy`), and
  `ComputeUsage.busy`/`can_release` drive the top-bar menu; job and Live transitions emit
  `compute_changed`. Unloading = drop the caches, `gc.collect()`, then `synchronize`,
  `empty_cache`, `ipc_collect` (`engine/runtime.soft_empty_cache`). **Out of memory**
  (`engine/runtime.is_oom`: `torch.OutOfMemoryError`, `AcceleratorError` code 2, MPS/DirectML
  messages) becomes `compute_unavailable` with `detail.reason = "out_of_memory"` — in-process,
  from a worker's error event, or from the live worker — and the core then unloads every model not
  in use by a running session (`release_after_oom`), as ComfyUI does. The live worker unloads its
  own models first. A job that failed this way is not retried automatically.
- **Assets** (`core/assets/catalog.json`): hubert, rmvpe, rmvpe-onnx, fcpe (torchfcpe's bundled
  `fcpe_c_v001.pt`, only in rvc-model under `fcpe/`; required when FCPE is chosen), `pretrained-{v1,v2}-{32k,40k,48k}`
  (G/D with and without pitch; finer than the plan's two ids so training at one rate downloads 4
  files), `separation-<model>`. Sizes and SHA-256 from the Hugging Face listing, re-hashed
  against the full downloaded repository (revision `e6d0c1a`). Downloads: `.part` files, `Range`
  resume, 4 retries, checksum; verified results cached in `assets-verified.json`. Models found
  only inside the official distribution packages are summarised in `site/content/docs/development-assets.mdx`.
  `DELETE /assets/{id}` removes an asset's files (refused while it downloads); an official base
  model's delete removes only its own G and D (`delete_files`), leaving the asset `partial`.
- **Two download repositories** (`catalog.json` `repositories`; setting `downloads.repository`):
  `rvc-model` = [licyk/rvc-model](https://huggingface.co/licyk/rvc-model) (default; every file
  rvc-next uses, by category — `hubert/`, `rmvpe/`, `pretrained/v1|v2/`, `separation/` — plus the
  demo voices under `voices/<name>/` and a `manifest.json` with sizes, SHA-256 and upstream paths)
  and `official` = lj1995/VoiceConversionWebUI. Each catalog file lists its path per repository;
  a file the chosen repository lacks comes from another (the demo voices exist only in
  rvc-model). `downloads.source` is the endpoint (huggingface.co, hf-mirror.com or custom) both
  are reached through. Uploads to rvc-model keep `manifest.json` in step, and `catalog.json` is
  generated from it.
- **Demo voices** (`catalog.json` `voices`): `assets.voice_catalog()` / `GET /models/catalog`,
  `download_voices()` / `POST /models/catalog/download`, `rvc-next model catalog|download`, and
  **Models › Voices › Demo voices**. A download is a `download` job (request `{"voices": [...]}`)
  that fetches the `.pth` and `.index`, checks their SHA-256 and adds them to the library with
  `catalog_id` in `voice.json` (DB migration 2), which marks the entry as installed.
- **Voice library** (`core/models/service.py`): folders `models/voices/<slug>/model.pth`, `model.index` or
  `spkid<N>.index`, `voice.json` (with `index_sources`, the names the indexes were imported
  under). A library still at `models/<slug>/` is moved once on rescan (`_migrate_layout`). **Speakers:** a voice is multi-speaker only when its `.pth`
  carries `speaker_info` (or names were given); every trained voice keeps 109 embedding rows, so
  the row count alone would make every voice multi-speaker. `speaker_slots` still reports the rows.
  Legacy roots are read in place; index pairing uses the original's scoring once, as a suggestion,
  stored in `legacy-voices.json` with the user's edits. Temporary voices (`tmp-…`, a `.pth` path on
  the CLI, a training try-out) are hidden from lists.
- **Export and upload:** `models.export_archive` streams a voice as a zip (`core/files.zip_stream`,
  shared with the outputs zip): `<name>/<name>.pth`, rewritten into a temporary copy only to carry
  speaker names, and `<name>.index`/`<name>_spkid<N>.index`, so an import pairs them again.
  `training.add_dataset_file` takes raw uploads (audio, or a zip, flattened; in multi-speaker mode
  its top-level `Name_ID_Repeat` folders become the table) into `<experiment>/dataset_upload/`
  and points the dataset there.
- **Model layout:** `models/voices|indexes|base|separation|attached-indexes/`. Only voices are
  in the database; base models, separation models and unassigned indexes are folders with a JSON
  sidecar, scanned on every listing. `attached-indexes/<voice>/` holds indexes copied for legacy
  voices, whose own folders are never written.
- **Model imports** (`core/models/imports.py`; `site/content/docs/development-core.mdx`): every import —
  web drop zone, `PUT /models/imports/{sid}/files`, `POST …/paths`, `rvc-next model import`, and the
  older `/models/import` routes — is a session under `data/uploads/imports/<sid>/` (24 h): stage,
  identify by **content** (never by extension or name), plan, commit with the user's decisions.
  Identification: `.index` width → version (256 v1, 768 v2; `ntotal == 0` = a `trained_` index,
  `engine/index/inspect.py`); `.pth` small model → voice, G → generator (version, rate and pitch from
  `enc_p.emb_phone`, `enc_q.pre`, the upsample kernels, `dec.m_source`), D → discriminator (7 or 9
  sub-discriminators); `.yaml` via pymss_core's `detect_model_type`; any other weights file →
  separation checkpoint. **Pairing** (`core/models/pairing.py`, pure): the version check is hard;
  then the names (experiment name 1.0, contains 0.8, token Jaccard × 0.7) and the folder (+0.2) score
  each compatible voice. An index pairs automatically only when it is the one compatible voice in
  the import, or scores ≥ 0.8 with a 0.25 lead; otherwise the plan is not `confident` and the UI
  opens `ImportReviewDialog` (the CLI prints the plan and exits 3; `--pair`, `--keep-unassigned`,
  `--extract` decide). Library voices are candidates only when no staged voice fits. An index left
  unassigned goes to the inbox (`core/models/inbox.py`, `GET /models/indexes`, assign/delete).
  Every attach checks the index against the voice (`empty_index`, `version_mismatch`).
- **Links** (`core/net/fetch.py`, `imports.add_urls`, `POST /models/imports/{sid}/urls`): a `fetch`
  job resolves Hugging Face files/repos/folders (through `downloads.source`), Google Drive files
  (`drive.usercontent.google.com/download?…&confirm=t`) and plain links, downloads them (≤ 8 GiB,
  a web page is refused) and stages them like uploads. Every hop is checked: http(s) only, and the
  host's addresses and the connected peer must be public (`allow_private` only from the CLI; the
  API never, since a reverse proxy makes remote browsers look local; peer check skipped behind an
  environment proxy).
- **Base models** (`core/models/base.py`): official ones from the assets plus imported G (+ D)
  pairs; `GET /models/base?sample_rate&version&pitch_guidance` lists only those that fit, and
  `FitSettings.base_model` (`train run --base`) picks one; `check_fits` rejects a mismatch.
- **Applio voices** (`engine/models/checkpoint.py`): Applio's HiFi-GAN voices are RVC v2 voices
  with extra keys. A `vocoder` other than HiFi-GAN (or MRF/RefineGAN decoder keys), or an
  `embedder_model` other than `contentvec`, is refused as `ModelFormat` (`detail.reason`
  `vocoder`/`embedder`; imports stage it `unsupported`), as are their G and v3 (MRD, kernels 3×9)
  D files. `speakers_id > 1` without names becomes numbered speakers; `author epoch step
  creation_date dataset_length` become `VoiceModel.provenance` (About).
- **Separation models** (`core/separation/library.py`): checkpoint + YAML, each one a one-step preset
  (`source: "imported"`, id `user-<slug>`); the import runs the separate worker's `check` action to
  load the model once before accepting it (`separation_load`).
- **Presets:** one default per voice, created on import from `convert.default_params`.
- **Audio** (`core/audio/service.py`): uploads are raw `PUT` bodies spooled to a file, then probed
  with PyAV; server paths are checked against `paths.browse_roots` **in the API** (the CLI is
  trusted). Left empty, the roots are home, the legacy roots, the data directory and every drive
  (Windows, `core/files.windows_drives`) or mounted volume (`/Volumes`, `/media/<user>`, `/mnt`), never resolved (resolving a drive opens it: a
  disconnected network drive would stall every `GET /settings`); peaks cached under `cache/peaks/`; outputs carry provenance; zip streamed.
- **Conversion:** one job per request; separate first (one worker for all inputs), convert each
  primary stem, then remix over the `instrumental` stem; names `<stem>.<voice slug>.<fmt>`, never
  overwriting (the CLI's `--overwrite` with `-o` excepted). Preview = first N s, priority 10.
- **Separation:** presets from `core/separation/presets.json` (vocals, vocals-aggressive,
  dereverb, dereverb-aggressive, karaoke; chains vocals-clean, lead-clean). Weights in
  `assets/pymss_weights/` (the original's layout). Exit 75 → rerun in fp32 (DirectML).
- **Training options** (`FitSettings` → `loop.TrainRequest`): `precision` auto|fp32|bf16 (bf16:
  autocast without a scaler, fp32 where unsupported), `tf32`, `checkpointing`
  (`loop.enable_checkpointing` wraps `dec.ups`, `dec.resblocks` and each sub-discriminator's
  `forward`, so the ported nets stay as they are), `fresh_speakers` (the experiment's rows of
  `emb_g` redrawn at the base model's mean norm, seeded so DDP ranks agree), `previews` (default
  on: the longest slice rendered at each save into `previews/`, registered as outputs of kind
  `sample` with the slice as source; `GET /train/experiments/{name}/samples`). The G checkpoint
  carries the fp16 scaler state; metrics add `grad_g`/`grad_d`; small models carry `epoch`,
  `step`, `creation_date` and `training.author`. Training pitch takes every neural method too.
- **Training:** `experiment.json` per folder; stage fingerprints (`engine/train/fingerprint.py`);
  **Run all** runs what is not done or whose fingerprint changed, and everything after it (fit alone
  when only `epochs` grew); editing the
  dataset or rates marks later stages `stale`. Stages run in `workers.train`; metrics become
  `train_metrics` events. Export extracts a G checkpoint when needed and adds the experiment's
  `added_*.index` files. A folder without `experiment.json` is read as a legacy experiment.
- **Live:** `LiveSupervisor` starts `workers.live` on demand (a `Listener` with a random authkey;
  the request file carries the address) and turns its messages into `live_state`/`live_stats`.
  Devices come from `workers.devices` (a short subprocess, never two at once: callers that ask
  meanwhile share the next list, `LiveService.devices`); selections are resolved (§7.6) before
  every start and change; an output that would fall back to the default needs
  `allow_output_fallback`.
  Start publishes `starting` with `LiveState.stage` before its slow steps (`devices`: the
  enumeration subprocess, skipped when the cached list is under 10 s old, `START_LIST_MAX_AGE`, as
  the Live page re-enumerates every 5 s; `worker`: spawning the worker, ~0.1 s of imports since
  `scipy.signal` loads lazily in `engine/audio/dsp.py`) and rolls back if they fail; the worker then reports `loading` stages (`runtime`:
  torch/transformers import and the device, `voice`, `index`, `hubert`, `pitch`), skipping what is
  already loaded. The UI's Start button shows the stage, and "Starting…" while the request is out. Device loss → `reconnecting`, re-enumeration every 2 s, reopen on
  `exact`/`matched`. The worker's first `stopped` after connecting is ignored until Start lands.
  A voice switch while a session is active is a hot swap: the worker loads the new voice beside the
  old one (HuBERT and F0 stay) and swaps between two blocks. Start and the hot updates run under
  one lock (`LiveService._control`), so a switch and the speaker change it causes cannot interleave;
  the worker answers every `SetVoice` with `State.voice_path` (the voice now converting — the old
  one, with `detail.reason == "voice_load"`, when the new one fails) and the state follows it.
- **Measured latency** (`engine/audio_io/loopback.py`, `POST /live/devices/latency-test`, `live
  latency`, **Measure latency**): the devices worker opens an `AudioSession` with a
  `LoopbackProbe` as its processor, plays four distinct band-limited noise bursts and finds each in
  the input by FFT cross-correlation (found at ≥ 20 dB over the correlation's median floor; they
  must agree within 2 ms). That round trip covers prefill, rings and device buffers; the engine's
  own delay is added, computed (`engine_delay_ms`: crossfade + 10 ms, + min(crossfade, 40 ms) with
  input denoise; `test_engine_delay` measures it on passthrough). Live must be stopped; the meter
  is paused. `LiveState.latency_test` keeps it while input, output and block match. The fake
  backend's `loopback` feeds an output back with one block plus `delay_ms`.

## 7. Web UI

Vue 3 + Vite + TypeScript 6, hash router, Pinia, TanStack Vue Query, `openapi-fetch` over the
generated `src/api/schema.d.ts`, `socket.io-client`, `@material/web` and `@lucide/vue` only inside
`src/ui/`. bun is the package manager (`/root/.bun/bin/bun` here). Theme, `ui/`, motion and the
rule tests come from Hanaikada.

- **The unified-experience contract** is built as shared components and is the review
  checklist for every screen: `VoicePicker` (on `ui/PickerMenu`, the one picker: voices and audio
  devices alike, second line, badges, search from a length, wrapping names), `VoiceParamsPanel` (ranges, defaults and help from
  `paramSpecs.ts`, generated by `typegen` from the schema), `AudioSourceInput`, `JobCard` +
  `JobsSheet`, `ResultsList` with `ABCompare` on one shared player (`stores/player.ts`),
  `AssetGate`, `ErrorNotice` (translated code + Details), `DevicePanel`, `LiveStatsBar`.
  Hand-offs between screens ("Use as input", "Convert this", "Try") go through `stores/handoff.ts`.
- **Data flow:** REST is the source of truth; socket events patch or invalidate the query cache;
  `live_stats`, `train_metrics` and job progress go to Pinia stores; on reconnect the active jobs,
  live state and visible lists are refetched. The Jobs sheet merges recent history from REST.
- **Types:** `typegen` runs openapi-typescript with `--default-non-nullable false` (defaulted
  request fields stay optional) and writes `paramSpecs.ts`; `typegen-check` diffs both. The API uses
  `separate_input_output_schemas=False` and event schemas in serialization mode; FastAPI still
  splits `Dataset`, `FitSettings` and `SpeakerEntry` into `-Input`/`-Output` (see `api/types.ts`).
- A field with a button beside it (Add, Browse, Build) centres the button on the field.
- **Status bars glide.** `LevelMeter` moves by transform with audio-meter ballistics (rise in
  100 ms, fall in 300 ms, the peak tick in 500 ms; `useFalling`). `ProgressBar` draws determinate
  bars itself with an ease-out transition (Material's eases in and out and restarts slow on every
  update, which stutters at the stats rate); indeterminate ones stay `md-linear-progress`.
- **Tabs** (Models, Settings) slide the way they moved: `shared-axis-x` with `--axis-dir` from
  `useAxisDirection`, in a `position: relative; overflow-x: clip` box, as Hanaikada's tabs.
- **Top bar:** always the app name (RVC Next); actions end with Help (a link to the docs site in
  the UI's language: `/docs`, `/en/docs`) and the theme button, which cycles
  light → dark → follow the system (Hanaikada's), its icon showing the current choice.
- **Motion:** MD3 transitions; route changes run inside `document.startViewTransition` when
  available, finishing on `nextTick` (`requestAnimationFrame` is paused inside the update callback,
  which hung navigation); views then render without the out-in `<Transition>`.
  Reduced motion (the system setting or Settings › Appearance) still runs the view transition, as
  a 100 ms crossfade without the scale: never no transition at all, as in Hanaikada.
- **Imports:** every drop zone on Models (Voices, Base models, Separation models) calls
  `useModelImport`, which commits a confident plan at once and otherwise hands it to
  `ImportReviewDialog`; index choices list only the plan's compatible candidates. Unassigned
  indexes show in `IndexInbox` on the Voices tab. Train's settings list the fitting base models.
- **Shell and navigation:** the shell is still between destinations. Only the scroller
  `main.content` (`view-transition-name: app-content`) fades through; the root, the rail and the
  top bar do not animate, and the rounded `.frame` around the scroller stays outside the snapshot.
  Views are kept alive (`KeepAlive`, keyed per destination, per experiment and per voice), and
  `App.vue` keeps each one's scroll offset in memory and restores it before the new view is
  shown. A refresh resets every view; only real preferences (theme, last voice, last preset)
  persist.
- **Width:** pages fill the content area (no page `max-width`, as in Hanaikada). `main.content` is
  the size container `app-content`. Convert, Separate, Live, a voice's page and Models › Tools are
  one column of full-width sections, as Hanaikada's Settings; the controls inside a section wrap in `auto-fill` grids
  (`VoiceParamsPanel`, `StreamParamsPanel`, `DevicePanel`'s Advanced), so a wider page
  shows more controls per row, never longer ones. Pickers are the exception: a device's full name
  is long, so each `DevicePanel` role is a full-width row with its meter or Test below. Their run buttons, active jobs and errors sit
  in the sticky `PageFooter` (`layout.test.ts` holds both rules).
  Dropdowns that list in place (`PickerMenu`) stack under that footer
  (`--app-z-dropdown` < `--app-z-page-footer`) and scroll clear of it when opened (`revealDropdown`).
- **Models:** `ResourceRow` is the one row for anything downloadable (state, progress with
  speed and ETA, Download/Verify/Delete/Open). Assets lists every asset group and the demo voices;
  Base models and Separation models list the official/built-in ones the same way, with the
  imported ones above them.
- `src/test/setup.ts` stubs `attachInternals` (happy-dom lacks it; `@material/web` needs it).
- Device problem messages come from the server in English; the UI translates the reason code and
  shows the message as sent.

## 8. Departures from the plan

| Plan | Built | Why |
| --- | --- | --- |
| `transformers` below 4.50, as the original | `transformers>=5,<6`, tested with 5.18 | Current `huggingface_hub` (1.x) needs transformers 5; HuBERT layer outputs are unchanged and the golden runs are bit-identical |
| Pydantic v1 and v2 (`Record` shim) | v2 only | FastAPI ≥ 0.100 needs v2; no host pins v1 |
| Asset ids `pretrained-v1`, `pretrained-v2` | Per version and rate | 4 files instead of 12 for one training run |
| Separation weights under `assets/separation/` | `assets/pymss_weights/` | The original's layout, so `paths.assets_dir` can point at an RVC install |
| Speaker count from `emb_g` | From `speaker_info`; `speaker_slots` from `emb_g` | Every trained voice has 109 rows |
| Index k-means with scikit-learn | faiss k-means | One dependency fewer; same role |
| Single-speaker slicing of every file under the folder | Top-level audio files | Matches the dataset scan |
| torchaudio's `Resample` | Ported into `engine/audio/sinc_resample.py`, typed | torchaudio ends at 2.11; outputs bit-identical |
| torchfcpe (FCPE pitch) | Ported into `engine/f0/fcpe_model.py`; weights are the `fcpe` asset | It required torchaudio and bundled 43 MB of weights; only the conv-only path RVC runs is kept, bit-identical |
| Import commit as a job, with an `imports_changed` event | Synchronous commit returning `ImportResult`; `models_changed` | Copies are local and fast; the separation load check is the slowest step (seconds) |
| Tables for base models, separation models and the index inbox | Folders with JSON sidecars | Nothing queries them; a folder scan cannot drift from the files |
| `--type` to force a file's kind on the CLI | Not built | Content identification has been exact on every file tried; decisions cover the rest |
