# rvc-next

English | [简体中文](https://github.com/licyk/rvc-next/blob/main/README.zh-CN.md)

Retrieval-based voice conversion, rewritten as one application: **convert** files, convert
**live** through your audio devices, **separate** vocals from music, **train** your own voices,
and manage them all in one **library**, from a web UI or the command line.

rvc-next is a rewrite of
[Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).
The numerical code (the synthesizer, RMVPE, the conversion and streaming maths, the training
loop) is ported so existing voices sound the same; everything around it is new:

- **One voice picker, one parameter panel** everywhere. A voice's default preset makes it sound
  the same in Convert and Live.
- **Every long operation is a job** you can watch, cancel, retry and come back to; nothing
  blocks the screen.
- **Pipelines**: separate, convert and add the accompaniment back in one job.
- **Audio devices** listed once per physical device, input and output on any drivers, an
  optional monitor output, a level meter and a test sound, recovery when a device returns, and
  a loopback test that measures the real latency instead of estimating it.
- **Existing models load unchanged** (`.pth` and `added_*.index`), an existing RVC install can
  be read in place, and new models are written in the same format.

The documentation (tutorials, the command line and developer documentation, in Chinese and
English) is in [site/content/docs/](site/content/docs/index.mdx), part of the project's website in
`site/` (`python scripts/dev.py site-dev` to preview it).
The design record, the conventions and the known gaps are in [AGENTS.md](AGENTS.md).

## Install

Use a fresh virtual environment (or a standalone Python): rvc-next pins its dependencies to
tested ranges and is not meant to share an environment with other applications. Install PyTorch
for your hardware first (2.7.1 or newer; torchaudio is not needed):

```bash
# Windows / Linux, NVIDIA GPU
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
# Windows / Linux, AMD GPU (AMD's ROCm build; the cards appear as cuda:N)
python -m pip install "torch[device-all]==2.13.0+rocm10.0.0" --index-url https://stable.repo.amd.com/rocm/whl-next
# Windows / Linux, Intel GPU (the cards appear as xpu:N)
python -m pip install torch==2.14.0+xpu --index-url https://download.pytorch.org/whl/xpu
# macOS (Apple silicon; Metal is experimental)
python -m pip install torch==2.14.0
```

Without a GPU, use the CPU build (`--index-url https://download.pytorch.org/whl/cpu`). On
Windows, AMD and Intel cards that the ROCm and XPU builds do not support can use DirectML: the
CPU build plus `pip install "rvc-next[directml]"`.

Then, for example with an NVIDIA card:

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install rvc-next
rvc-next assets download          # HuBERT and the pitch models, RMVPE and FCPE (about 400 MB)
rvc-next model download --all     # optional: the official RVC demo voices
rvc-next webui                    # http://127.0.0.1:7868
```

`rvc-next doctor` reports the torch build (CUDA, ROCm or XPU), the GPU rule's choice, audio
device access and the assets. ROCm, XPU, DirectML and Metal have not yet been tried on real
hardware. On Linux, live conversion needs PortAudio (`libportaudio2`).

To reuse the models of an existing RVC install, point `paths.assets_dir` at its `assets/`
folder and add the install as a legacy root (Models › Voices › Original RVC installs, or
`rvc-next model import <RVC folder>` to copy its voices into the library).

## Models

Models download from [licyk/rvc-model](https://huggingface.co/licyk/rvc-model), which holds every
model rvc-next uses, organised by category, together with the official RVC demo voices. The
official repository, [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI),
can be used instead (`rvc-next config set downloads.repository official`, or Settings ›
Downloads); `downloads.source` switches the endpoint to `hf-mirror.com` or a custom one.

Your own models are imported on the Models screen or with `rvc-next model import`: voices
(`.pth`, with their `.index`), training base models (G and D `.pth`) and separation models
(a checkpoint with its `.yaml`), loose or in a `.zip`. Files are recognised by their contents, so
a `.pth` and an `.index` with unrelated names are still paired: the index's width fixes the voice
version it fits, and when more than one voice fits, you choose. An index can also be imported on
its own and assigned to a voice later. Imported base models appear in an experiment's settings,
and imported separation models as presets in Separate.

## Command line

```bash
rvc-next model import voice.zip --name "My voice"
rvc-next convert song.wav --voice "My voice" --pitch 2 --separate vocals --remix -o out/
rvc-next separate track.flac --preset vocals-clean
rvc-next train new alice --dataset ~/datasets/alice && rvc-next train run alice
rvc-next live devices && rvc-next live run --voice "My voice"
rvc-next --help
```

Every listing takes `--json`, which prints the same shape the API returns.

## Data

Settings, the database, the voice library, experiments and outputs live in
`$XDG_DATA_HOME/rvc-next`, `~/Library/Application Support/rvc-next` or `%APPDATA%\rvc-next`
(`RVC_NEXT_DATA_DIR` moves it). Any setting can be overridden with
`RVC_NEXT_<GROUP>__<FIELD>`, for example `RVC_NEXT_COMPUTE__DEVICE=cpu`.

## Licence

GPL-3.0. Ported code from RVC (MIT) and other projects is listed in [NOTICE](NOTICE).
