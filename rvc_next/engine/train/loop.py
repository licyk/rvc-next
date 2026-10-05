"""Fit stage: train the synthesizer and discriminator (the original ``train/train.py`` and ``utils.py``).

Changes from the original: no ``os._exit``, no ``argparse`` or global state; progress, logs,
metrics and saved files go through a sink (callbacks in this process, a queue from DDP rank 0);
TensorBoard is optional; small models go to ``<experiment>/weights``; a resumed run continues after
the saved epoch instead of repeating it. One GPU or the CPU trains in this process; several GPUs
train with DDP over gloo, one spawned process per card.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from random import randint, shuffle
from typing import Any

from rvc_next.engine.cancel import CancelToken
from rvc_next.engine.errors import Cancelled, EngineError
from rvc_next.engine.train.parallel import Log, Progress

logger = logging.getLogger(__name__)

LATEST_ONLY_STEP = 2333333
METRICS_FILE = "metrics.jsonl"


@dataclass(frozen=True)
class TrainRequest:
    exp_dir: str
    sample_rate: str = "40k"
    version: str = "v2"
    pitch_guidance: bool = True
    epochs: int = 20
    save_every: int = 5
    batch_size: int = 4
    pretrained_g: str = ""
    pretrained_d: str = ""
    gpus: tuple[int, ...] = ()
    """CUDA cards; empty trains on the CPU. More than one trains with DDP."""
    cache_in_gpu: bool = False
    save_small_every: bool = False
    """Also write a small model into ``weights/`` at every save."""
    save_latest_only: bool = False
    """Keep one G/D pair (``G_2333333.pth``), as the original's option."""
    name: str = ""
    """Small-model file name; default the experiment folder's name."""
    speaker_id: int = 0
    """Speaker of a single-speaker dataset."""
    multi_speaker: bool = False
    config_override: dict[str, Any] = field(default_factory=dict)
    """Deep-merged into ``config.json`` (tests use a tiny model)."""
    log_interval: int | None = None
    """Steps between metric lines; None keeps the config's (200)."""
    num_workers: int = 4
    """DataLoader worker processes; 0 loads in the training process."""
    tensorboard: bool = False
    seed: int | None = None
    """File-list shuffle seed; None is random."""
    ddp_cpu_processes: int = 0
    """Testing only: run DDP over gloo with this many CPU processes."""


class Sink:
    """Where a training process reports. ``Sink.direct`` calls back; a queue sink crosses processes."""

    def __init__(self, put: Callable[[tuple], None]) -> None:
        self.put = put

    def progress(self, value: float | None, step: str | None) -> None:
        self.put(("progress", value, step))

    def log(self, message: str) -> None:
        self.put(("log", message))

    def metric(self, item: dict[str, Any]) -> None:
        self.put(("metric", item))

    def output(self, path: str, kind: str) -> None:
        self.put(("output", path, kind))


def _dispatch(msg: tuple, progress: Progress | None, log: Log | None, metrics: Callable[[dict], None] | None, output: Callable[[str, str], None] | None) -> None:
    kind = msg[0]
    if kind == "progress" and progress:
        progress(msg[1], msg[2])
    elif kind == "log" and log:
        log(msg[1])
    elif kind == "metric" and metrics:
        metrics(msg[1])
    elif kind == "output" and output:
        output(msg[1], msg[2])


def training_is_half(gpus: tuple[int, ...]) -> bool:
    """fp16 (AMP) only when every chosen card qualifies, as the original's ``get_training_dtype``."""
    if not gpus:
        return False
    import torch

    if not torch.cuda.is_available():
        return False
    from rvc_next.engine.runtime import cuda_profile

    profiles = [cuda_profile(i) for i in range(torch.cuda.device_count())]
    return all(profiles[i].eligible and profiles[i].fp16 for i in range(len(gpus)) if i < len(profiles))


def latest_checkpoint_path(dir_path: str, regex: str = "G_*.pth") -> str:
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f))) or 0))
    return f_list[-1]


def load_checkpoint(checkpoint_path: str, model: Any, optimizer: Any = None, load_opt: int = 1) -> tuple[Any, Any, float, int]:
    import torch

    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_state_dict = checkpoint_dict["model"]
    target = model.module if hasattr(model, "module") else model
    state_dict = target.state_dict()
    new_state_dict = {}
    embedding_resized = False
    for k, v in state_dict.items():
        if k not in saved_state_dict:
            logger.info("%s is not in the checkpoint", k)
            new_state_dict[k] = v
            continue
        saved = saved_state_dict[k]
        if saved.shape != v.shape:
            if k == "emb_g.weight" and saved.dim() == v.dim() and saved.shape[1:] == v.shape[1:]:
                grown = v.clone()
                rows = min(saved.shape[0], v.shape[0])
                grown[:rows].copy_(saved[:rows])
                new_state_dict[k] = grown
                embedding_resized = True
                continue
            logger.warning("shape-%s-mismatch|need-%s|get-%s", k, v.shape, saved.shape)
            new_state_dict[k] = v
            continue
        new_state_dict[k] = saved
    target.load_state_dict(new_state_dict, strict=False)
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt == 1 and not embedding_resized:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model: Any, optimizer: Any, learning_rate: float, iteration: int, checkpoint_path: str) -> None:
    import torch

    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    tmp = checkpoint_path + ".tmp"
    torch.save({"model": state_dict, "iteration": iteration, "optimizer": optimizer.state_dict(), "learning_rate": learning_rate}, tmp)
    os.replace(tmp, checkpoint_path)


def load_pretrained_generator(model: Any, path: str) -> Any:
    """Load a base G, growing the speaker embedding when the experiment has more speakers."""
    import torch

    target = model.module if hasattr(model, "module") else model
    saved_state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    current_state = target.state_dict()
    key = "emb_g.weight"
    if key in saved_state and key in current_state and saved_state[key].shape != current_state[key].shape:
        saved, current = saved_state[key], current_state[key]
        if saved.dim() == current.dim() and saved.shape[1:] == current.shape[1:]:
            expanded = current.clone()
            rows = min(saved.shape[0], current.shape[0])
            expanded[:rows].copy_(saved[:rows])
            saved_state[key] = expanded
    return target.load_state_dict(saved_state)


def prepare(request: TrainRequest) -> dict[str, Any]:
    """Write ``filelist.txt`` and ``config.json``; return the hyperparameter dict for the run."""
    from rvc_next.engine.train.filelist import prepare_config, write_filelist

    exp = Path(request.exp_dir)
    listing = write_filelist(exp, request.sample_rate, request.version, request.pitch_guidance, request.speaker_id, request.multi_speaker, request.seed)
    config = prepare_config(exp, request.sample_rate, request.version, listing["speakers"] if request.multi_speaker else None, request.config_override)
    hps = dict(config)
    hps.update(
        {
            "model_dir": str(exp),
            "experiment_dir": str(exp),
            "name": request.name or exp.name,
            "total_epoch": request.epochs,
            "save_every_epoch": request.save_every,
            "pretrainG": request.pretrained_g,
            "pretrainD": request.pretrained_d,
            "version": request.version,
            "sample_rate": request.sample_rate,
            "if_f0": 1 if request.pitch_guidance else 0,
            "if_latest": 1 if request.save_latest_only else 0,
            "save_every_weights": "1" if request.save_small_every else "0",
            "if_cache_data_in_gpu": 1 if request.cache_in_gpu else 0,
            "num_workers": request.num_workers,
            "tensorboard": request.tensorboard,
            "lines": listing["lines"],
        }
    )
    hps["train"] = dict(hps["train"])
    hps["train"]["batch_size"] = request.batch_size
    if request.log_interval:
        hps["train"]["log_interval"] = request.log_interval
    hps["data"] = dict(hps["data"])
    hps["data"]["training_files"] = str(exp / "filelist.txt")
    return hps


def run(
    request: TrainRequest,
    progress: Progress | None = None,
    log: Log | None = None,
    cancel: CancelToken | None = None,
    metrics: Callable[[dict[str, Any]], None] | None = None,
    output: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Train to ``request.epochs``; resume from the newest G/D pair when there is one."""
    hps = prepare(request)
    if log:
        log(
            f"Training {hps['name']}: {hps['lines']} file-list lines, batch {request.batch_size}, {request.epochs} epochs, {'GPUs ' + ','.join(map(str, request.gpus)) if request.gpus else 'CPU'}"
        )
    n_procs = len(request.gpus) if len(request.gpus) > 1 else (request.ddp_cpu_processes if request.ddp_cpu_processes > 1 else 0)
    if not n_procs:
        sink = Sink(lambda msg: _dispatch(msg, progress, log, metrics, output))
        device = f"cuda:{request.gpus[0]}" if request.gpus else "cpu"
        return _train(0, 1, hps, sink, cancel, device, False, training_is_half(request.gpus))

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    if request.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, request.gpus))
    half = training_is_half(request.gpus)
    procs = [ctx.Process(target=_ddp_entry, args=(rank, n_procs, hps, q, bool(request.gpus), half), daemon=False) for rank in range(n_procs)]
    for p in procs:
        p.start()
    result: dict[str, Any] = {}
    error: str | None = None
    import queue as queue_mod

    try:
        while any(p.is_alive() for p in procs) or not q.empty():
            if cancel is not None and cancel.cancelled:
                raise Cancelled("Cancelled")
            try:
                msg = q.get(timeout=0.2)
            except queue_mod.Empty:
                continue
            if msg[0] == "result":
                result = msg[1]
            elif msg[0] == "error":
                error = msg[1]
            else:
                _dispatch(msg, progress, log, metrics, output)
    finally:
        for p in procs:
            if p.is_alive() and (error or (cancel is not None and cancel.cancelled)):
                p.terminate()
            p.join(timeout=60)
    if error:
        raise EngineError(error)
    if any(p.exitcode for p in procs):
        raise EngineError(f"A training process exited with code {[p.exitcode for p in procs]}")
    return result


def _ddp_entry(rank: int, n_procs: int, hps: dict[str, Any], q: Any, cuda: bool, half: bool) -> None:
    import traceback

    sink = Sink(q.put) if rank == 0 else Sink(lambda msg: None)
    try:
        result = _train(rank, n_procs, hps, sink, None, f"cuda:{rank}" if cuda else "cpu", True, half)
        if rank == 0:
            q.put(("result", result))
    except BaseException as e:
        q.put(("error", f"rank {rank}: {type(e).__name__}: {e}\n{traceback.format_exc()}"))
        raise


def _train(rank: int, n_procs: int, hps_dict: dict[str, Any], sink: Sink, cancel: CancelToken | None, device: str, use_ddp: bool, is_half: bool) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader

    from rvc_next.engine.models import commons
    from rvc_next.engine.models.checkpoint import save_small_from_state, small_config_from_hparams
    from rvc_next.engine.models.synthesizer import (
        MultiPeriodDiscriminator,
        MultiPeriodDiscriminatorV2,
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )
    from rvc_next.engine.train.dataset import DistributedBucketSampler, TextAudioCollate, TextAudioCollateMultiNSFsid, TextAudioLoader, TextAudioLoaderMultiNSFsid
    from rvc_next.engine.train.hparams import HParams
    from rvc_next.engine.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
    from rvc_next.engine.train.mel import mel_spectrogram_torch, spec_to_mel_torch

    hps = HParams(**hps_dict)
    cuda = device.startswith("cuda")
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    if use_ddp:
        dist.init_process_group(backend="gloo", init_method="env://?use_libuv=False", world_size=n_procs, rank=rank)
    torch.manual_seed(hps.train.seed)
    if cuda:
        torch.cuda.set_device(torch.device(device))

    writer = None
    if rank == 0 and hps.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=hps.model_dir)
        except Exception as e:
            sink.log(f"TensorBoard is not available: {e}")

    f0 = hps.if_f0 == 1
    train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data) if f0 else TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset, hps.train.batch_size * n_procs, [100, 200, 300, 400, 500, 600, 700, 800, 900], num_replicas=n_procs, rank=rank, shuffle=True
    )
    collate_fn = TextAudioCollateMultiNSFsid() if f0 else TextAudioCollate()
    workers = int(hps.num_workers)
    loader_kwargs: dict[str, Any] = {"persistent_workers": True, "prefetch_factor": 8} if workers > 0 else {}
    train_loader = DataLoader(train_dataset, num_workers=workers, shuffle=False, pin_memory=cuda, collate_fn=collate_fn, batch_sampler=train_sampler, **loader_kwargs)
    if len(train_loader) == 0:
        raise EngineError("No training batches: the slices are too short or too few for the batch size")

    if hps.version == "v1":
        model_f0, model_nof0, mpd = SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, MultiPeriodDiscriminator
    else:
        model_f0, model_nof0, mpd = SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono, MultiPeriodDiscriminatorV2
    model_kwargs = {k: v for k, v in hps.model.items()}
    if f0:
        net_g = model_f0(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **model_kwargs, is_half=is_half, sr=hps.sample_rate)
    else:
        net_g = model_nof0(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **model_kwargs, is_half=is_half)
    net_d = mpd(hps.model.use_spectral_norm)
    if cuda:
        net_g = net_g.to(device)
        net_d = net_d.to(device)
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    if use_ddp:
        if cuda:
            net_g = DDP(net_g, device_ids=[torch.device(device).index])
            net_d = DDP(net_d, device_ids=[torch.device(device).index])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)

    last_epoch = 0
    try:
        _, _, _, last_epoch = load_checkpoint(latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        _, _, _, last_epoch = load_checkpoint(latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        if rank == 0:
            sink.log(f"Resumed from the checkpoint of epoch {last_epoch}")
    except Exception:
        last_epoch = 0
        if hps.pretrainG:
            if rank == 0:
                sink.log(f"Base generator: {hps.pretrainG}")
            load_pretrained_generator(net_g, hps.pretrainG)
        if hps.pretrainD:
            if rank == 0:
                sink.log(f"Base discriminator: {hps.pretrainD}")
            target: Any = net_d.module if hasattr(net_d, "module") else net_d
            target.load_state_dict(torch.load(hps.pretrainD, map_location="cpu", weights_only=False)["model"])
    for opt in (optim_g, optim_d):
        for group in opt.param_groups:
            group.setdefault("initial_lr", hps.train.learning_rate)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=last_epoch - 1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=last_epoch - 1)
    scaler = torch.amp.GradScaler("cuda", enabled=is_half)

    steps_per_epoch = len(train_loader)
    global_step = last_epoch * steps_per_epoch
    start_epoch = last_epoch + 1
    total = int(hps.total_epoch)
    cache: list = []
    saved: list[str] = []
    final_path = ""
    metrics_path = os.path.join(hps.model_dir, METRICS_FILE)
    small_dir = os.path.join(hps.model_dir, "weights")

    def emit_metric(epoch: int, values: dict[str, float], lr: float, kind: str) -> None:
        item = {"epoch": epoch, "step": global_step, "losses": values, "lr": lr, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")
        sink.metric(item)

    def state_dict_g() -> dict[str, Any]:
        module: Any = net_g.module if hasattr(net_g, "module") else net_g
        return module.state_dict()

    def save_small(path: str, epoch: int) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_small_from_state(
            state_dict_g(),
            small_config_from_hparams(hps),
            path,
            sr=hps.sample_rate,
            pitch_guidance=f0,
            version=hps.version,
            info=f"{epoch}epoch",
            speaker_info=getattr(hps, "speaker_info", None),
        )
        sink.output(path, "small")

    if start_epoch > total and rank == 0:
        sink.log(f"Already trained to epoch {last_epoch}")
    for epoch in range(start_epoch, total + 1):
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()
        if hps.if_cache_data_in_gpu == 1 and cuda:
            if not cache:
                for batch_idx, info in enumerate(train_loader):
                    cache.append((batch_idx, tuple(t.to(device, non_blocking=True) for t in info)))
            else:
                shuffle(cache)
            data_iterator: Any = cache
        else:
            data_iterator = enumerate(train_loader)
        sums: dict[str, float] = {}
        count = 0
        lr = optim_g.param_groups[0]["lr"]
        for batch_idx, info in data_iterator:
            if cancel is not None:
                cancel.check()
            if not (hps.if_cache_data_in_gpu == 1 and cuda) and cuda:
                info = tuple(t.to(device, non_blocking=True) for t in info)
            if f0:
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info
            else:
                phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
            with torch.autocast("cuda", enabled=is_half):
                if f0:
                    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                else:
                    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
                mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                with torch.autocast("cuda", enabled=False):
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )
                if is_half:
                    y_hat_mel = y_hat_mel.half()
                wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                with torch.autocast("cuda", enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with torch.autocast("cuda", enabled=is_half):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with torch.autocast("cuda", enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if rank == 0:
                lr = optim_g.param_groups[0]["lr"]
                # Clamped for display, as the original does for TensorBoard.
                values = {
                    "g_total": float(loss_gen_all),
                    "d_total": float(loss_disc),
                    "mel": min(float(loss_mel), 75.0),
                    "kl": min(float(loss_kl), 9.0),
                    "fm": float(loss_fm),
                    "gen": float(loss_gen),
                }
                for k, v in values.items():
                    sums[k] = sums.get(k, 0.0) + v
                count += 1
                if global_step % hps.train.log_interval == 0:
                    emit_metric(epoch, values, lr, "step")
                    if writer is not None:
                        for k, v in values.items():
                            writer.add_scalar(f"loss/{k}", v, global_step)
                        writer.add_scalar("learning_rate", lr, global_step)
                sink.progress(((epoch - start_epoch) + (batch_idx + 1) / steps_per_epoch) / max(1, total - start_epoch + 1), f"epoch {epoch}/{total}")
            global_step += 1

        if rank == 0:
            if count:
                emit_metric(epoch, {k: v / count for k, v in sums.items()}, lr, "epoch")
            sink.log(f"Epoch {epoch}/{total} done, step {global_step}")
            if epoch % hps.save_every_epoch == 0:
                step_name = LATEST_ONLY_STEP if hps.if_latest == 1 else global_step
                g_path = os.path.join(hps.model_dir, f"G_{step_name}.pth")
                d_path = os.path.join(hps.model_dir, f"D_{step_name}.pth")
                save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_path)
                save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_path)
                sink.output(g_path, "G")
                sink.output(d_path, "D")
                saved += [g_path, d_path]
                if hps.save_every_weights == "1":
                    path = os.path.join(small_dir, f"{hps.name}_e{epoch}_s{global_step}.pth")
                    save_small(path, epoch)
                    saved.append(path)
            if epoch >= total:
                # The last epoch is always kept, so a finished run can resume and export.
                if epoch % hps.save_every_epoch != 0:
                    step_name = LATEST_ONLY_STEP if hps.if_latest == 1 else global_step
                    for net, opt, prefix in ((net_g, optim_g, "G"), (net_d, optim_d, "D")):
                        path = os.path.join(hps.model_dir, f"{prefix}_{step_name}.pth")
                        save_checkpoint(net, opt, hps.train.learning_rate, epoch, path)
                        sink.output(path, prefix)
                        saved.append(path)
                final_path = os.path.join(small_dir, f"{hps.name}.pth")
                save_small(final_path, epoch)
        scheduler_g.step()
        scheduler_d.step()

    if writer is not None:
        writer.close()
    if use_ddp:
        dist.destroy_process_group()
    if not final_path and os.path.isfile(os.path.join(small_dir, f"{hps.name}.pth")):
        final_path = os.path.join(small_dir, f"{hps.name}.pth")
    return {"epochs": total, "start_epoch": start_epoch, "global_step": global_step, "steps_per_epoch": steps_per_epoch, "final_model": final_path, "saved": saved}
