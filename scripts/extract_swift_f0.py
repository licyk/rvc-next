"""Extract SwiftF0's weights from the installed ``swift-f0`` package into ``rvc_next/engine/f0/data/swift_f0.pt``.

``rvc_next.engine.f0.swift_model`` runs SwiftF0's ``model.onnx`` as torch code. This script reads
that graph with the ``onnx`` package and writes the tensors the torch module needs, as a plain dict
loadable with ``torch.load(..., weights_only=True)``:

- the 18 convolutions, named after the module (``stem``, ``blocks.N.conv1/2``, ``pitch_conv``,
  ``pitch_logits``, ``voicing_conv``, ``voicing_hidden``, ``voicing_out``), taken in graph order
  and checked against the module's kernel, padding and dilation;
- per resolution (n_fft 1024, 2048, 512, the graph's channel order), the analysis window, the
  sparse projection the graph builds with ScatterElements (flat indices int32, values, rows), and
  the cos/sin tables of its three DFT stages;
- the 95 pitch-bin frequencies of the decoder.

The trig tables are what onnxruntime's Cos and Sin give for the graph's float32 angles
``float32(m) · float32(2π/N)``, ``m < N``: torch's cos and sin differ from them by an ulp in about a
third of the entries, and that is enough to move the log of near-empty spectrogram bins (pure tones)
by whole units. They are computed with a two-node ONNX model and checked, entry for entry, against
the Cos/Sin outputs of ``model.onnx`` itself. The script also checks the constants the module
computes instead of storing (the log epsilon, the decoder's log frequencies, the DFT index
patterns). Dev-only; needs ``onnx``, ``onnxruntime`` and ``swift-f0``:

    .venv/bin/python scripts/extract_swift_f0.py [--onnx path/to/model.onnx] [--out path.pt]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnx import TensorProto, helper, numpy_helper
from torch.nn.modules.conv import _ConvNd

from rvc_next.engine.f0.swift_model import DEFAULT_WEIGHTS, LOG_EPS, RESOLUTIONS, SwiftF0Model, dense_state

CONV_NAMES = ["stem"] + [f"blocks.{i}.conv{j}" for i in range(6) for j in (1, 2)] + ["pitch_conv", "voicing_conv", "pitch_logits", "voicing_hidden", "voicing_out"]


def default_onnx() -> Path:
    import swift_f0

    return Path(swift_f0.__file__).with_name("model.onnx")


def attributes(node: onnx.NodeProto) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def ort_trig(size: int) -> np.ndarray:
    """``(2, size)``: onnxruntime's Cos and Sin of ``float32(m) · float32(2π/size)``, as the graph computes its angles."""
    nodes = [
        helper.make_node("Cast", ["m"], ["mf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["mf", "step"], ["angle"]),
        helper.make_node("Cos", ["angle"], ["cos"]),
        helper.make_node("Sin", ["angle"], ["sin"]),
    ]
    inputs = [helper.make_tensor_value_info("m", TensorProto.INT64, [None]), helper.make_tensor_value_info("step", TensorProto.FLOAT, [])]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [None]) for name in ("cos", "sin")]
    model = helper.make_model(helper.make_graph(nodes, "trig", inputs, outputs), opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    cos, sin = session.run(None, {"m": np.arange(size, dtype=np.int64), "step": np.asarray(2 * math.pi / size, dtype=np.float32)})
    return np.stack([np.asarray(cos), np.asarray(sin)])


def graph_trig(model: onnx.ModelProto) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """Every Cos/Sin of the graph in order: (op, size, the ``k·n mod size`` index, onnxruntime's output)."""
    graph = model.graph
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    producers = {o: n for n in graph.node for o in n.output}
    nodes = [n for n in graph.node if n.op_type in ("Cos", "Sin")]
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    probe.graph.output.extend(helper.make_tensor_value_info(n.output[0], TensorProto.FLOAT, None) for n in nodes)
    session = onnxruntime.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    feed = {"audio": np.zeros((1, 4096), np.float32), "fmin": np.asarray(100, np.float32), "fmax": np.asarray(1000, np.float32)}
    values = session.run([n.output[0] for n in nodes], feed)
    found = []
    for node, value in zip(nodes, values, strict=True):
        # Cos(Mul(Cast(Sub(k·n, (k·n / size)·size)), step)): the angle index is k·n mod size.
        mul = producers[node.input[0]]
        step = init[mul.input[1]]
        size = round(2 * math.pi / float(step))
        assert np.float32(2 * math.pi / size) == step, (node.name, step)
        product = producers[producers[producers[mul.input[0]].input[0]].input[0]]
        k, n = init[product.input[0]], init[product.input[1]]
        found.append((node.op_type, size, np.asarray((k * n) % size), np.asarray(value)))
    return found


def check_conv(name: str, conv: _ConvNd, attrs: dict) -> None:
    kernel = tuple(attrs["kernel_shape"])
    assert tuple(conv.kernel_size) == kernel, (name, conv.kernel_size, kernel)
    assert tuple(conv.dilation) == tuple(attrs.get("dilations", [1] * len(kernel))), name
    assert attrs.get("group", 1) == 1 and tuple(attrs.get("strides", [1] * len(kernel))) == (1,) * len(kernel), name
    pads = list(attrs.get("pads", [0] * 2 * len(kernel)))
    if name == "pitch_conv":  # asymmetric: (1, 0) in frequency, (1, 1) in time, applied with F.pad
        assert pads == [1, 1, 0, 1], pads
    else:
        assert pads == list(conv.padding) * 2, (name, pads, conv.padding)


def extract(model_path: Path) -> dict[str, torch.Tensor]:
    model = onnx.load(str(model_path))
    graph = model.graph
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    module = SwiftF0Model()
    out: dict[str, torch.Tensor] = {}

    convs = [n for n in graph.node if n.op_type == "Conv"]
    assert len(convs) == len(CONV_NAMES), len(convs)
    for name, node in zip(CONV_NAMES, convs, strict=True):
        conv = module.get_submodule(name)
        assert isinstance(conv, _ConvNd), name
        check_conv(name, conv, attributes(node))
        weight, bias = init[node.input[1]], init[node.input[2]]
        assert conv.bias is not None and weight.shape == tuple(conv.weight.shape) and bias.shape == tuple(conv.bias.shape), name
        out[f"{name}.weight"] = torch.from_numpy(weight.copy())
        out[f"{name}.bias"] = torch.from_numpy(bias.copy())

    # Windows: the Mul right after each framing Concat; projections: the ScatterElements, in order.
    producers = {o: n for n in graph.node for o in n.output}
    windows = [init[n.input[1]] for n in graph.node if n.op_type == "Mul" and producers.get(n.input[0], onnx.NodeProto()).op_type == "Concat"]
    scatters = [n for n in graph.node if n.op_type == "ScatterElements"]
    assert len(windows) == len(scatters) == len(RESOLUTIONS)
    for i, ((n_fft, _, _), window, scatter) in enumerate(zip(RESOLUTIONS, windows, scatters, strict=True)):
        rows = n_fft // 2 + 1
        index, value = init[scatter.input[1]], init[scatter.input[2]]
        assert window.shape == (n_fft,) and index.dtype == np.int32 and len(np.unique(index)) == len(index) and index.min() >= 0 and index.max() < rows * 128
        out[f"spectrograms.{i}.window"] = torch.from_numpy(window.copy())
        out[f"spectrograms.{i}.projection_index"] = torch.from_numpy(index.copy())
        out[f"spectrograms.{i}.projection_value"] = torch.from_numpy(value.copy())
        out[f"spectrograms.{i}.projection_rows"] = torch.tensor(rows)

    # The decoder: the band mask compares these with fmin and fmax; the pitch weights their logs.
    bins = next(init[n.input[0]] for n in graph.node if n.op_type == "GreaterOrEqual" and n.input[1] == "fmin")
    out["bin_hz"] = torch.from_numpy(bins.copy())
    log_bins = next(init[n.input[0]] for n in graph.node if n.op_type == "Gather" and n.input[0] in init and init[n.input[0]].dtype == np.float64)
    assert np.array_equal(log_bins, np.log(bins.astype(np.float64))), "the decoder's log frequencies are no longer log(bin_hz)"

    # The DFT tables: per resolution, Cos and Sin for dft_a, twiddle and dft_b, in that order.
    trig = graph_trig(model)
    assert len(trig) == 6 * len(RESOLUTIONS), len(trig)
    tables: dict[int, np.ndarray] = {}
    for i, spectrogram in enumerate(module.spectrograms):
        for j, stage in enumerate(("dft_a", "twiddle", "dft_b")):
            index = getattr(spectrogram, f"{stage}_index").numpy()
            size = getattr(spectrogram, f"{stage}_trig").shape[1]
            table = tables.setdefault(size, ort_trig(size))
            for row, (op, graph_size, graph_index, value) in enumerate(trig[6 * i + 2 * j : 6 * i + 2 * j + 2]):
                assert (op, graph_size) == (("Cos", "Sin")[row], size) and np.array_equal(graph_index, index), (i, stage, op)
                assert np.array_equal(table[row][index], value), f"onnxruntime's {op} is not a function of the angle alone ({i}, {stage})"
            out[f"spectrograms.{i}.{stage}_trig"] = torch.from_numpy(table.copy())

    # The log epsilon after each projection.
    adds = [n for n in graph.node if n.op_type == "Add" and producers.get(n.input[0], onnx.NodeProto()).op_type == "MatMul" and n.input[1] in init]
    assert len(adds) == len(RESOLUTIONS) and all(init[n.input[1]] == np.float32(LOG_EPS) for n in adds)

    module.load_state_dict(dense_state(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", type=Path, default=None, help="model.onnx (default: the installed swift_f0 package's)")
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    weights = extract(args.onnx or default_onnx())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {len(weights)} tensors)")


if __name__ == "__main__":
    main()
