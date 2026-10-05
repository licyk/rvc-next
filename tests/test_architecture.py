"""The layer rule: each layer imports only from the columns it is allowed."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "rvc_next"

FRAMEWORKS = {"fastapi", "starlette", "typer", "click", "socketio", "uvicorn", "rich"}
ENGINE_THIRD_PARTY = {
    "numpy",
    "scipy",
    "torch",
    "transformers",
    "faiss",
    "parselmouth",
    "librosa",
    "av",
    "soundfile",
    "onnxruntime",
    "torch_directml",
    "tensorboard",
}

# layer -> (allowed rvc_next subpackages, allowed third-party top-level modules)
RULES: dict[str, tuple[set[str], set[str]]] = {
    "engine": ({"engine"}, ENGINE_THIRD_PARTY | {"sounddevice", "pymss", "pymss_core", "yaml"}),
    "protocol": ({"protocol"}, set()),
    "workers": ({"engine", "protocol", "workers"}, ENGINE_THIRD_PARTY | {"sounddevice", "pymss", "yaml"}),
    "core": ({"core", "engine", "protocol", "version", "logger"}, {"pydantic", "httpx", "tomli", "tomli_w", "send2trash", "numpy", "typing_extensions"}),
    "api": ({"api", "core", "version", "logger", "webui"}, {"fastapi", "starlette", "socketio", "uvicorn", "pydantic", "anyio"}),
    "cli": ({"cli", "core", "version", "logger", "api"}, {"typer", "click", "rich", "pydantic", "uvicorn"}),
}

# Narrower rules inside the engine: device access and separation stay in their own folders.
ENGINE_ONLY_IN = {"sounddevice": "engine/audio_io", "pymss": "engine/separate", "pymss_core": "engine/separate"}


def _stdlib() -> set[str]:
    return set(sys.stdlib_module_names)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append(node.module)
    return out


def _modules(layer: str) -> list[Path]:
    root = PACKAGE / layer
    return [p for p in root.rglob("*.py") if "node_modules" not in p.parts]


@pytest.mark.parametrize("layer", sorted(RULES))
def test_layer_imports(layer: str) -> None:
    own, third = RULES[layer]
    stdlib = _stdlib()
    bad: list[str] = []
    for path in _modules(layer):
        rel = path.relative_to(PACKAGE).as_posix()
        for name in _imports(path):
            top = name.split(".")[0]
            if top == "rvc_next":
                sub = name.split(".")[1] if "." in name else ""
                if sub not in own:
                    bad.append(f"{rel}: {name}")
            elif top in stdlib or top == "__future__":
                continue
            elif top not in third:
                bad.append(f"{rel}: {name}")
            elif layer == "engine" and top in ENGINE_ONLY_IN and not rel.startswith(ENGINE_ONLY_IN[top]):
                bad.append(f"{rel}: {name} (only in {ENGINE_ONLY_IN[top]})")
            elif layer in ("api", "cli") and top in FRAMEWORKS - third:
                bad.append(f"{rel}: {name}")
    assert not bad, "Imports outside the layer rule:\n" + "\n".join(bad)


def test_core_import_does_not_load_torch() -> None:
    code = "import sys, rvc_next.core.context; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_cli_import_does_not_load_torch_or_fastapi() -> None:
    code = "import sys, rvc_next.cli.app; print('torch' in sys.modules, 'fastapi' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False False"
