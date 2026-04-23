"""
Reusable Hugging Face model loading for corpus scoring.

Models are cached under the project-level models/ directory. Loading is
GPU-only by default: if a model cannot fit on the selected GPU using the
configured strategies, loading fails instead of silently offloading to CPU.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Force Transformers into PyTorch-only mode before importing transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

DEFAULT_DOWNLOAD_WORKERS = int(os.environ.get("HF_DOWNLOAD_WORKERS", "2"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MODELS_DIR = PROJECT_ROOT / "models"
HF_HOME = MODELS_DIR / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE))

from huggingface_hub import scan_cache_dir, snapshot_download


@dataclass(frozen=True)
class ModelConfig:
    hf_id: str
    role: str
    strategies: tuple[str, ...]


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llama_3_1_8b": ModelConfig(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        role="compact accessible baseline",
        strategies=("bf16",),
    ),
    "qwen_2_5_14b": ModelConfig(
        hf_id="Qwen/Qwen2.5-14B-Instruct",
        role="mid-sized instruction-tuned model",
        strategies=("bf16", "8bit", "4bit"),
    ),
    "deepseek_r1_distill_qwen_14b": ModelConfig(
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        role="reasoning-distilled model at similar scale to Qwen 14B",
        strategies=("bf16", "8bit", "4bit"),
    ),
    "mistral_small_3_1_24b": ModelConfig(
        hf_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        role="upper local capability tier",
        strategies=("8bit", "4bit"),
    ),
}


def list_models() -> dict[str, dict[str, Any]]:
    """Return model registry metadata in a notebook-friendly shape."""
    return {
        key: {
            "hf_id": config.hf_id,
            "role": config.role,
            "strategies": " -> ".join(config.strategies),
        }
        for key, config in MODEL_REGISTRY.items()
    }


def model_is_cached(model_key: str) -> bool:
    """Return True when a project-local HF snapshot has complete weights."""
    config = get_model_config(model_key)
    try:
        snapshot_path = Path(snapshot_download(
            repo_id=config.hf_id,
            repo_type="model",
            cache_dir=str(HF_HUB_CACHE),
            local_files_only=True,
        ))
    except Exception:
        return False
    return _snapshot_has_complete_weights(snapshot_path)


def _snapshot_has_complete_weights(snapshot_path: Path) -> bool:
    """Check for complete model weights in a local HF snapshot."""
    index_path = snapshot_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
            required = set(index.get("weight_map", {}).values())
        except Exception:
            return False
        return bool(required) and all((snapshot_path / filename).exists() for filename in required)

    weight_patterns = ("*.safetensors", "*.bin", "*.gguf", "*.pt")
    return any(any(snapshot_path.rglob(pattern)) for pattern in weight_patterns)


def cache_status() -> dict[str, dict[str, Any]]:
    """Return project-local cache status for each registered model."""
    try:
        cache_info = scan_cache_dir(HF_HUB_CACHE)
        cached_repos = {
            repo.repo_id: {
                "size_on_disk_gb": repo.size_on_disk / 1024**3,
                "nb_files": repo.nb_files,
            }
            for repo in cache_info.repos
            if repo.repo_type == "model"
        }
    except Exception:
        cached_repos = {}

    status = {}
    for key, config in MODEL_REGISTRY.items():
        repo_info = cached_repos.get(config.hf_id)
        complete = model_is_cached(key)
        status[key] = {
            "hf_id": config.hf_id,
            "cached": complete,
            "partial_cache_present": repo_info is not None and not complete,
            "cache_dir": str(HF_HUB_CACHE),
            "size_on_disk_gb": None if repo_info is None else repo_info["size_on_disk_gb"],
            "nb_files": None if repo_info is None else repo_info["nb_files"],
        }
    return status


def download_model(
    model_key: str,
    *,
    force: bool = False,
    max_workers: int | None = None,
) -> str:
    """
    Download one registered model into the project-local HF cache.

    Existing cached models are skipped unless force=True.
    """
    config = get_model_config(model_key)
    workers = DEFAULT_DOWNLOAD_WORKERS if max_workers is None else max_workers
    if model_is_cached(model_key) and not force:
        print(f"Already cached: {model_key} ({config.hf_id})")
        return str(HF_HUB_CACHE)

    print(f"Downloading: {model_key} ({config.hf_id})")
    print(f"Download workers: {workers}")
    snapshot_download(
        repo_id=config.hf_id,
        repo_type="model",
        cache_dir=str(HF_HUB_CACHE),
        local_files_only=False,
        max_workers=workers,
        resume_download=True,
    )
    print(f"Cached under: {HF_HUB_CACHE}")
    return str(HF_HUB_CACHE)


def download_missing_models(
    model_keys: list[str] | None = None,
    *,
    max_workers: int | None = None,
) -> dict[str, dict[str, str]]:
    """Download only registered models that are not already cached.

    Returns per-model status and continues if one model is gated or fails.
    """
    keys = model_keys or list(MODEL_REGISTRY)
    results: dict[str, dict[str, str]] = {}
    for key in keys:
        try:
            cache_dir = download_model(key, force=False, max_workers=max_workers)
            results[key] = {"status": "ok", "cache_dir": cache_dir}
        except Exception as exc:
            print(f"Download failed for {key}: {type(exc).__name__}: {exc}")
            results[key] = {"status": "failed", "error": repr(exc)}
    return results


def get_model_config(model_key: str) -> ModelConfig:
    """Fetch one model config by key."""
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model_key={model_key!r}. Available: {available}") from exc


def clear_gpu_memory() -> None:
    """Release Python and CUDA cache between failed load attempts."""
    torch = _get_torch()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _get_torch():
    import torch

    return torch


def _get_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _vram_gb(device: int = 0) -> dict[str, float]:
    torch = _get_torch()
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
    }


def _quantization_config(strategy: str):
    torch = _get_torch()
    _, _, BitsAndBytesConfig = _get_transformers()
    if strategy == "bf16":
        return None
    if strategy == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if strategy == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unsupported loading strategy: {strategy}")


def _load_once(
    model_id: str,
    strategy: str,
    device: int,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    torch = _get_torch()
    AutoModelForCausalLM, AutoTokenizer, _ = _get_transformers()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        cache_dir=str(HF_HUB_CACHE),
    )

    quantization_config = _quantization_config(strategy)
    model_kwargs: dict[str, Any] = {
        "device_map": {"": device},
        "trust_remote_code": trust_remote_code,
        "cache_dir": str(HF_HUB_CACHE),
    }

    if quantization_config is None:
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model


def _is_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "out of memory",
            "cuda error",
            "cublas",
            "paging file is too small",
            "winerror 1455",
            "not enough memory",
            "defaultcpuallocator",
        )
    )


def load_model(
    model_key: str,
    *,
    device: int = 0,
    strategies: tuple[str, ...] | None = None,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    """
    Load tokenizer/model with the least aggressive configured strategy.

    Returns a dict with tokenizer, model, selected strategy, device map, cache
    paths, and VRAM usage. CPU offload is intentionally not used.
    """
    torch = _get_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU-only model loading requires CUDA.")
    if device >= torch.cuda.device_count():
        raise ValueError(f"Requested CUDA device {device}, but only {torch.cuda.device_count()} device(s) exist.")

    config = get_model_config(model_key)
    attempted_strategies = strategies or config.strategies
    failures: list[tuple[str, str]] = []

    print(f"Model key: {model_key}")
    print(f"Hugging Face ID: {config.hf_id}")
    print(f"Cache directory: {HF_HUB_CACHE}")
    print(f"CUDA device: {device} - {torch.cuda.get_device_name(device)}")
    print(f"Strategy order: {' -> '.join(attempted_strategies)}")

    for strategy in attempted_strategies:
        clear_gpu_memory()
        print(f"\nAttempting load strategy: {strategy}")
        try:
            tokenizer, model = _load_once(
                config.hf_id,
                strategy=strategy,
                device=device,
                trust_remote_code=trust_remote_code,
            )
            usage = _vram_gb(device)
            device_map = getattr(model, "hf_device_map", {"": device})
            print(f"Loaded with strategy: {strategy}")
            print(f"Device map: {device_map}")
            print(f"Allocated VRAM: {usage['allocated_gb']:.2f} GB")
            print(f"Reserved VRAM: {usage['reserved_gb']:.2f} GB")
            return {
                "tokenizer": tokenizer,
                "model": model,
                "model_key": model_key,
                "model_id": config.hf_id,
                "strategy": strategy,
                "device": device,
                "device_map": device_map,
                "cache_dir": str(HF_HUB_CACHE),
                "vram": usage,
            }
        except Exception as exc:
            failures.append((strategy, repr(exc)))
            print(f"Strategy failed: {strategy}")
            print(f"{type(exc).__name__}: {exc}")
            clear_gpu_memory()
            if not _is_memory_error(exc):
                raise

    failure_text = "\n".join(f"- {strategy}: {error}" for strategy, error in failures)
    raise RuntimeError(
        f"Could not load {model_key!r} on CUDA device {device} with GPU-only strategies:\n{failure_text}"
    )
