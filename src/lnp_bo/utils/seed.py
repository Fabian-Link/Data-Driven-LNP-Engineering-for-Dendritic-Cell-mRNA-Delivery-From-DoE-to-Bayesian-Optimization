# src/lnp_bo/utils/seed.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class RNGState:
    """Container for Python, NumPy, and (optionally) Torch RNG states."""
    py_state: object
    np_state: tuple
    torch_cpu_state: Optional[bytes] = None
    torch_cuda_states: Optional[List[bytes]] = None


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = False,
    cudnn_benchmark: Optional[bool] = None,
) -> None:
    """
    Set global seeds for Python, NumPy, and Torch (if available).

    Parameters
    ----------
    seed : int
        Any 32-bit integer.
    deterministic : bool
        If True and torch is present, enable deterministic algorithms
        (may reduce performance and restrict some ops).
    cudnn_benchmark : Optional[bool]
        If provided and torch is present, explicitly set cudnn.benchmark.
        Default behavior when `deterministic=True` is to set it False.
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # Torch (CPU and CUDA)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # PyTorch-wide deterministic behavior
            torch.use_deterministic_algorithms(True, warn_only=True)
            # cuDNN determinism knobs
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                # If user didn't override, set benchmark to False for determinism
                if cudnn_benchmark is None:
                    torch.backends.cudnn.benchmark = False
            # Some CUDA libraries need this env var for full determinism
            # (PyTorch docs recommend one of the following values)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        else:
            # User preference for cudnn.benchmark if provided
            if cudnn_benchmark is not None and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def seed_worker(worker_id: int) -> None:
    """
    Torch DataLoader worker seeding helper.

    Usage
    -----
    DataLoader(..., worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    """
    # derive per-worker seed from the torch initial seed
    base_seed = torch.initial_seed() if torch is not None else (np.random.get_state()[1][0])
    ss = (base_seed + worker_id) % (2**32 - 1)
    random.seed(ss)
    np.random.seed(ss)
    if torch is not None:
        torch.manual_seed(ss)


def get_rng_state() -> RNGState:
    """Capture the current RNG state for Python, NumPy and Torch (if available)."""
    py_state = random.getstate()
    np_state = np.random.get_state()

    torch_cpu_state = None
    torch_cuda_states = None
    if torch is not None:
        torch_cpu_state = torch.get_rng_state().cpu().numpy().tobytes()
        if torch.cuda.is_available():
            torch_cuda_states = []
            for dev in range(torch.cuda.device_count()):
                st = torch.cuda.get_rng_state(dev).cpu().numpy().tobytes()
                torch_cuda_states.append(st)
    return RNGState(py_state, np_state, torch_cpu_state, torch_cuda_states)


def set_rng_state(state: RNGState) -> None:
    """Restore a previously captured RNG state."""
    random.setstate(state.py_state)
    np.random.set_state(state.np_state)

    if torch is not None and state.torch_cpu_state is not None:
        cpu_state = torch.from_numpy(np.frombuffer(state.torch_cpu_state, dtype=np.uint8))
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available() and state.torch_cuda_states is not None:
            for dev, st in enumerate(state.torch_cuda_states):
                buf = torch.from_numpy(np.frombuffer(st, dtype=np.uint8))
                torch.cuda.set_rng_state(buf, device=dev)


class temp_seed:
    """
    Context manager to temporarily set global seeds for Python/NumPy/(Torch).

    Example
    -------
    with temp_seed(123):
        # reproducible block
        ...
    # original RNG state is restored afterwards
    """

    def __init__(self, seed: int, deterministic: bool = False):
        self.seed = seed
        self.deterministic = deterministic
        self._saved: Optional[RNGState] = None

    def __enter__(self):
        self._saved = get_rng_state()
        set_global_seed(self.seed, deterministic=self.deterministic)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._saved is not None:
            set_rng_state(self._saved)
        return False  # don't suppress exceptions
