"""Simulator access — the single point where an Aer fallback would happen.

Aer 0.17.2 predates Qiskit 2.5.1 and declares ``qiskit>=1.1.0`` with no upper
bound, so the pairing is untested upstream. It passes ``tools/smoke_gate.py`` on
this machine, but every notebook still goes through here so that swapping to
``BasicSimulator`` + reference primitives would be a one-file change rather than
30 edits.

Aer is deliberately used in only a handful of notebooks; everything that can be
done with ``qiskit.quantum_info`` or the reference primitives is.
"""

from __future__ import annotations

import os

SEED = 20260804
"""Base seed. Each notebook uses ``SEED + <notebook number>`` so runs are
reproducible and independent. Determinism is mandatory: without it every rebuild
produces a different diff and review becomes impossible."""


def seed_for(notebook: str) -> int:
    """Deterministic per-notebook seed, e.g. ``seed_for("A03")``."""
    digits = "".join(ch for ch in notebook if ch.isdigit())
    return SEED + (int(digits) if digits else 0)


def _thread_cap():
    # Several notebooks may execute concurrently during a build; let Aer's
    # OpenMP pool stay small so the machine does not thrash.
    return int(os.environ.get("QUTH_MAX_THREADS", "2"))


def get_backend(*, noise_model=None, method="automatic", **kwargs):
    """An AerSimulator, or the stdlib fallback if Aer is unavailable.

    ``noise_model`` is accepted directly; a noise model with the fallback raises,
    because ``BasicSimulator`` has no noise support (in that case a notebook
    should use ``quantum_info.Kraus`` on a ``DensityMatrix`` instead).
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:  # pragma: no cover - fallback path
        if noise_model is not None:
            raise RuntimeError(
                "qiskit-aer unavailable and a noise model was requested. Use "
                "qiskit.quantum_info.Kraus with DensityMatrix.evolve instead."
            )
        from qiskit.providers.basic_provider import BasicProvider
        return BasicProvider().get_backend("basic_simulator")

    opts = dict(method=method, max_parallel_threads=_thread_cap())
    if noise_model is not None:
        opts["noise_model"] = noise_model
    opts.update(kwargs)
    return AerSimulator(**opts)


def sampler(*, shots=4096, seed=SEED, noise_model=None):
    """A V2 sampler. Aer's SamplerV2 accepts ``seed=``.

    Note the asymmetry with :func:`estimator` below — it is a real quirk of the
    Aer primitives, not a typo.
    """
    try:
        from qiskit_aer.primitives import SamplerV2
    except ImportError:  # pragma: no cover
        from qiskit.primitives import StatevectorSampler
        return StatevectorSampler(seed=seed)

    opts = {}
    if noise_model is not None:
        opts["backend_options"] = {"noise_model": noise_model}
    return SamplerV2(default_shots=shots, seed=seed, options=opts or None)


def estimator(*, seed=SEED, noise_model=None):
    """A V2 estimator.

    Aer's ``EstimatorV2`` takes **no** ``seed=`` argument (unlike its SamplerV2);
    it is seeded through ``options={"backend_options": {"seed_simulator": ...}}``.
    The Phase 0 smoke gate caught this.
    """
    try:
        from qiskit_aer.primitives import EstimatorV2
    except ImportError:  # pragma: no cover
        from qiskit.primitives import StatevectorEstimator
        return StatevectorEstimator(seed=seed)

    backend_options = {"seed_simulator": seed}
    if noise_model is not None:
        backend_options["noise_model"] = noise_model
    return EstimatorV2(options={"backend_options": backend_options})
