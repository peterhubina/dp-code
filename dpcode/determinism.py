"""Record what CLAM's seeding actually does. Do not change it.

Everything described here is already true of `project/CLAM`. This module writes it
down so a run directory tells its reader what was and was not controlled; it does
not "improve" any of it.

That restriction is not stylistic. Adding `worker_init_fn`, calling
`torch.use_deterministic_algorithms(True)` or setting `CUBLAS_WORKSPACE_CONFIG`
would change float accumulation order and therefore change the published numbers,
while looking like a reproducibility improvement. It is forbidden here.

The honest claim, which the reproduction documentation must make in these terms:
seeds are fixed and fold assignments are fixed, so a re-run reproduces the
reported metrics to roughly 1e-3 — **not** bitwise.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "DETERMINISM_CLAIM",
    "SEEDING_SITES",
    "SEEDED_GENERATORS",
    "NOT_DONE",
    "RESIDUAL_NONDETERMINISM",
    "describe_determinism",
]

DETERMINISM_CLAIM = (
    "Seeds and fold assignments are fixed, so a re-run reproduces reported metrics "
    "to roughly 1e-3. Bitwise reproduction is NOT claimed: no deterministic-algorithm "
    "mode is enabled and nn.MultiheadAttention (cross_attention, coattn) is not "
    "bitwise deterministic on CUDA."
)

#: Where the seed is applied, and how often.
SEEDING_SITES = (
    {
        "site": "project/CLAM/main.py:234-244",
        "what": "seed_torch(seed) definition",
    },
    {
        "site": "project/CLAM/main.py:246",
        "what": "seed_torch(args.seed) at module import, before the dataset is built",
    },
    {
        "site": "project/CLAM/main.py:50",
        "what": (
            "seed_torch(args.seed) again at the top of EVERY fold, with the same seed. "
            "Folds therefore share an identical RNG stream at initialisation; the only "
            "cross-fold variation comes from different split content."
        ),
    },
)

#: What `seed_torch` sets.
SEEDED_GENERATORS = (
    "random.seed(seed)",
    "os.environ['PYTHONHASHSEED'] = str(seed)",
    "numpy.random.seed(seed)",
    "torch.manual_seed(seed)",
    "torch.cuda.manual_seed(seed)        # cuda only",
    "torch.cuda.manual_seed_all(seed)    # cuda only, all visible devices",
    "torch.backends.cudnn.benchmark = False",
    "torch.backends.cudnn.deterministic = True",
)

#: Deliberately absent from CLAM. Verified by grep over `project/CLAM/**/*.py`:
#: zero hits for any of these names.
NOT_DONE = (
    "torch.use_deterministic_algorithms",
    "CUBLAS_WORKSPACE_CONFIG",
    "DataLoader(worker_init_fn=...)",
    "DataLoader(generator=...)",
    "persistent_workers",
    "torch.amp / GradScaler (no mixed precision anywhere)",
)

#: Why a re-run is not bitwise identical.
RESIDUAL_NONDETERMINISM = (
    (
        "nn.MultiheadAttention is used by fusion_mode=cross_attention and "
        "fusion_mode=coattn and is not bitwise deterministic on CUDA."
    ),
    (
        "No deterministic-algorithm mode is enabled, so cuBLAS/cuDNN reduction "
        "kernels and scatter-style backward atomics may reorder float accumulation "
        "between runs."
    ),
    (
        "DataLoader num_workers is hardcoded to 4 with no worker seeding "
        "(project/CLAM/utils/utils.py:61). The dataset __getitem__ is a pure h5 read "
        "with no RNG, so the workers cannot inject variation today — but nothing "
        "enforces that."
    ),
    (
        "get_simple_loader has a duplicated dict key at "
        "project/CLAM/utils/utils.py:53: {'num_workers': 4, 'pin_memory': False, "
        "'num_workers': num_workers}. The second occurrence wins, so num_workers is "
        "the argument default (1), not 4."
    ),
    (
        "torch.cuda.manual_seed_all seeds whatever devices are visible, so the GPU "
        "count and CUDA_VISIBLE_DEVICES are part of the run's identity."
    ),
    (
        "Neither the checkpoint nor experiment_<exp_code>.txt records RNG state, "
        "optimizer state, epoch, embed_dim or data_root_dir, so a stranger cannot "
        "verify a re-run from CLAM's artifacts alone. That is what "
        "run_metadata.json and config.resolved.yaml are for."
    ),
)


def describe_determinism(seed: int | None = None, clam_seed: int | None = None) -> dict[str, Any]:
    """Return the structured record embedded in `run_metadata.json`.

    `seed` is dpcode's own `run.seed`; `clam_seed` is `clam.seed`, which is what
    CLAM actually trains with. Both are recorded because they are separate knobs.

    `observed_env` reports whether the *current process* carries any of the
    environment switches that would change numerics. It is a report, never an
    action: if `CUBLAS_WORKSPACE_CONFIG` is set in the environment, the run is not
    comparable to the published numbers and the metadata should say so.
    """
    return {
        "claim": DETERMINISM_CLAIM,
        "run_seed": seed,
        "clam_seed": clam_seed,
        "seeding_sites": [dict(site) for site in SEEDING_SITES],
        "seeded_generators": list(SEEDED_GENERATORS),
        "not_done": list(NOT_DONE),
        "residual_nondeterminism": list(RESIDUAL_NONDETERMINISM),
        "observed_env": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING",
                "OMP_NUM_THREADS",
            )
        },
    }
