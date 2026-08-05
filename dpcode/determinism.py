"""Record what CLAM's seeding actually does. Do not change it.

Everything described here is already true of `project/CLAM`. This module writes it
down so a run directory tells its reader what was and was not controlled; it does
not "improve" any of it.

That restriction is not stylistic. Calling `torch.use_deterministic_algorithms(True)`
or setting `CUBLAS_WORKSPACE_CONFIG` changes which kernels run, and therefore float
accumulation order and therefore the published numbers, while looking like a
reproducibility improvement. Adding a `worker_init_fn` is a weaker case — the
feature-loading Dataset is a pure h5 read that uses neither `numpy.random` nor
`random`, and torch already derives worker seeds from the seeded parent generator —
but it is equally forbidden, because "probably inert" is not a basis for touching
the code path behind a published table.

The honest claim, which the reproduction documentation must make in these terms:
seeds and fold assignments are fixed; run-to-run variance has **not** been
measured; bitwise reproducibility is neither claimed nor achievable here. No
tolerance is quoted, because nobody computed one — quoting a number nobody
measured is the same class of error this project's own reporting rules criticise.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "DETERMINISM_CLAIM",
    "SEEDING_SITES",
    "SEEDED_GENERATORS",
    "SEEDING_WITHOUT_EFFECT",
    "NOT_DONE",
    "RESIDUAL_NONDETERMINISM",
    "describe_determinism",
]

DETERMINISM_CLAIM = (
    "Seeds and fold assignments are fixed. Run-to-run variance has NOT been measured, "
    "so no tolerance is quoted. Bitwise reproducibility is neither claimed nor "
    "achievable here: nn.MultiheadAttention (used by fusion_mode=cross_attention and "
    "fusion_mode=coattn) is not bitwise deterministic on CUDA, and neither "
    "torch.use_deterministic_algorithms nor CUBLAS_WORKSPACE_CONFIG is set."
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

#: What `seed_torch` sets AND that takes effect. `PYTHONHASHSEED` is deliberately
#: not in this list — see :data:`SEEDING_WITHOUT_EFFECT`.
SEEDED_GENERATORS = (
    "random.seed(seed)",
    "numpy.random.seed(seed)",
    "torch.manual_seed(seed)",
    "torch.cuda.manual_seed(seed)        # cuda only",
    "torch.cuda.manual_seed_all(seed)    # cuda only, all visible devices",
    "torch.backends.cudnn.benchmark = False",
    "torch.backends.cudnn.deterministic = True",
)

#: Statements inside `seed_torch` that do NOT do what their name suggests. Listed
#: separately so `run_metadata.json` never reports a seed as being "in effect"
#: when it is not.
SEEDING_WITHOUT_EFFECT = (
    {
        "statement": "os.environ['PYTHONHASHSEED'] = str(seed)",
        "site": "project/CLAM/main.py:237",
        "why": (
            "PYTHONHASHSEED is read by the interpreter at startup. Assigning it "
            "from inside an already-running interpreter does not re-seed string "
            "hashing for that process; it only changes the environment CHILD "
            "processes inherit. Recorded as observed environment (see "
            "`observed_env`), never as a seed in effect."
        ),
    },
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
    comparable to the published numbers and the metadata should say so. It is
    also where `PYTHONHASHSEED` belongs — as what the process inherited, not as
    something `seed_torch` established.
    """
    return {
        "claim": DETERMINISM_CLAIM,
        "measured_tolerance": None,
        "run_seed": seed,
        "clam_seed": clam_seed,
        "seeding_sites": [dict(site) for site in SEEDING_SITES],
        "seeded_generators": list(SEEDED_GENERATORS),
        "seeding_without_effect": [dict(item) for item in SEEDING_WITHOUT_EFFECT],
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
