"""Structured configs for the dp-code pipeline.

Importing this module has two side effects, both required before any config is
composed: it registers the ``dp.repo_root`` OmegaConf resolver, and it registers
the dataclasses below in Hydra's ``ConfigStore``.

The payoff of typing the surface is fail-fast: ``clam.lrr=0.1`` dies at
composition with ``Key 'lrr' is not in struct`` instead of at ``float()`` twenty
minutes into a training run. The escape hatch Hydra itself suggests in that error
message — ``+clam.lrr=0.1`` — silently succeeds, is recorded in
``.hydra/overrides.yaml``, is logged to W&B, and is read by nothing; a sweep
launched with a ``+``-prefixed typo produces N identical runs that look
different. :func:`reject_appended_overrides` closes that hole.

``ClamConf`` mirrors all 52 arguments of ``project/CLAM/main.py`` with CLAM's own
default and type. It is hand-written for readability and pinned to the real
parser by ``dp-config sync-check`` and by ``tests/test_schema.py``: adding an
argument to ``main.py`` without adding it here fails a check rather than becoming
silently unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from . import paths as _paths

__all__ = [
    "PathsConf",
    "SourcesConf",
    "ClamConf",
    "FusionConf",
    "WandbConf",
    "RunConf",
    "RootConf",
    "CONFIG_SCHEMA_NAME",
    "register_configs",
    "reject_appended_overrides",
]

#: Name under which :class:`RootConf` is stored; referenced from the defaults
#: list of ``dpcode/conf/config.yaml``.
CONFIG_SCHEMA_NAME = "dp_config_schema"


@dataclass
class PathsConf:
    """Every filesystem location, resolved from `dpcode/conf/paths/default.yaml`.

    `nou_root` is `None` unless `DP_NOU_ROOT` is set: it points at a private
    institutional cohort and has no committed default.
    """

    repo_root: str = MISSING
    data_root: str = MISSING
    scratch_root: str = MISSING
    results_root: str = MISSING

    clam_root: str = MISSING
    splits_root: str = MISSING
    dataset_csv_dir: str = MISSING
    labels_dir: str = MISSING

    tcga_embeddings: str = MISSING
    cptac_embeddings: str = MISSING
    cptac_root: str = MISSING
    cnv_dir: str = MISSING
    cnv_reference_dir: str = MISSING
    cnv_tabular_dir: str = MISSING
    cptac_validation_dir: str = MISSING
    analysis_dir: str = MISSING
    harmonisation_dir: str = MISSING
    rna_gdc_dir: str = MISSING
    rna_legacy_dir: str = MISSING
    uni_checkpoint_dir: str = MISSING

    hsi_bc_root: str = MISSING
    nou_root: Optional[str] = None


@dataclass
class SourcesConf:
    """Remote endpoints and dataset identifiers every acquisition step uses.

    Two of these are deliberately visible rather than hidden in code because they
    are *not pinned*: the cBioPortal datahub is fetched from a moving branch, and
    UCSC `refGene` is a live table. The 39 arm values are therefore not
    reproducible from the remotes alone — the cached
    `${paths.cnv_reference_dir}/gene_arm_hg38.csv` is what pins the gene->arm map.
    """

    # cBioPortal
    cbioportal_datahub_base: str = "https://media.githubusercontent.com/media/cBioPortal/datahub"
    cbioportal_datahub_ref: str = "master"  # NOT PINNED - a moving branch
    cbioportal_datahub_url: str = "${.cbioportal_datahub_base}/${.cbioportal_datahub_ref}/public"
    cbioportal_api_base: str = "https://www.cbioportal.org/api"
    tcga_study_id: str = "brca_tcga_pan_can_atlas_2018"
    cptac_study_id: str = "brca_cptac_2020"

    # UCSC hg38 annotation (refGene is a live table - see the class docstring)
    ucsc_goldenpath_base: str = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database"
    ucsc_cytoband_file: str = "cytoBand.txt.gz"
    ucsc_refgene_file: str = "refGene.txt.gz"

    # GDC (open access; no token, no dbGaP)
    gdc_api_base: str = "https://api.gdc.cancer.gov"

    # TCIA PathDB - CPTAC slides over plain HTTPS, no Aspera
    pathdb_base: str = "https://pathdb.cancerimagingarchive.net"

    # Zenodo record 8394329 - CPTAC pan-cancer clinical table
    zenodo_record_id: str = "8394329"
    zenodo_files_url: str = "https://zenodo.org/api/records/${.zenodo_record_id}/files"

    # HGNC complete set, used by the dormant RNA harmonisation arm
    hgnc_complete_set_url: str = (
        "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
    )

    # TCGA-CDR supplement (Liu 2018). An Elsevier CDN link fetched with a spoofed
    # User-Agent; it may rot or block. The xlsx it produces is tracked.
    tcga_cdr_xlsx_url: str = (
        "https://ars.els-cdn.com/content/image/1-s2.0-S0092867418302290-mmc1.xlsx"
    )

    # UCSC Xena - the LEGACY RNA route, superseded by the GDC tables
    xena_hub_base: str = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"

    # GATED HuggingFace repositories. Access is by request on the hub; set
    # HF_TOKEN once approved.
    hf_features_repo: str = "MahmoodLab/UNI2-h-features"
    hf_encoder_repo: str = "MahmoodLab/UNI2-h"


@dataclass
class ClamConf:
    """All 52 arguments of `project/CLAM/main.py`, with CLAM's own defaults.

    Field order follows the parser's declaration order so this class reads as the
    same document as `main.py:120-209` and `dpcode/conf/clam/base.yaml`.

    NOTE the arguments that are silently inert on the multimodal path: when
    `fusion_mode` is set, `utils/core_utils.py:434` routes to `train_loop` rather
    than `train_loop_clam`, so `bag_weight`, `inst_loss` and `no_inst_cluster`
    have no effect at all and `B` is passed to the model but never used. The
    ladder passes them anyway; they are recorded, not obeyed.
    """

    # generic training settings
    data_root_dir: Optional[str] = None
    embed_dim: int = 1024
    max_epochs: int = 200
    lr: float = 1e-4
    label_frac: float = 1.0
    reg: float = 1e-5
    seed: int = 1
    k: int = 10
    k_start: int = -1
    k_end: int = -1
    results_dir: str = "./results"
    split_dir: Optional[str] = None
    log_data: bool = False
    testing: bool = False
    early_stopping: bool = False
    patience: int = 10
    opt: str = "adam"
    drop_out: float = 0.25
    bag_loss: str = "ce"
    model_type: str = "clam_sb"
    exp_code: Optional[str] = None
    weighted_sample: bool = False
    model_size: str = "small"
    task: Optional[str] = None

    # multimodal fusion options (this project's fork)
    fusion_mode: Optional[str] = None
    film_rank: int = 32
    modality_dropout: float = 0.0
    tabular_group_spec: Optional[str] = None
    tabular_csv: Optional[str] = None
    tabular_case_id_col: str = "case_id"
    tabular_label_col: str = "label"
    tabular_hidden_dim: int = 256
    tabular_num_layers: int = 2
    tabular_top_n_features: int = 0
    fusion_hidden_dim: int = 128
    pretrained_wsi_ckpt: Optional[str] = None
    freeze_wsi_branch: bool = False
    pretrained_rna_ckpt: Optional[str] = None
    freeze_rna_branch: bool = False
    rna_hidden_dims: str = "1024,512"
    rna_dropout: float = 0.4
    residual_scale: float = 0.2

    # CLAM-specific options
    no_inst_cluster: bool = False
    inst_loss: Optional[str] = None
    subtyping: bool = False
    bag_weight: float = 0.7
    B: int = 8

    # W&B flags on CLAM's own CLI (distinct from the `wandb` config group)
    wandb: bool = False
    wandb_project: str = "clam-subtyping"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    log_heatmaps: int = 0


@dataclass
class FusionConf:
    """Which fusion operator a run is, as a label.

    The operator's actual CLAM flags (`fusion_mode`, `film_rank`,
    `modality_dropout`, `tabular_*`, `fusion_hidden_dim`) live in `clam` and
    nowhere else, so there is exactly one source of truth for what is passed to
    `main.py`. This node exists so a run directory, a W&B run and a metrics table
    can name the ladder arm without re-deriving it from flags.
    """

    name: str = "none"


@dataclass
class WandbConf:
    """The dpcode-level W&B settings, selected by the `wandb` config group.

    This is NOT the same thing as `clam.wandb*`. CLAM's own W&B flags are part of
    its argparse surface and are rendered into argv; this node is what dpcode's
    own entry points use, and what an entry point may choose to map onto the
    `clam.wandb*` fields. Keeping them separate is deliberate: `clam/base.yaml`
    holds CLAM's defaults verbatim, so it cannot quietly change what a legacy
    wrapper passed.
    """

    enabled: bool = False
    mode: str = "disabled"  # online | offline | disabled
    project: str = "clam-brca-subtyping-cv"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dir: str = "${paths.scratch_root}/wandb"


@dataclass
class RunConf:
    """Run-level bookkeeping that is not a CLAM flag.

    `seed` seeds dpcode's own entry points (the analysis scripts, bootstraps).
    It is NOT what CLAM trains with — that is `clam.seed`, kept separate so
    `clam/base.yaml` stays a verbatim record of CLAM's defaults. Both are written
    into `run_metadata.json`.
    """

    name: str = "unnamed"
    seed: int = 1
    #: Permit writing into a run directory that already holds results.
    overwrite: bool = False
    #: Permit `+key=…` / `~key` overrides. Off by default; see
    #: :func:`reject_appended_overrides`.
    allow_config_surgery: bool = False


@dataclass
class RootConf:
    paths: PathsConf = field(default_factory=PathsConf)
    sources: SourcesConf = field(default_factory=SourcesConf)
    clam: ClamConf = field(default_factory=ClamConf)
    fusion: FusionConf = field(default_factory=FusionConf)
    wandb: WandbConf = field(default_factory=WandbConf)
    run: RunConf = field(default_factory=RunConf)

    # Placeholders for the config groups owned by the other tracks, so their
    # group options have a package to merge into under struct mode. Each stays
    # `None` until its group is selected in the defaults list.
    experiment: Optional[Any] = None  # Track B
    analysis: Optional[Any] = None  # Track C
    evaluate: Optional[Any] = None  # Track D
    acquire: Optional[Any] = None  # Track D
    cptac: Optional[Any] = None  # Track D


def register_configs() -> None:
    """Register the root schema. Idempotent — ConfigStore overwrites by name."""
    _paths.register_resolvers()
    ConfigStore.instance().store(name=CONFIG_SCHEMA_NAME, node=RootConf)


def reject_appended_overrides(
    overrides: Optional[Sequence[str]] = None,
    *,
    allow: bool = False,
) -> None:
    """Abort on `+key=…`, `++key=…` or `~key` overrides.

    Struct mode already rejects a typo'd `clam.lrr=0.1`. The danger is Hydra's own
    suggested remedy: `+clam.lrr=0.1` succeeds, is recorded everywhere, and is
    read by nothing. `~key` is equally silent — the job runs to completion if
    nothing reads the deleted key.

    Pass `overrides` explicitly when composing with `compose()`; leave it `None`
    under `@hydra.main` and the list is read from `HydraConfig`.

    `allow` corresponds to `run.allow_config_surgery=true`.
    """
    if allow:
        return
    if overrides is None:
        from hydra.core.hydra_config import HydraConfig

        overrides = list(HydraConfig.get().overrides.task)
    surgery = [o for o in overrides if o.startswith(("+", "~"))]
    if surgery:
        raise ValueError(
            f"Config-surgery overrides are not allowed: {surgery}. "
            "`+key=…` appends a key that nothing reads (a typo'd sweep produces N "
            "identical runs); `~key` deletes one. Add the key to a config file, or "
            "pass run.allow_config_surgery=true if you really mean it."
        )


register_configs()
