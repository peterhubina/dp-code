from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from train_rna import run_experiment


def _none_if_missing(value):
    return None if value == "null" else value


def _hidden_dims(value):
    if isinstance(value, ListConfig):
        return [int(dim) for dim in value]
    if isinstance(value, list):
        return [int(dim) for dim in value]
    return value


def args_from_cfg(cfg: DictConfig) -> SimpleNamespace:
    return SimpleNamespace(
        rna_dir=str(cfg.data.rna_dir),
        data_path=_none_if_missing(cfg.data.data_path),
        class_set=str(cfg.data.class_set),
        k=int(cfg.splits.k),
        k_start=int(cfg.splits.k_start),
        k_end=int(cfg.splits.k_end),
        seed=int(cfg.train.seed),
        label_frac=float(cfg.splits.label_frac),
        val_frac=float(cfg.splits.val_frac),
        test_frac=float(cfg.splits.test_frac),
        split_dir=_none_if_missing(cfg.splits.split_dir),
        force_splits=bool(cfg.splits.force_splits),
        no_auto_splits=bool(cfg.splits.no_auto_splits),
        results_dir=str(cfg.output.results_dir),
        exp_code=str(cfg.output.exp_code),
        max_epochs=int(cfg.train.max_epochs),
        batch_size=int(cfg.train.batch_size),
        lr=float(cfg.optim.lr),
        reg=float(cfg.optim.reg),
        opt=str(cfg.optim.opt),
        hidden_dims=_hidden_dims(cfg.model.hidden_dims),
        drop_out=float(cfg.model.drop_out),
        label_smoothing=float(cfg.optim.label_smoothing),
        top_n_genes=int(cfg.model.top_n_genes),
        weighted_loss=bool(cfg.train.weighted_loss),
        weighted_sample=bool(cfg.train.weighted_sample),
        early_stopping=bool(cfg.train.early_stopping),
        patience=int(cfg.train.patience),
        grad_clip=float(cfg.train.grad_clip),
        num_workers=int(cfg.train.num_workers),
        wandb=bool(cfg.wandb.enabled),
        wandb_project=str(cfg.wandb.project),
        wandb_entity=_none_if_missing(cfg.wandb.entity),
        wandb_tags=list(cfg.wandb.tags) if cfg.wandb.tags is not None else None,
        wandb_mode=_none_if_missing(cfg.wandb.mode),
        wandb_log_artifacts=bool(cfg.wandb.log_artifacts),
        hydra_config=OmegaConf.to_container(cfg, resolve=False),
    )


@hydra.main(version_base="1.3", config_path="configs/rna", config_name="default")
def main(cfg: DictConfig) -> None:
    clam_dir = Path(__file__).resolve().parent
    args = args_from_cfg(cfg)
    run_experiment(args, clam_dir=clam_dir)


if __name__ == "__main__":
    main()
