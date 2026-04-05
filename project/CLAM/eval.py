from __future__ import print_function

import json
import sys
import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# ---------------------------------------------------------------------------
# External evaluation helpers
# ---------------------------------------------------------------------------
DEFAULT_LABEL_MAP = {0: "LumA", 1: "LumB", 2: "Basal", 3: "Her2"}


def _load_model_external(args, ckpt_path, device):
    """Load a CLAM checkpoint for external evaluation.

    Uses ``initiate_model`` from ``utils/eval_utils.py`` which already handles
    checkpoint key cleaning (removes ``instance_loss_fn`` and ``.module``
    prefixes).  The ``device`` argument is forwarded via ``torch.device`` so
    the caller controls CPU / GPU placement.
    """
    model = initiate_model(args, ckpt_path, device=device)
    return model


def _infer_slide(model, h5_path, device):
    """Run inference on a single slide from an h5 feature file.

    Handles both (N, D) and (1, N, D) stored feature tensor shapes.
    Returns a dict with ``probs`` (numpy, shape n_classes), ``pred`` (int),
    and ``attention`` (numpy).
    """
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]

    features = torch.from_numpy(features).float().to(device)
    if features.ndim == 2:
        features = features.unsqueeze(0)

    with torch.inference_mode():
        logits, Y_prob, Y_hat, A_raw, _ = model(features)

    return {
        "probs": Y_prob.cpu().numpy().squeeze(),
        "pred": Y_hat.item(),
        "attention": A_raw.cpu().numpy(),
    }


def run_external_eval(args, device):
    """Full external-dataset evaluation: per-fold inference + ensemble.

    Reads ``args.feature_dir``, ``args.dataset_csv``, ``args.ckpt_dir``,
    ``args.output_dir``, ``args.label_map`` (dict), and the shared model
    hyper-parameters already present in the argparse namespace (``model_type``,
    ``model_size``, ``drop_out``, ``embed_dim``, ``n_classes``, ``k``).
    """
    label_map = args.label_map  # dict {int: str}
    label_to_int = {v: k for k, v in label_map.items()}

    os.makedirs(args.output_dir, exist_ok=True)

    prob_columns = [f"p_{label_map[i]}" for i in range(args.n_classes)]

    # --- Load dataset CSV ------------------------------------------------
    dataset = pd.read_csv(args.dataset_csv)
    print(f"External dataset: {len(dataset)} slides")

    if dataset["label"].dtype == object:
        dataset["label_int"] = dataset["label"].map(label_to_int)
    else:
        dataset["label_int"] = dataset["label"].astype(int)

    print(f"Label distribution:\n{dataset['label'].value_counts().to_string()}")

    # --- Check for missing feature files ---------------------------------
    missing = []
    for slide_id in dataset["slide_id"]:
        h5_path = os.path.join(args.feature_dir, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            missing.append(slide_id)
    if missing:
        print(f"WARNING: {len(missing)} slides missing features, will skip: {missing}")
        dataset = dataset[~dataset["slide_id"].isin(missing)].reset_index(drop=True)

    # --- Determine folds to run ------------------------------------------
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    if args.fold == -1:
        folds = list(range(start, end))
    else:
        folds = [args.fold]

    # --- Per-fold inference -----------------------------------------------
    all_fold_probs = []

    for fold in folds:
        ckpt_path = os.path.join(args.ckpt_dir, f"s_{fold}_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"Fold {fold}: checkpoint not found at {ckpt_path}, skipping")
            continue

        print(f"\nFold {fold}: loading {ckpt_path}")
        model = _load_model_external(args, ckpt_path, device)

        fold_results = []
        for _, row in dataset.iterrows():
            slide_id = str(row["slide_id"])
            h5_path = os.path.join(args.feature_dir, f"{slide_id}.h5")
            result = _infer_slide(model, h5_path, device)

            entry = {
                "slide_id": slide_id,
                "case_id": str(row["case_id"]),
                "true_label": int(row["label_int"]),
                "true_name": label_map[int(row["label_int"])],
                "pred_label": result["pred"],
                "pred_name": label_map[result["pred"]],
            }
            for i, col in enumerate(prob_columns):
                entry[col] = result["probs"][i]
            fold_results.append(entry)

        fold_df = pd.DataFrame(fold_results)
        fold_path = os.path.join(args.output_dir, f"fold_{fold}_predictions.csv")
        fold_df.to_csv(fold_path, index=False)

        acc = (fold_df["true_label"] == fold_df["pred_label"]).mean()
        print(f"  Fold {fold} accuracy: {acc:.4f}")

        all_fold_probs.append(fold_df[prob_columns].values)

        del model
        torch.cuda.empty_cache()

    # --- Ensemble predictions (average softmax across folds) --------------
    if all_fold_probs:
        mean_probs = np.mean(all_fold_probs, axis=0)
        ensemble_preds = mean_probs.argmax(axis=1)

        ensemble_df = dataset[["slide_id", "case_id", "label_int"]].copy()
        ensemble_df = ensemble_df.rename(columns={"label_int": "true_label"})
        ensemble_df["true_name"] = ensemble_df["true_label"].map(label_map)
        ensemble_df["pred_label"] = ensemble_preds
        ensemble_df["pred_name"] = ensemble_df["pred_label"].map(label_map)
        for i, col in enumerate(prob_columns):
            ensemble_df[col] = mean_probs[:, i]
        ensemble_df["max_prob"] = mean_probs.max(axis=1)

        ensemble_path = os.path.join(args.output_dir, "ensemble_predictions.csv")
        ensemble_df.to_csv(ensemble_path, index=False)

        acc = (ensemble_df["true_label"] == ensemble_df["pred_label"]).mean()
        print(f"\n=== Ensemble Results ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"\nPredicted distribution:")
        print(ensemble_df["pred_name"].value_counts().to_string())
        print(f"\nTrue distribution:")
        print(ensemble_df["true_name"].value_counts().to_string())
        print(f"\nSaved ensemble predictions: {ensemble_path}")
    else:
        print("\nNo fold checkpoints found. Skipping ensemble.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False,
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'],
                    default=None)
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)

# External evaluation arguments
parser.add_argument('--external', action='store_true', default=False,
                    help='run in external dataset evaluation mode (bypasses task/split logic)')
parser.add_argument('--feature_dir', type=str, default=None,
                    help='[external] directory with per-slide .h5 feature files')
parser.add_argument('--dataset_csv', type=str, default=None,
                    help='[external] CSV with slide_id, case_id, label columns')
parser.add_argument('--ckpt_dir', type=str, default=None,
                    help='[external] directory containing s_<fold>_checkpoint.pt files')
parser.add_argument('--output_dir', type=str, default=None,
                    help='[external] directory to write prediction CSVs')
parser.add_argument('--n_classes', type=int, default=None,
                    help='[external] number of output classes (required for external mode)')
parser.add_argument('--label_map', type=str, default=None,
                    help='[external] JSON string mapping int->name, '
                         'e.g. \'{"0":"LumA","1":"LumB","2":"Basal","3":"Her2"}\'. '
                         'Defaults to PAM50 4-class map.')

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# External evaluation mode
# ---------------------------------------------------------------------------
if args.external:
    # Validate required arguments for external mode
    for req_arg in ['feature_dir', 'dataset_csv', 'ckpt_dir', 'output_dir', 'n_classes']:
        if getattr(args, req_arg) is None:
            parser.error(f'--{req_arg} is required when using --external mode')

    # Parse label_map: accept JSON string or fall back to default PAM50 map
    if args.label_map is not None:
        raw = json.loads(args.label_map)
        args.label_map = {int(k): v for k, v in raw.items()}
    else:
        args.label_map = dict(DEFAULT_LABEL_MAP)

    # Verify n_classes matches label_map length
    if len(args.label_map) != args.n_classes:
        parser.error(
            f'--n_classes ({args.n_classes}) does not match label_map length '
            f'({len(args.label_map)}). Provide a matching --label_map.'
        )

    print('=== External Evaluation Mode ===')
    print(f'  feature_dir : {args.feature_dir}')
    print(f'  dataset_csv : {args.dataset_csv}')
    print(f'  ckpt_dir    : {args.ckpt_dir}')
    print(f'  output_dir  : {args.output_dir}')
    print(f'  model_type  : {args.model_type}')
    print(f'  model_size  : {args.model_size}')
    print(f'  n_classes   : {args.n_classes}')
    print(f'  embed_dim   : {args.embed_dim}')
    print(f'  drop_out    : {args.drop_out}')
    print(f'  label_map   : {args.label_map}')
    print(f'  k           : {args.k}')
    print(f'  device      : {device}')

    run_external_eval(args, device)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Original task-based evaluation mode (unchanged)
# ---------------------------------------------------------------------------
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False,
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
