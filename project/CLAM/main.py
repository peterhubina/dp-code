from __future__ import print_function

import argparse
import pdb
import os
import math

import wandb

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from dataset_modules.multimodal_dataset import Generic_Multimodal_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    # Track whether we own the wandb run (vs sweep agent owning it)
    sweep_mode = args.wandb and wandb is not None and wandb.run is not None

    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)

        if args.wandb and wandb is not None and not sweep_mode:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name='{}_fold{}'.format(args.exp_code, i),
                tags=args.wandb_tags,
                group=args.exp_code,
                config={**settings, 'fold': int(i)},
                dir=args.results_dir,
                reinit=True,
            )
            wandb.define_metric('epoch')
            wandb.define_metric('train/*', step_metric='epoch')
            wandb.define_metric('val/*', step_metric='epoch')
            wandb.define_metric('final/*')

        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

        if args.wandb and wandb is not None and not sweep_mode:
            wandb.finish()

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

    # Log aggregate CV summary as a separate run in the same group
    if args.wandb and wandb is not None and not sweep_mode and len(folds) > 1:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name='{}_summary'.format(args.exp_code),
            tags=(args.wandb_tags or []) + ['summary'],
            group=args.exp_code,
            config=settings,
            dir=args.results_dir,
            reinit=True,
        )
        wandb.summary['mean_test_auc'] = float(np.mean(all_test_auc))
        wandb.summary['std_test_auc'] = float(np.std(all_test_auc))
        wandb.summary['mean_val_auc'] = float(np.mean(all_val_auc))
        wandb.summary['std_val_auc'] = float(np.std(all_val_auc))
        wandb.summary['mean_test_acc'] = float(np.mean(all_test_acc))
        wandb.summary['std_test_acc'] = float(np.std(all_test_acc))
        wandb.summary['mean_val_acc'] = float(np.mean(all_val_acc))
        wandb.summary['std_val_acc'] = float(np.std(all_val_acc))
        for fi, (ta, va) in enumerate(zip(all_test_auc, all_val_auc)):
            wandb.summary['fold_{}_test_auc'.format(fi)] = ta
            wandb.summary['fold_{}_val_auc'.format(fi)] = va
        wandb.finish()

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience (default: 10)')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'tcga_brca_recurrence', 'tcga_brca_subtyping', 'tcga_brca_er', 'nou_ctc_ep', 'nou_ctc_emt', 'nou_ctc_any'])
### Multimodal fusion options
parser.add_argument('--fusion_mode', type=str,
                    choices=['concat', 'gated', 'residual', 'cross_attention', 'film_attention', 'coattn'], default=None,
                    help='enable WSI + tabular multimodal fusion; supports concat, gated, residual, '
                         'cross_attention, film_attention and coattn')
parser.add_argument('--film_rank', type=int, default=32,
                    help='rank of the FiLM conditioner for --fusion_mode film_attention; 0 disables the '
                         'attention conditioning, leaving additive-logit fusion')
parser.add_argument('--modality_dropout', type=float, default=0.0,
                    help='probability of dropping the tabular modality during training (film_attention/coattn)')
parser.add_argument('--tabular_group_spec', type=str, default=None,
                    help='token grouping for --fusion_mode coattn: path to a signature CSV, or "prefix" to '
                         'group one-hot blocks by column-name prefix')
parser.add_argument('--tabular_csv', type=str, default=None,
                    help='CSV containing case_id, label and RNA/tabular feature columns')
parser.add_argument('--tabular_case_id_col', type=str, default='case_id',
                    help='case identifier column in --tabular_csv')
parser.add_argument('--tabular_label_col', type=str, default='label',
                    help='label column in --tabular_csv')
parser.add_argument('--tabular_hidden_dim', type=int, default=256,
                    help='hidden dimension for the tabular MLP encoder')
parser.add_argument('--tabular_num_layers', type=int, default=2,
                    help='number of hidden layers in the tabular MLP encoder')
parser.add_argument('--tabular_top_n_features', type=int, default=0,
                    help='select top-N tabular features by training-fold variance; 0 keeps all features')
parser.add_argument('--fusion_hidden_dim', type=int, default=128,
                    help='hidden dimension for the fusion head; <=0 uses a linear concat head')
parser.add_argument('--pretrained_wsi_ckpt', type=str, default=None,
                    help='optional WSI-only CLAM checkpoint used to initialize the WSI branch; may include {fold}')
parser.add_argument('--freeze_wsi_branch', action='store_true', default=False,
                    help='freeze the WSI branch after loading --pretrained_wsi_ckpt')
parser.add_argument('--pretrained_rna_ckpt', type=str, default=None,
                    help='optional matched RNA checkpoint used by residual fusion; may include {fold}')
parser.add_argument('--freeze_rna_branch', action='store_true', default=False,
                    help='freeze the RNA branch after loading --pretrained_rna_ckpt')
parser.add_argument('--rna_hidden_dims', type=str, default='1024,512',
                    help='hidden dimensions for the residual RNA MLP branch')
parser.add_argument('--rna_dropout', type=float, default=0.4,
                    help='dropout used when constructing the residual RNA MLP branch')
parser.add_argument('--residual_scale', type=float, default=0.2,
                    help='scale applied to WSI-conditioned residual logits')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--wandb', action='store_true', default=False, help='enable wandb logging')
parser.add_argument('--wandb_project', type=str, default='clam-subtyping', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='wandb tags (space-separated)')
parser.add_argument('--log_heatmaps', type=int, default=0,
                    help='log top-K correct + top-K wrong attention heatmaps to W&B per fold (0=off)')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.fusion_mode is not None:
    if args.tabular_csv is None:
        parser.error('--tabular_csv is required when --fusion_mode is set')
    if args.model_type not in ['clam_sb', 'clam_mb']:
        parser.error('--fusion_mode requires --model_type clam_sb or clam_mb')
    if args.freeze_wsi_branch and args.pretrained_wsi_ckpt is None:
        parser.error('--freeze_wsi_branch requires --pretrained_wsi_ckpt')
    if args.fusion_mode == 'residual' and args.pretrained_rna_ckpt is None:
        parser.error('--fusion_mode residual requires --pretrained_rna_ckpt')
    if args.fusion_mode == 'coattn' and args.tabular_group_spec is None:
        parser.error('--fusion_mode coattn requires --tabular_group_spec')
    if args.fusion_mode == 'coattn' and args.log_heatmaps:
        # coattn returns token-to-patch attention (1 x n_tokens x n_patches), not the
        # per-class patch attention the heatmap logger expects.
        parser.error('--log_heatmaps is not supported with --fusion_mode coattn')
    if args.film_rank < 0:
        parser.error('--film_rank must be >= 0')
    if not 0.0 <= args.modality_dropout < 1.0:
        parser.error('--modality_dropout must lie in [0, 1)')

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'fusion_mode': args.fusion_mode,
            'tabular_csv': args.tabular_csv,
            'tabular_case_id_col': args.tabular_case_id_col,
            'tabular_label_col': args.tabular_label_col,
            'tabular_hidden_dim': args.tabular_hidden_dim,
            'tabular_num_layers': args.tabular_num_layers,
            'tabular_top_n_features': args.tabular_top_n_features,
            'fusion_hidden_dim': args.fusion_hidden_dim,
            'pretrained_wsi_ckpt': args.pretrained_wsi_ckpt,
            'freeze_wsi_branch': args.freeze_wsi_branch,
            'pretrained_rna_ckpt': args.pretrained_rna_ckpt,
            'freeze_rna_branch': args.freeze_rna_branch,
            'rna_hidden_dims': args.rna_hidden_dims,
            'rna_dropout': args.rna_dropout,
            'residual_scale': args.residual_scale,
            'film_rank': args.film_rank,
            'modality_dropout': args.modality_dropout,
            'tabular_group_spec': args.tabular_group_spec}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

def build_mil_dataset(data_dir, **kwargs):
    if args.fusion_mode is None:
        return Generic_MIL_Dataset(data_dir=data_dir, **kwargs)

    return Generic_Multimodal_MIL_Dataset(
        data_dir=data_dir,
        tabular_csv=args.tabular_csv,
        tabular_case_id_col=args.tabular_case_id_col,
        tabular_label_col=args.tabular_label_col,
        **kwargs,
    )

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = build_mil_dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = build_mil_dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping

elif args.task == 'tcga_brca_recurrence':
    args.n_classes = 2
    dataset = build_mil_dataset(csv_path = 'dataset_csv/tcga_brca_recurrence.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'no_recurrence': 0, 'recurrence': 1},
                            patient_strat = True,
                            ignore = [])
    dataset.load_from_h5(True)

elif args.task == 'tcga_brca_subtyping':
    args.n_classes = 4
    dataset = build_mil_dataset(csv_path = 'dataset_csv/tcga_brca_subtyping.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'LumA': 0, 'LumB': 1, 'Basal': 2, 'Her2': 3},
                            patient_strat = True,
                            ignore = ['Normal'])
    dataset.load_from_h5(True)

elif args.task == 'tcga_brca_er':
    args.n_classes = 2
    dataset = build_mil_dataset(csv_path = 'dataset_csv/tcga_brca_er.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'ER-negative': 0, 'ER-positive': 1},
                            patient_strat = True,
                            ignore = [])
    dataset.load_from_h5(True)

elif args.task == 'nou_ctc_ep':
    args.n_classes = 2
    dataset = build_mil_dataset(csv_path = 'dataset_csv/nou_ctc_ep.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'no_ep': 0, 'ep': 1},
                            patient_strat = True,
                            ignore = [])
    dataset.load_from_h5(True)

elif args.task == 'nou_ctc_emt':
    args.n_classes = 2
    dataset = build_mil_dataset(csv_path = 'dataset_csv/nou_ctc_emt.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'no_emt': 0, 'emt': 1},
                            patient_strat = True,
                            ignore = [])
    dataset.load_from_h5(True)

elif args.task == 'nou_ctc_any':
    args.n_classes = 2
    dataset = build_mil_dataset(csv_path = 'dataset_csv/nou_ctc_any.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'no_any': 0, 'any': 1},
                            patient_strat = True,
                            ignore = [])
    dataset.load_from_h5(True)

else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if args.wandb and wandb is not None and wandb.run is not None:
    # Sweep mode: run already initialized by agent — update config once
    wandb.config.update(settings, allow_val_change=True)

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
