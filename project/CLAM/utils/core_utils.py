import json
import os

import numpy as np
import pandas as pd
import torch
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_multimodal import CLAMRNAFusion
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from utils.utils import *

try:
    import wandb
except ImportError:
    wandb = None

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

FUSION_RESULT_KEYS = (
    'fusion_wsi_gate_mean',
    'fusion_rna_gate_mean',
    'fusion_gate_std',
)


def _move_data_to_device(data):
    if isinstance(data, (tuple, list)):
        return tuple(item.to(device) if torch.is_tensor(item) else item for item in data)
    return data.to(device)


def _bag_size(data):
    if isinstance(data, (tuple, list)):
        data = data[0]
    if data.dim() == 3 and data.size(0) == 1:
        return data.size(1)
    return data.size(0)


def _is_multimodal(args):
    return getattr(args, 'fusion_mode', None) is not None


def _fold_path(path, fold):
    return path.format(fold=fold)


def _compute_auc(labels, prob, n_classes):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float('nan')

    try:
        if n_classes == 2:
            return roc_auc_score(labels, prob[:, 1])

        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx not in labels:
                continue
            class_labels = binary_labels[:, class_idx]
            if class_labels.min() == class_labels.max():
                continue
            fpr, tpr, _ = roc_curve(class_labels, prob[:, class_idx])
            aucs.append(calc_auc(fpr, tpr))
        return float(np.mean(aucs)) if aucs else float('nan')
    except ValueError:
        return float('nan')


def _fit_multimodal_transform(train_split, val_split, test_split, args):
    if not hasattr(train_split, 'fit_tabular_transform'):
        return None

    transform = train_split.fit_tabular_transform(
        top_n_features=getattr(args, 'tabular_top_n_features', 0)
    )
    for split in (val_split, test_split):
        if split is not None and hasattr(split, 'set_tabular_transform'):
            split.set_tabular_transform(transform)

    print(
        'Fitted tabular transform with {} selected features.'.format(
            len(transform.selected_feature_names)
        )
    )
    return transform


def _save_multimodal_transform(split, output_path):
    if not hasattr(split, 'tabular_transform_dict'):
        return

    payload = split.tabular_transform_dict()
    if payload is None:
        return

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')


def _wandb_log(metrics):
    """Log metrics to wandb if a run is active."""
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics)


def _class_accuracy_metrics(prefix, acc_logger, n_classes):
    metrics = {}
    for class_idx in range(n_classes):
        acc, correct, count = acc_logger.get_summary(class_idx)
        if acc is not None:
            metrics[f'{prefix}/class_{class_idx}_acc'] = acc
        metrics[f'{prefix}/class_{class_idx}_correct'] = int(correct)
        metrics[f'{prefix}/class_{class_idx}_count'] = int(count)
    return metrics


def _scalar_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _extract_fusion_metrics(results_dict):
    if not isinstance(results_dict, dict):
        return {}

    metrics = {}
    for key in FUSION_RESULT_KEYS:
        if key in results_dict:
            metrics[key] = _scalar_float(results_dict[key])
    return metrics


def _collect_fusion_metrics(storage, results_dict):
    for key, value in _extract_fusion_metrics(results_dict).items():
        storage.setdefault(key, []).append(value)


def _summarize_fusion_metrics(prefix, storage):
    metrics = {}
    for key, values in storage.items():
        if values:
            metrics[f'{prefix}/{key}'] = float(np.mean(values))
    return metrics


def _summarize_patient_fusion_metrics(prefix, patient_results):
    storage = {}
    for result in patient_results.values():
        for key in FUSION_RESULT_KEYS:
            if key in result:
                storage.setdefault(key, []).append(float(result[key]))
    return _summarize_fusion_metrics(prefix, storage)

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split) if test_split is not None else 0))
    _fit_multimodal_transform(train_split, val_split, test_split, args)

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if _is_multimodal(args):
        if args.model_type not in ['clam_sb', 'clam_mb']:
            raise ValueError("Multimodal fusion requires a CLAM WSI branch.")
        if not hasattr(train_split, 'tabular_feature_dim'):
            raise ValueError("Multimodal fusion requires a multimodal dataset split.")

        model = CLAMRNAFusion(
            wsi_model_type=args.model_type,
            tabular_input_dim=train_split.tabular_feature_dim,
            tabular_hidden_dim=args.tabular_hidden_dim,
            tabular_num_layers=args.tabular_num_layers,
            fusion_hidden_dim=args.fusion_hidden_dim,
            fusion_mode=args.fusion_mode,
            k_sample=args.B,
            subtyping=args.subtyping,
            **model_dict,
        )

        if getattr(args, 'pretrained_wsi_ckpt', None):
            wsi_ckpt_path = _fold_path(args.pretrained_wsi_ckpt, cur)
            model.load_wsi_checkpoint(wsi_ckpt_path)
            print('Loaded pretrained WSI branch from {}'.format(wsi_ckpt_path))
        if getattr(args, 'freeze_wsi_branch', False):
            model.freeze_wsi_branch()
            print('Frozen WSI branch parameters.')

    elif args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing) if test_split is not None else None
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = args.patience, stop_epoch=5, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    history = []
    history_path = os.path.join(args.results_dir, "fold_{}_history.csv".format(cur))
    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not _is_multimodal(args) and not args.no_inst_cluster:
            train_metrics = train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop, val_metrics = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            train_metrics = train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop, val_metrics = validate(cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)

        history.append({'fold': cur, 'epoch': epoch, **train_metrics, **val_metrics})
        pd.DataFrame(history).to_csv(history_path, index=False)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _save_multimodal_transform(
        train_split,
        os.path.join(args.results_dir, "s_{}_tabular_transform.json".format(cur)),
    )

    val_results, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    final_metrics = {
        'final/val_error': val_error,
        'final/val_auc': val_auc,
        'final/val_accuracy': 1.0 - val_error,
    }
    final_metrics.update(_summarize_patient_fusion_metrics('final/val', val_results))

    if test_loader is not None:
        results_dict, test_error, test_auc, acc_logger = summary(
            model, test_loader, args.n_classes,
            log_heatmaps=getattr(args, 'log_heatmaps', 0))
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
        final_metrics.update(
            {
                'final/test_error': test_error,
                'final/test_auc': test_auc,
                'final/test_accuracy': 1.0 - test_error,
            }
        )
        final_metrics.update(_summarize_patient_fusion_metrics('final/test', results_dict))

        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer and acc is not None:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    else:
        results_dict, test_error, test_auc = {}, val_error, val_auc
        final_metrics.update(
            {
                'final/test_error': test_error,
                'final/test_auc': test_auc,
                'final/test_accuracy': 1.0 - test_error,
            }
        )
        print('No test split available; reporting val metrics as test metrics.')

    if writer:
        for key, value in final_metrics.items():
            writer.add_scalar(key, value, 0)
        writer.close()

    _wandb_log(final_metrics)

    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = _move_data_to_device(data), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), _bag_size(data)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

    metrics = {
        'train/loss': train_loss,
        'train/error': train_error,
        'train/accuracy': 1.0 - train_error,
        'train/clustering_loss': train_inst_loss,
    }
    metrics.update(_class_accuracy_metrics('train', acc_logger, n_classes))
    _wandb_log({'epoch': epoch, **metrics})
    return metrics

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    fusion_metric_values = {}

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = _move_data_to_device(data), label.to(device)

        logits, Y_prob, Y_hat, _, results_dict = model(data)
        _collect_fusion_metrics(fusion_metric_values, results_dict)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), _bag_size(data)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

    metrics = {
        'train/loss': train_loss,
        'train/error': train_error,
        'train/accuracy': 1.0 - train_error,
    }
    metrics.update(_class_accuracy_metrics('train', acc_logger, n_classes))
    metrics.update(_summarize_fusion_metrics('train', fusion_metric_values))
    if writer:
        for key, value in metrics.items():
            if key.startswith('train/fusion_'):
                writer.add_scalar(key, value, epoch)
    _wandb_log({'epoch': epoch, **metrics})
    return metrics

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    fusion_metric_values = {}
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = _move_data_to_device(data), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, results_dict = model(data)
            _collect_fusion_metrics(fusion_metric_values, results_dict)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    auc = _compute_auc(labels, prob, n_classes)
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    metrics = {
        'val/loss': val_loss,
        'val/error': val_error,
        'val/accuracy': 1.0 - val_error,
        'val/auc': auc,
    }
    metrics.update(_class_accuracy_metrics('val', acc_logger, n_classes))
    metrics.update(_summarize_fusion_metrics('val', fusion_metric_values))
    if writer:
        for key, value in metrics.items():
            if key.startswith('val/fusion_'):
                writer.add_scalar(key, value, epoch)
    _wandb_log({'epoch': epoch, **metrics})

    if early_stopping:
        assert results_dir
        # Monitor -auc so the best checkpoint maximises AUC, not minimises loss.
        monitor_value = -auc if np.isfinite(auc) else val_loss
        early_stopping(epoch, monitor_value, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True, metrics

    return False, metrics

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = _move_data_to_device(data), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    auc = _compute_auc(labels, prob, n_classes)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    metrics = {
        'val/loss': val_loss,
        'val/error': val_error,
        'val/accuracy': 1.0 - val_error,
        'val/auc': auc,
        'val/inst_loss': val_inst_loss,
    }
    metrics.update(_class_accuracy_metrics('val', acc_logger, n_classes))
    _wandb_log({'epoch': epoch, **metrics})

    if early_stopping:
        assert results_dir
        # Monitor -auc so the best checkpoint maximises AUC, not minimises loss.
        monitor_value = -auc if np.isfinite(auc) else val_loss
        early_stopping(epoch, monitor_value, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True, metrics

    return False, metrics

def _log_attention_heatmaps(collected, dataset, k):
    """Generate spatial attention scatter plots and log to W&B as a table."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py

    try:
        import wandb
        if wandb.run is None:
            return
    except ImportError:
        return

    correct = [s for s in collected if s['pred'] == s['label']]
    wrong = [s for s in collected if s['pred'] != s['label']]
    correct.sort(key=lambda s: s['confidence'], reverse=True)
    wrong.sort(key=lambda s: s['confidence'], reverse=True)
    selected = correct[:k] + wrong[:k]

    if not selected:
        return

    data_dir = dataset.data_dir
    if isinstance(data_dir, dict):
        data_dir = list(data_dir.values())[0]

    table = wandb.Table(columns=['slide_id', 'pred', 'true', 'correct', 'confidence', 'heatmap'])

    for s in selected:
        h5_path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(s['slide_id']))
        if not os.path.isfile(h5_path):
            continue

        with h5py.File(h5_path, 'r') as f:
            if 'coords_patching' in f:
                coords = f['coords_patching'][:]
            else:
                coords = f['coords'][:].squeeze(0)

        attn = s['attention']
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(coords[:, 0], -coords[:, 1], c=attn, cmap='coolwarm',
                        s=1, edgecolors='none')
        fig.colorbar(sc, ax=ax, label='Attention', shrink=0.8)
        is_correct = s['pred'] == s['label']
        ax.set_title('{}\npred={} true={} p={:.3f}'.format(
            s['slide_id'], s['pred'], s['label'], s['confidence']))
        ax.set_aspect('equal')
        ax.axis('off')
        fig.tight_layout()

        table.add_data(s['slide_id'], s['pred'], s['label'],
                       is_correct, round(s['confidence'], 4),
                       wandb.Image(fig))
        plt.close(fig)

    wandb.log({'attention_heatmaps': table})


def summary(model, loader, n_classes, log_heatmaps=0):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    attn_collected = []

    for batch_idx, (data, label) in enumerate(loader):
        data, label = _move_data_to_device(data), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, A_raw, results_dict = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        slide_result = {
            'slide_id': np.array(slide_id),
            'prob': probs,
            'label': label.item(),
        }
        slide_result.update(_extract_fusion_metrics(results_dict))
        patient_results.update({slide_id: slide_result})
        error = calculate_error(Y_hat, label)
        test_error += error

        if log_heatmaps > 0:
            pred_cls = Y_hat.item()
            if A_raw.dim() == 2 and A_raw.size(0) > 1:
                attn = A_raw[pred_cls]
            else:
                attn = A_raw.squeeze(0)
            attn = torch.nn.functional.softmax(attn, dim=0).cpu().numpy()
            attn_collected.append({
                'slide_id': slide_id,
                'pred': pred_cls,
                'label': label.item(),
                'confidence': float(probs.flatten()[pred_cls]),
                'attention': attn,
            })

    test_error /= len(loader)

    auc = _compute_auc(all_labels, all_probs, n_classes)

    if log_heatmaps > 0 and attn_collected:
        _log_attention_heatmaps(attn_collected, loader.dataset, log_heatmaps)

    return patient_results, test_error, auc, acc_logger
