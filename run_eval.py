# -*- coding: utf-8 -*-
"""Inference script — compute metrics (AUC, ACC, sensitivity, specificity,
precision, NPV) on train / test_inter / test_exter / test_exter2 splits.

No visualisation is performed; only a metrics.json per fold is saved.
"""
import json
import os
import random
import time
import warnings

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

from utils.opts import parse_opt
from utils.baseTrainer import _build_cls_criterion
from utils.valid import _tta_augments
from utils.utils import AverageMeter
from data.STAS_dataloader import Dataloader_2D
from models.Maintrainer_stas import Model_STAS_2D


def _compute_metrics(target, pred_logits):
    """Compute classification metrics from targets and logits."""
    pred_labels = np.argmax(pred_logits, axis=1)
    prob_pos = F.softmax(torch.tensor(pred_logits, dtype=torch.float32), dim=1)[:, 1].numpy()

    try:
        auc = sklearn.metrics.roc_auc_score(target, prob_pos)
    except ValueError:
        auc = float('nan')

    acc = sklearn.metrics.accuracy_score(target, pred_labels)
    cm = sklearn.metrics.confusion_matrix(target, pred_labels, labels=[0, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

    return {
        'auc': float(auc),
        'acc': float(acc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
    }


def evaluate_fold(val_loader, model, criterion_cls, criterion_gcn, args, fold_name, save_root):
    """Run inference on one fold and save metrics.json."""
    batch_time = AverageMeter()
    model.eval()

    target_all = np.zeros((len(val_loader.dataset), 1))
    pred_all = np.zeros((len(val_loader.dataset), 2))
    pred_init_all = np.zeros((len(val_loader.dataset), 2))
    rec_errors = []
    batch_begin = 0

    device = next(model.parameters()).device
    end = time.time()

    for i, (input_data, target, A_1, gcn_target, m_id, h_id) in enumerate(val_loader):
        target_var = target.to(device, non_blocking=True)
        input_var = input_data.to(device, non_blocking=True)
        gcn_target_var = gcn_target.to(device, non_blocking=True)
        m_id = m_id.to(device, non_blocking=True)
        h_id = h_id.to(device, non_blocking=True)
        A_1 = A_1.to(device, non_blocking=True).float()

        use_tta = bool(int(getattr(args, 'tta', 0)))

        if use_tta:
            views = _tta_augments(input_var)
            logits_acc = torch.zeros(input_var.size(0), 2, device=device)
            logits_init_acc = torch.zeros_like(logits_acc)
            for v in views:
                out_v = model(v, m_id, h_id, A_1, gcn_target_var, criterion_gcn, 'val')
                logits_acc += out_v[-1]
                logits_init_acc += out_v[-2]
            output = model(input_var, m_id, h_id, A_1, gcn_target_var, criterion_gcn, 'val')
            output = list(output)
            output[-1] = logits_acc / len(views)
            output[-2] = logits_init_acc / len(views)
        else:
            output = model(input_var, m_id, h_id, A_1, gcn_target_var, criterion_gcn, 'val')

        with torch.no_grad():
            rec_errors.append(F.mse_loss(output[1], output[0], reduction='mean').item())

        B = input_data.size(0)
        target_all[batch_begin:batch_begin + B] = target_var.detach().cpu().numpy().reshape(-1, 1)
        pred_all[batch_begin:batch_begin + B] = output[-1].detach().cpu().numpy()
        pred_init_all[batch_begin:batch_begin + B] = output[-2].detach().cpu().numpy()
        batch_begin += B

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Eval [{i}/{len(val_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

    target_all = target_all[:batch_begin]
    pred_all = pred_all[:batch_begin]
    pred_init_all = pred_init_all[:batch_begin]

    metrics_final = _compute_metrics(target_all, pred_all)
    metrics_init = _compute_metrics(target_all, pred_init_all)

    fold_dir = os.path.join(save_root, fold_name)
    os.makedirs(fold_dir, exist_ok=True)
    json_path = os.path.join(fold_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump({
            'fold': fold_name,
            'num_samples': int(batch_begin),
            'tta': bool(int(getattr(args, 'tta', 0))),
            'metrics_final': metrics_final,
            'metrics_init': metrics_init,
            'rec_mse_mean': round(float(np.mean(rec_errors)), 6),
            'rec_mse_std': round(float(np.std(rec_errors)), 6),
        }, f, indent=2, ensure_ascii=False)

    print(f'\n[{fold_name}] metrics saved to {json_path}')
    print(f'[{fold_name}] metrics_final: {metrics_final}')
    print(f'[{fold_name}] metrics_init:  {metrics_init}')
    print(f'[{fold_name}] rec MSE: {np.mean(rec_errors):.6f} +/- {np.std(rec_errors):.6f}')

    return metrics_final, metrics_init


def main():
    args = parse_opt()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            'You have chosen to seed training. This will turn on the CUDNN '
            'deterministic setting, which can slow down your training '
            'considerably! You may see unexpected behavior when restarting '
            'from checkpoints.'
        )

    if args.gpu is None:
        args.gpu = 0

    # Device
    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Model
    model = Model_STAS_2D(1, 64, 'adjacency_matrix.pkl', args)

    if not args.resume:
        raise ValueError('--resume is required for evaluation.')
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f'No checkpoint found at {args.resume}')

    print(f"=> loading checkpoint '{args.resume}'")
    checkpoint = torch.load(args.resume, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'})")

    model = model.to(device)
    model.eval()

    if args.seed is None:
        cudnn.benchmark = True

    # Data
    dataset_dir = args.dataset_excel
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = Dataloader_2D(dataset_dir, 'train', eval_transform, args)
    criterion_cls = _build_cls_criterion(args, train_dataset, device)
    criterion_gcn = (nn.MultiLabelSoftMarginLoss().to(device) if args.multilabel
                     else nn.CrossEntropyLoss().to(device))

    folds = ['train', 'test_inter', 'test_exter', 'test_exter2']
    loaders = {}
    for fold in folds:
        ds = train_dataset if fold == 'train' else Dataloader_2D(dataset_dir, fold, eval_transform, args)
        loaders[fold] = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    save_root = './' + str(args.saved) + '_res/eval'

    results = {}
    for fold in folds:
        metrics_final, metrics_init = evaluate_fold(
            loaders[fold], model, criterion_cls, criterion_gcn, args, fold, save_root)
        results[fold] = (metrics_final, metrics_init)

    # Summary
    print('\n' + '*_* ' * 20)
    for fold in folds:
        mf = results[fold][0]
        mi = results[fold][1]
        print(f"{fold.ljust(12)} : AUC {mf['auc']:.4f}, sensitivity {mf['sensitivity']:.4f}, "
              f"specificity {mf['specificity']:.4f}, acc {mf['acc']:.4f}. AUC_init {mi['auc']:.4f}.")
    print('^_^ ' * 20)


if __name__ == '__main__':
    main()
