
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from data.STAS_dataloader import Dataloader_2D
from .utils import generate_random_sampler_for_train
from .train import train
from .valid import validate
from .utils import adjust_learning_rate
from .save_checkpoint import save_checkpoint
from models.Maintrainer_stas import Model_STAS_2D
from data.gen_mix import load_generated_rows, extend_dataset
import torchvision.transforms as transforms
import copy
import numpy as np


def _build_cls_criterion(args, train_dataset, device):
    class_weights = None
    if int(args.use_class_weights):
        labels = np.asarray(train_dataset.get_labels(), dtype=np.int64)
        class_counts = np.bincount(labels, minlength=2).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(args.cls_label_smoothing),
    ).to(device)


def _set_module_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def _apply_finetune_freeze(model, args):
    if getattr(args, 'freeze_backbone', 0):
        if hasattr(model, 'Resnetencoder') and hasattr(model.Resnetencoder, 'backbone'):
            _set_module_trainable(model.Resnetencoder.backbone, False)
        if hasattr(model, 'Resnetencoder2') and hasattr(model.Resnetencoder2, 'backbone'):
            _set_module_trainable(model.Resnetencoder2.backbone, False)
    if getattr(args, 'freeze_rec_decoder', 0) and hasattr(model, 'dec_x'):
        _set_module_trainable(model.dec_x, False)
    if getattr(args, 'freeze_visual_decoder', 0) and hasattr(model, 'visual_dec_x'):
        _set_module_trainable(model.visual_dec_x, False)
    if getattr(args, 'freeze_input_skip_encoder', 0) and hasattr(model, 'input_skip_encoder'):
        _set_module_trainable(model.input_skip_encoder, False)


def _build_optimizer(model, args):
    params = [param for param in model.parameters() if param.requires_grad]
    if args.adam:
        return torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    return torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def _build_lr_scheduler(optimizer, args):
    """Build cosine annealing scheduler if requested (backported from V7)."""
    if getattr(args, 'lr_schedule', 'decay') == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    return None



def main_worker(gpu, args):
    args.gpu = gpu

    if torch.cuda.is_available() and args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    backbone_name = getattr(args, 'backbone', 'resnet18')
    if args.pretrained:
        print("=> using pre-trained backbone '{}'".format(backbone_name))
    else:
        print("=> creating model without pretraining")

    # Pretrained weights are loaded inside backbone_factory automatically
    model = Model_STAS_2D(1, 64, "adjacency_matrix.pkl", args)
    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    model = model.to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            if int(getattr(args, 'resume_weights_only', 0)):
                print("=> loaded weights only from '{}'".format(args.resume))
            else:
                args.start_epoch = checkpoint.get('epoch', 0)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint.get('epoch', 'unknown')))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    _apply_finetune_freeze(model, args)
    optimizer = _build_optimizer(model, args)
    lr_scheduler = _build_lr_scheduler(optimizer, args)
    if args.resume and os.path.isfile(args.resume) and not int(getattr(args, 'resume_weights_only', 0)):
        checkpoint = torch.load(args.resume)
        if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.seed is None:
        cudnn.benchmark = True

    if args.dataset_excel:
        dataset_dir = args.dataset_excel
    elif args.if_randomTest == 1:
        dataset_dir = "data/STAS_dataset_2D_maskOnly_randomTest82_20250728.xls"
    elif args.if_randomTest == 2:
        dataset_dir = "data/STAS2d-full-attribute-noExter-1226.xls"
    else:
        dataset_dir = "data/STAS_dataset_2D_maskOnly_machineID_20250728.xls"

    train_transform = transforms.Compose([
        transforms.Resize((140, 140)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    train_dataset = Dataloader_2D(dataset_dir, "train", train_transform, args)

    gen_ratio = getattr(args, 'gen_data_ratio', 0.0)
    if gen_ratio > 0:
        gen_df, gen_info = load_generated_rows(
            getattr(args, 'gen_data_xls', ''),
            gen_ratio,
            len(train_dataset),
            seed=int(getattr(args, 'gen_data_seed', 1234)),
        )
        if gen_df is not None:
            gen_dataset = Dataloader_2D(
                dataset_dir, "train", train_transform, args, dataframe=gen_df)
            extend_dataset(train_dataset, gen_dataset)

    bac_sampler = generate_random_sampler_for_train(torch_dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, batch_sampler=bac_sampler)

    if not args.multilabel:
        criterion_cls = _build_cls_criterion(args, train_dataset, device)
        criterion_gcn = nn.CrossEntropyLoss().to(device)
    else:
        criterion_cls = _build_cls_criterion(args, train_dataset, device)
        criterion_gcn = nn.MultiLabelSoftMarginLoss().to(device)

    val_loader_inter = torch.utils.data.DataLoader(
        Dataloader_2D(dataset_dir,
                      "test_inter",
                      transforms.Compose([
                          transforms.Resize((128, 128)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5], std=[0.5]),
                      ]),
                      args),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader_exter = torch.utils.data.DataLoader(
        Dataloader_2D(dataset_dir,
                      "test_exter",
                      transforms.Compose([
                          transforms.Resize((128, 128)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5], std=[0.5]),
                      ]),
                      args),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    checkpoint_root = './' + str(args.saved) + '_' + str(args.if_transformer) + '/checkpoints'

    best_auc_inter = 0
    best_auc_inter_ep = 0
    best_sensi_inter = 0
    best_speci_inter = 0
    best_acc_inter = 0

    best_auc_exter = 0
    best_auc_exter_ep = 0

    try:
        for epoch in range(args.start_epoch, args.epochs):
            # LR schedule: cosine annealing or legacy step decay
            if lr_scheduler is not None:
                lr_scheduler.step()
            else:
                adjust_learning_rate(optimizer, epoch, args)

            acc_train, auc_train, sensi_train, speci_train = train(
                train_loader, model, criterion_cls, criterion_gcn, optimizer, epoch, args)

            acc_inter, auc_inter, sensi_inter, speci_inter, auc_inter_init = validate(
                val_loader_inter, model, criterion_cls, criterion_gcn, args)
            acc_exter, auc_exter, sensi_exter, speci_exter, auc_exter_init = validate(
                val_loader_exter, model, criterion_cls, criterion_gcn, args)

            # --- Inter best AUC (only checkpoint saved) ---
            if auc_inter > best_auc_inter:
                best_auc_inter = auc_inter
                best_acc_inter = acc_inter
                best_sensi_inter = sensi_inter
                best_speci_inter = speci_inter
                best_auc_inter_ep = epoch
                print("Inter_AUC*" * 6)
                print('Best Inter AUC: %0.4f, sensitivity: %0.4f, specificity %0.4f, acc: %0.4f, at ep: %d.'
                      % (best_auc_inter, best_sensi_inter, best_speci_inter, best_acc_inter, epoch))
                print("Inter_AUC*" * 6)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'acc': acc_inter,
                    'auc': best_auc_inter,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint_root, 'best_inter_auc.pth.tar')

            # --- Exter: track best but do not save ---
            if auc_exter > best_auc_exter:
                best_auc_exter = auc_exter
                best_auc_exter_ep = epoch
                print("Exter*" * 6)
                print('Best Exter AUC: %0.4f, sensitivity: %0.4f, specificity %0.4f, acc: %0.4f, at ep: %d.'
                      % (auc_exter, sensi_exter, speci_exter, acc_exter, epoch))
                print("Exter*" * 6)

    finally:
        print("*_* " * 20)
        print('Train  last ep : AUC %0.4f, sensitivity %0.4f, specificity %0.4f, acc %0.4f'
              % (auc_train, sensi_train, speci_train, acc_train))
        print('Inter  best AUC: %0.4f (ep %d), sensitivity %0.4f, specificity %0.4f, acc %0.4f'
              % (best_auc_inter, best_auc_inter_ep, best_sensi_inter, best_speci_inter, best_acc_inter))
        print('Exter  best AUC: %0.4f (ep %d)'
              % (best_auc_exter, best_auc_exter_ep))
        print("^_^ " * 20)
