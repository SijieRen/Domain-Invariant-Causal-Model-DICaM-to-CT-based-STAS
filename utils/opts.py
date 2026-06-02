import argparse
# import torchvision.models as models


# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))

def parse_opt(inputs=None):
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#     parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                         choices=model_names,
#                         help='model architecture: ' +
#                              ' | '.join(model_names) +
#                              ' (default: resnet18)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit',
                    #     choices=model_names,
                        help='model architecture: ' +
                             ' | '.join('vit') +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size for single-GPU training (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_weights_only', default=0, type=int,
                        help='when set to 1, load checkpoint weights but do not restore optimizer state or epoch')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use ImageNet pre-trained backbone')
    parser.add_argument('--backbone', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'efficientnet_b0'],
                        help='backbone architecture for both encoders')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--saved', default=None, type=str,
                        help='parameter to use.')
    parser.add_argument('--para_cls', default=1.0, type=float,
                        help='parameter to clasification.')
    parser.add_argument('--para_gcn', default=0.5, type=float,
                        help='parameter to gcn.')
    parser.add_argument('--para_recon', default=1.0, type=float,
                        help='parameter to reconstruction.')
    parser.add_argument('--para_kld', default=1.0, type=float,
                        help='parameter to kld.')
    parser.add_argument('--beta_max', default=0.05, type=float,
                        help='max KL warmup coefficient.')
    parser.add_argument('--kl_warmup_epochs', default=20, type=int,
                        help='number of epochs for KL warmup.')
    parser.add_argument('--decouple_weight', default=0.01, type=float,
                        help='weight for latent decoupling regularization.')
    parser.add_argument('--decouple_warmup_epochs', default=20, type=int,
                        help='number of epochs for decouple warmup.')
    parser.add_argument('--ssim_weight', default=0.1, type=float,
                        help='weight for SSIM term in reconstruction loss.')
    parser.add_argument('--grad_weight', default=0.15, type=float,
                        help='weight for gradient consistency term in reconstruction loss.')
    parser.add_argument('--hf_weight', default=0.0, type=float,
                        help='weight for high-frequency Laplacian detail term in reconstruction loss.')
    parser.add_argument('--ms_recon_weight', default=0.2, type=float,
                        help='weight for multi-scale reconstruction consistency.')
    parser.add_argument('--para_kld_dom', default=1.0, type=float,
                        help='weight for domain-conditional KL divergence.')
    parser.add_argument('--para_recon_val', default=0.1, type=float,
                        help='reconstruction loss weight during test-time latent optimization.')
    parser.add_argument('--para_cli', default=0.1, type=float,
                        help='weight for clinical prediction loss.')
    parser.add_argument('--task_warmup_epochs', default=15, type=int,
                        help='number of epochs to warm up downstream losses after reconstruction stabilizes.')
    parser.add_argument('--component_warmup_epochs', default=25, type=int,
                        help='number of epochs to warm up component-level interpretability losses.')
    parser.add_argument('--latent_noise_warmup_epochs', default=20, type=int,
                        help='number of epochs before reconstruction fully uses stochastic latent samples.')
    parser.add_argument('--latent_drop_prob', default=0.1, type=float,
                        help='drop probability for latent chunk intervention during training.')
    parser.add_argument('--component_consistency_weight', default=0.2, type=float,
                        help='weight for matching the sum of factor reconstructions to the fused reconstruction.')
    parser.add_argument('--component_overlap_weight', default=0.1, type=float,
                        help='weight for discouraging different factor reconstructions from attending to the same region.')
    parser.add_argument('--residual_sparsity_weight', default=0.02, type=float,
                        help='weight for keeping the residual factor visually compact.')
    parser.add_argument('--component_skip_strength', default=0.14, type=float,
                        help='maximum skip strength used by single-factor visual reconstructions.')
    parser.add_argument('--use_class_weights', default=1, type=int,
                        help='when set to 1, build class-balanced weights for classification loss from the training set.')
    parser.add_argument('--cls_label_smoothing', default=0.0, type=float,
                        help='label smoothing applied to classification cross entropy.')
    parser.add_argument('--cls_use_mu', default=1, type=int,
                        help='when set to 1, use deterministic latent mean for the main classification head instead of sampled latent.')
    parser.add_argument('--aux_use_mu', default=1, type=int,
                        help='when set to 1, use deterministic latent mean for domain and clinical auxiliary heads.')
    parser.add_argument('--classifier_latent_drop_prob', default=0.0, type=float,
                        help='drop probability applied only to classifier-side latent chunks.')
    parser.add_argument('--freeze_backbone', default=0, type=int,
                        help='when set to 1, freeze pretrained backbone in both encoders (only train MLP heads, decoders, classifiers).')
    parser.add_argument('--freeze_rec_decoder', default=0, type=int,
                        help='when set to 1, freeze the main reconstruction decoder during fine-tuning.')
    parser.add_argument('--freeze_visual_decoder', default=0, type=int,
                        help='when set to 1, freeze the factor-visualization decoder during fine-tuning.')
    parser.add_argument('--freeze_input_skip_encoder', default=0, type=int,
                        help='when set to 1, freeze the input skip encoder during fine-tuning.')
    parser.add_argument('--num_domains', default=4, type=int,
                        help='number of domain classes for domain head.')
    parser.add_argument('--multilabel', default=True, type=bool,
                        help='multilabel training.')
    parser.add_argument('--adam', default=True, type=bool,
                        help='optimizer.')
    parser.add_argument('--if_transformer', default="res", type=str,
                        help='if we use transformer as backnone structure.')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='if we use transformer as backnone structure.')
    parser.add_argument('--pool', default="cls", type=str,
                        help='pool operation in transformer.')
    parser.add_argument('--if_randomTest', default=0, type=int,
                        help='use the mask only data or not')

    parser.add_argument('--val_ep', default=20, type=int,
                        help='number of optimization steps during test-time latent refinement.')
    parser.add_argument('--lr2', default=0.00001, type=float,
                        help='learning rate for test-time latent optimization.')
    parser.add_argument('--wd2', default=0.0001, type=float,
                        help='weight decay for test-time latent optimization.')
    
    parser.add_argument('--lr_decay', default=0.00001, type=float,
                        help='lr_decay or not')
    parser.add_argument('--lr_controller', default=0.0001, type=float,
                        help='lr decay period')
    

    parser.add_argument('--rec_loss_check', default="l1_ssim", type=str,
                        help='catagaty of rec_loss')
    parser.add_argument('--data_root', default='../dataset', type=str,
                        help='数据根目录，用于拼接 Excel 中的相对路径。默认为 ../dataset（相对于 Darmo_stas_v3）')
    parser.add_argument('--dataset_excel', default='../data/STAS_dataset_2D_NoMask_randomTest82_20260519_unique.xlsx', type=str,
                        help='optional excel file path overriding the default train/val split file')
    parser.add_argument('--gen_data_ratio', default=0.0, type=float,
                        help='ratio of generated samples relative to real train size; 0 disables mixing.')
    parser.add_argument('--gen_data_xls', default='', type=str,
                        help='Excel file with generated sample metadata (sheets: batch_0..batch_9).')
    parser.add_argument('--gen_data_seed', default=1234, type=int,
                        help='random seed for sampling generated data.')
    parser.add_argument('--tta', default=0, type=int,
                        help='when set to 1, enable test-time augmentation (flip + rotate) and average predictions.')
    parser.add_argument('--ensemble_ckpts', nargs='+', default=[],
                        help='multiple checkpoint paths for ensemble inference.')

    # --- Cosine LR schedule (backported from V7) ---
    parser.add_argument('--lr_schedule', default='decay', type=str,
                        choices=['decay', 'cosine'],
                        help='learning rate schedule: decay (step decay, original V5/V6) or cosine (cosine annealing).')

    args = parser.parse_args()

    return args
