import time
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from utils.utils import AverageMeter
import numpy as np
import sklearn.metrics
from utils.losses import component_consistency_loss, component_overlap_loss, residual_sparsity_loss


def _warmup_coeff(epoch, warmup_epochs, max_value=1.0):
    if warmup_epochs <= 0:
        return max_value
    return max_value * min(1.0, float(epoch + 1) / float(warmup_epochs))


def _cross_covariance_loss(z1, z2):
    z1 = z1 - z1.mean(dim=0, keepdim=True)
    z2 = z2 - z2.mean(dim=0, keepdim=True)
    denom = max(1, z1.size(0) - 1)
    cov = (z1.transpose(0, 1) @ z2) / denom
    return torch.norm(cov, p='fro')


def _decouple_loss_from_mu(mu, chunk_dims):
    chunks = []
    start = 0
    for chunk_dim in chunk_dims:
        end = start + chunk_dim
        chunks.append(mu[:, start:end])
        start = end

    loss = mu.new_tensor(0.0)
    pair_count = 0
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            loss = loss + _cross_covariance_loss(chunks[i], chunks[j])
            pair_count += 1

    if pair_count == 0:
        return loss
    return loss / pair_count


def train(train_loader, model, criterion_cls, criterion_gcn, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses_recon = AverageMeter()
    losses_kl_inv = AverageMeter()
    losses_kl_dom = AverageMeter()
    losses_cls = AverageMeter()
    losses_gcn = AverageMeter()
    losses_decouple = AverageMeter()
    losses_cli = AverageMeter()
    losses_comp_consistency = AverageMeter()
    losses_comp_overlap = AverageMeter()
    losses_residual_sparse = AverageMeter()

    model.train()
    end = time.time()

    target_old = np.zeros((len(train_loader.dataset), 1))
    pred_old = np.zeros((len(train_loader.dataset), 2))
    batch_begin = 0

    beta = _warmup_coeff(epoch, args.kl_warmup_epochs, args.beta_max)
    decouple_coeff = _warmup_coeff(epoch, args.decouple_warmup_epochs, 1.0)
    task_coeff = _warmup_coeff(epoch, args.task_warmup_epochs, 1.0)
    component_coeff = _warmup_coeff(epoch, args.component_warmup_epochs, 1.0)
    latent_noise_coeff = _warmup_coeff(epoch, args.latent_noise_warmup_epochs, 1.0)

    device = next(model.parameters()).device

    for i, (input, target, A_1, gcn_target, m_id, h_id) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target_var = target.to(device, non_blocking=True)
        input_var = input.to(device, non_blocking=True)
        m_id = m_id.to(device, non_blocking=True).long()
        h_id = h_id.to(device, non_blocking=True)
        A_1 = A_1.to(device, non_blocking=True).float()
        gcn_target_var = gcn_target.to(device, non_blocking=True)

        B = input_var.size(0)

        output = model(
            input_var, m_id, 0, A_1, gcn_target_var, criterion_gcn,
            'train', latent_noise_coeff=latent_noise_coeff,
        )
        a1_pred = output[8]
        gcn_x = output[9]
        cls_pred = output[10]
        component_recons = [output[11], output[12], output[13], output[14]]

        loss_cls = criterion_cls(cls_pred, target_var)
        loss_gcn = criterion_gcn(gcn_x, gcn_target_var)
        loss_cli = F.smooth_l1_loss(a1_pred, A_1)

        mu_inv = output[2]
        mu_dom = output[4]
        mu_dis = mu_inv[:, :model.dis_dim]
        mu_cli = mu_inv[:, model.dis_dim:model.dis_dim + model.cli_dim]
        mu_res = mu_inv[:, model.dis_dim + model.cli_dim:]
        full_mu = torch.cat([mu_dis, mu_dom, mu_cli, mu_res], dim=1)
        loss_decouple = _decouple_loss_from_mu(
            full_mu,
            [model.dis_dim, model.dom_dim, model.cli_dim, model.res_dim],
        )

        loss_comp_consistency = component_consistency_loss(output[0], component_recons)
        loss_comp_overlap = component_overlap_loss(component_recons)
        loss_residual_sparse = residual_sparsity_loss(component_recons[-1])

        loss_vae_dict = model.loss_function(
            *output[:8],
            M_N=int(args.batch_size) / len(train_loader.dataset.samples),
            beta=beta,
        )

        loss = (
            loss_vae_dict['loss']
            + task_coeff * args.para_cls * loss_cls
            + task_coeff * args.para_gcn * loss_gcn
            + task_coeff * args.para_cli * loss_cli
            + decouple_coeff * args.decouple_weight * loss_decouple
            + component_coeff * args.component_consistency_weight * loss_comp_consistency
            + component_coeff * args.component_overlap_weight * loss_comp_overlap
            + component_coeff * args.residual_sparsity_weight * loss_residual_sparse
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), B)
        losses_recon.update(loss_vae_dict['Reconstruction_Loss'].item(), B)
        losses_kl_inv.update(loss_vae_dict['KLD_inv'].item(), B)
        losses_kl_dom.update(loss_vae_dict['KLD_dom'].item(), B)
        losses_cls.update(loss_cls.item(), B)
        losses_gcn.update(loss_gcn.item(), B)
        losses_cli.update(loss_cli.item(), B)
        losses_decouple.update(loss_decouple.item(), B)
        losses_comp_consistency.update(loss_comp_consistency.item(), B)
        losses_comp_overlap.update(loss_comp_overlap.item(), B)
        losses_residual_sparse.update(loss_residual_sparse.item(), B)

        target_old[batch_begin:batch_begin + B] = target_var.unsqueeze(1).detach().cpu().numpy()
        pred_old[batch_begin:batch_begin + B] = cls_pred.detach().cpu().numpy()
        batch_begin += B

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                'Loss_gcn {loss_gcn.val:.4f} ({loss_gcn.avg:.4f})\t'
                'Loss_cli {loss_cli.val:.4f} ({loss_cli.avg:.4f})\t'
                'Loss_recon {loss_recon.val:.4f} ({loss_recon.avg:.4f})\t'
                'KL_inv {kl_inv.val:.4f} ({kl_inv.avg:.4f})\t'
                'KL_dom {kl_dom.val:.4f} ({kl_dom.avg:.4f})\t'
                'Loss_decouple {loss_decouple.val:.4f} ({loss_decouple.avg:.4f})\t'
                'Loss_comp {loss_comp.val:.4f} ({loss_comp.avg:.4f})\t'
                'Loss_overlap {loss_overlap.val:.4f} ({loss_overlap.avg:.4f})\t'
                'Loss_res {loss_res.val:.4f} ({loss_res.avg:.4f})\t'
                'beta {beta:.4f}\t'
                'task {task_coeff:.4f}\t'
                'comp {component_coeff:.4f}\t'
                'noise {latent_noise_coeff:.4f}'.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_cls=losses_cls,
                    loss_gcn=losses_gcn,
                    loss_cli=losses_cli,
                    loss_recon=losses_recon,
                    kl_inv=losses_kl_inv,
                    kl_dom=losses_kl_dom,
                    loss_decouple=losses_decouple,
                    loss_comp=losses_comp_consistency,
                    loss_overlap=losses_comp_overlap,
                    loss_res=losses_residual_sparse,
                    beta=beta,
                    task_coeff=task_coeff,
                    component_coeff=component_coeff,
                    latent_noise_coeff=latent_noise_coeff,
                )
            )

    target_old = target_old[:batch_begin]
    pred_old = pred_old[:batch_begin]
    AUC_old = sklearn.metrics.roc_auc_score(target_old, pred_old[:, 1])
    acc_old = sklearn.metrics.accuracy_score(target_old, np.argmax(pred_old, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target_old, np.argmax(pred_old, axis=1))
    specificity_old = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1])
    sensitivity_old = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1])

    return acc_old, AUC_old, sensitivity_old, specificity_old
