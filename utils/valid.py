import time
import torch
import torch.nn.parallel
import torch.utils.data
from metrics.myAUC import AUCMeter
from utils.utils import AverageMeter
import sklearn.metrics
import numpy as np


def _tta_augments(x):
    """Generate TTA variants: original + 3 rotations + hflip + vflip + hflip+vflip = 7 views."""
    views = [x]
    views.append(torch.rot90(x, 1, [2, 3]))
    views.append(torch.rot90(x, 2, [2, 3]))
    views.append(torch.rot90(x, 3, [2, 3]))
    views.append(torch.flip(x, [3]))
    views.append(torch.flip(x, [2]))
    views.append(torch.flip(torch.flip(x, [2]), [3]))
    return views


def validate(val_loader, model, criterion_cls, criterion_gcn, args):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_gcn = AverageMeter()
    # switch to evaluate mode
    model.eval()
    eval_auc = AUCMeter()
    end = time.time()

    eval_auc_init = AUCMeter()

    target_old = np.zeros((len(val_loader.dataset), 1))
    pred_old = np.zeros((len(val_loader.dataset), 2))
    pred_init_old = np.zeros((len(val_loader.dataset), 2))
    batch_begin = 0


    device = next(model.parameters()).device

    # with torch.no_grad():
    for i, (input, target, A_1, gcn_target, m_id, h_id)  in enumerate(val_loader):

        target = target.to(device, non_blocking=True)
        target_var = target.to(device, non_blocking=True)
        input_var = input.to(device, non_blocking=True)
        gcn_target_var = gcn_target.to(device, non_blocking=True)
        m_id = m_id.to(device, non_blocking=True)
        h_id = h_id.to(device, non_blocking=True)
        A_1 = A_1.to(device, non_blocking=True).float()

        use_tta = bool(int(getattr(args, 'tta', 0)))

        if use_tta:
            views = _tta_augments(input_var)
            logits_acc = torch.zeros_like(torch.empty(input_var.size(0), 2, device=device))
            logits_init_acc = torch.zeros_like(logits_acc)
            for v in views:
                out_v = model(v, m_id, h_id, A_1, gcn_target_var, criterion_gcn, 'val')
                logits_acc += out_v[-1]
                logits_init_acc += out_v[-2]
            output_cls = logits_acc / len(views)
            output_cls_init = logits_init_acc / len(views)
            output = model(input_var, m_id, h_id, A_1, gcn_target_var, criterion_gcn, 'val')
            output = list(output)
            output[-1] = output_cls
            output[-2] = output_cls_init
        else:
            output = model(input_var, m_id, h_id, A_1, gcn_target_var, criterion_gcn, 'val')

        loss_cls = criterion_cls(output[-1], target_var)
        loss_gcn = criterion_gcn(output[-3], gcn_target_var)

        needata = output[-1]
        _, predi = needata.topk(1, 1, True, True)
        predi = predi.view(len(predi))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses_gcn.update(loss_gcn.item(), input.size(0))
        eval_auc.update(predi, target_var)
        # top1.update(acc1[0], input[0].size(0))
        # top5.update(acc5[0], input[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        target_old[batch_begin:batch_begin + input_var.size(0)] = target_var.unsqueeze(1).detach().cpu().numpy()
        pred_old[batch_begin:batch_begin + input_var.size(0)] = output[-1].detach().cpu().numpy()
        pred_init_old[batch_begin:batch_begin + input_var.size(0)] = output[-2].detach().cpu().numpy()
        batch_begin += input_var.size(0)
        
        # AUC_init
        needata_init = output[-2]
        _, predi_init = needata_init.topk(1, 1, True, True)
        predi_init = predi_init.view(len(predi_init))
        eval_auc_init.update(predi_init, target_var)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                    'AUC {AUC}'.format(
                i, len(val_loader), batch_time=batch_time, loss_cls=losses_cls, AUC=eval_auc.get_auc()))


    # print(' * Acc@1 {top1.avg:.3f} AUC {AUC}'
    #       .format(top1=top1, AUC=eval_auc.get_auc()))
    
    # sensi, speci = confusion_metrics(y_pred_all, y_true_all)

    # return top1.avg, eval_auc.get_auc(), sensi, speci, eval_auc_init.get_auc()
    try:
        AUC_old = sklearn.metrics.roc_auc_score(target_old, pred_old[:, 1])
    except ValueError:
        AUC_old = float('nan')
    try:
        AUC_init_old = sklearn.metrics.roc_auc_score(target_old, pred_init_old[:, 1])
    except ValueError:
        AUC_init_old = float('nan')
    acc_old = sklearn.metrics.accuracy_score(target_old, np.argmax(pred_old, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target_old, np.argmax(pred_old, axis=1), labels=[0, 1])
    specificity_old = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1]) if (cm_minus[0, 0] + cm_minus[0, 1]) > 0 else 0
    sensitivity_old = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1]) if (cm_minus[1, 0] + cm_minus[1, 1]) > 0 else 0
    # print("CM in sklearn: AUC acc sensi speci", AUC_old, acc_old, sensitivity_old, specificity_old)

    # return top1.avg, eval_auc.get_auc(), sensi, speci, eval_auc_init.get_auc()
    return acc_old, AUC_old, sensitivity_old, specificity_old, AUC_init_old
