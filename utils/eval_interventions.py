import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sklearn.metrics

from data.STAS_dataloader import Dataloader_2D
from models.Maintrainer_stas import Model_STAS_2D
from .opts import parse_opt


def _get_dataset_path(args):
    if args.if_randomTest == 1:
        return "data/STAS_dataset_2D_maskOnly_randomTest82_20250728.xls"
    if args.if_randomTest == 2:
        return "data/STAS2d-full-attribute-noExter-1226.xls"
    return "data/STAS_dataset_2D_maskOnly_machineID_20250728.xls"


def _build_loader(dataset_dir, fold, args):
    ds = Dataloader_2D(
        dataset_dir,
        fold,
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]),
        args,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate_one_mode(loader, model, criterion_gcn, mode):
    y_true = []
    y_prob = []
    a2_true = []
    a2_pred = []
    rec_l1 = []

    for input, target, a1, gcn_target, m_id, h_id in loader:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        a1 = a1.cuda(non_blocking=True).float()
        gcn_target = gcn_target.cuda(non_blocking=True)
        m_id = m_id.cuda(non_blocking=True)
        h_id = h_id.cuda(non_blocking=True)

        rec, cls, a2_hat, _ = model(
            input,
            m_id,
            h_id,
            a1,
            gcn_target,
            criterion_gcn,
            "intervene",
            intervention_chunk=mode,
        )

        y_true.append(target.detach().cpu().numpy())
        y_prob.append(cls[:, 1].detach().cpu().numpy())
        a2_true.append(gcn_target.detach().cpu().numpy())
        a2_pred.append(a2_hat.detach().cpu().numpy())
        rec_l1.append(torch.mean(torch.abs(rec - input)).item())

    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    a2_true = np.concatenate(a2_true, axis=0)
    a2_pred = np.concatenate(a2_pred, axis=0)
    y_hat = (y_prob >= 0.5).astype(np.int64)

    auc = sklearn.metrics.roc_auc_score(y_true, y_prob)
    acc = sklearn.metrics.accuracy_score(y_true, y_hat)
    a2_mae = np.mean(np.abs(a2_pred - a2_true))
    rec_l1 = float(np.mean(rec_l1))
    return {
        "auc": auc,
        "acc": acc,
        "a2_mae": a2_mae,
        "rec_l1": rec_l1,
    }


def main():
    args = parse_opt()
    if not args.resume:
        raise ValueError("Please provide --resume for intervention evaluation.")

    dataset_dir = _get_dataset_path(args)
    fold = "test_inter"
    loader = _build_loader(dataset_dir, fold, args)

    model = Model_STAS_2D(1, 64, "adjacency_matrix.pkl", args).cuda()
    ckpt = torch.load(args.resume, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    criterion_gcn = nn.MultiLabelSoftMarginLoss().cuda()

    modes = ["none", "dis", "dom", "cli", "res"]
    metrics = {m: evaluate_one_mode(loader, model, criterion_gcn, m) for m in modes}
    base = metrics["none"]

    print("=== Intervention Report (fold=test_inter) ===")
    print(
        "base: AUC={:.4f}, ACC={:.4f}, A2_MAE={:.4f}, REC_L1={:.4f}".format(
            base["auc"], base["acc"], base["a2_mae"], base["rec_l1"]
        )
    )
    for m in modes[1:]:
        cur = metrics[m]
        print(
            "zero {:>3s}: AUC={:.4f} (Δ{:+.4f}), ACC={:.4f} (Δ{:+.4f}), "
            "A2_MAE={:.4f} (Δ{:+.4f}), REC_L1={:.4f} (Δ{:+.4f})".format(
                m,
                cur["auc"], cur["auc"] - base["auc"],
                cur["acc"], cur["acc"] - base["acc"],
                cur["a2_mae"], cur["a2_mae"] - base["a2_mae"],
                cur["rec_l1"], cur["rec_l1"] - base["rec_l1"],
            )
        )


if __name__ == "__main__":
    main()
