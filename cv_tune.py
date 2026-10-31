"""
Hyperparameter re-selection by cross-validation on the TRAINING set only (reviewer comment R1-3.2).

The 168-sequence normal and occluded sets are NOT loaded or touched by this script.

Protocol
  * 392 training sequences, split into 3 class-stratified folds (fixed split seed 0).
  * Model: proposed multimodal network (identical to retrain_eval.py), batch 8, Adam, no weight decay.
  * Grid: learning rate {1e-3, 1e-4, 1e-5} x LSTM hidden units {128, 256}.
  * Each run trains for 60 epochs and evaluates the held-out fold after every epoch, so the number
    of epochs {30, 40, 50, 60} is selected from the same runs at no extra cost.
  * Selection: highest mean held-out-fold accuracy over the 3 folds; ties broken by lower mean loss.

Runs are resumable: finished runs (DONE.json) are skipped, so the script can be stopped and restarted.
Estimated time on a Quadro RTX 5000: about 40-45 min per run, 18 runs, about 12-13 h in total.

Requires retrain_eval.py (or "Retrin Eval.py") in the same folder; the data path and model code are
taken from it.

Usage:  python cv_tune.py            (full search)
        python cv_tune.py --quick    (1 epoch per run on 1 fold, smoke test, about 5 min)
        python cv_tune.py --summary-only
Output: cv_results/cv_summary.csv, cv_results/cv_selected.json, cv_results/runs/<config>_fold<k>/
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))


def load_retrain_module():
    for fname in ("retrain_eval.py", "Retrin Eval.py", "Retrain Eval.py"):
        path = os.path.join(HERE, fname)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("retrain_eval", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"Model and data code loaded from: {fname}")
            return mod
    sys.exit("Put retrain_eval.py (or 'Retrin Eval.py') in the same folder as cv_tune.py")


R = load_retrain_module()

# ===== search settings =====
LRS = [1e-3, 1e-4, 1e-5]
HIDDENS = [256, 128]
EPOCHS = 60
EPOCH_CHOICES = [30, 40, 50, 60]
BATCH = 8
N_FOLDS = 3
SPLIT_SEED = 0
OUT_DIR = os.path.join(HERE, "cv_results")
MODEL = "multimodal"


def stratified_folds(labels, k, seed):
    """Class-stratified k-fold split of sequence indices (each sequence = one video folder)."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    labels = np.asarray(labels)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(int(i))
    return [sorted(f) for f in folds]


def cfg_name(lr, hidden):
    return f"lr{lr:.0e}_h{hidden}"


def run_fold(ds, folds, fold, lr, hidden, epochs):
    name = cfg_name(lr, hidden)
    run_dir = os.path.join(OUT_DIR, "runs", f"{name}_fold{fold}")
    done = os.path.join(run_dir, "DONE.json")
    if os.path.exists(done):
        print(f"[skip] {name} fold {fold} already finished")
        return
    os.makedirs(run_dir, exist_ok=True)

    val_idx = folds[fold]
    tr_idx = sorted(i for f, idx in enumerate(folds) if f != fold for i in idx)
    seed = fold                                     # fixed, so every config sees identical initialisation per fold
    R.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(Subset(ds, tr_idx), batch_size=BATCH, shuffle=True, drop_last=True,
                              generator=g, num_workers=0)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=BATCH, shuffle=False, num_workers=0)

    model = R.build_model(MODEL, hidden).to(R.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    print(f"=== {name} fold {fold + 1}/{N_FOLDS} | train {len(tr_idx)} | held-out {len(val_idx)} | {epochs} epochs")

    t0 = time.time()
    with open(os.path.join(run_dir, "epoch_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"])
        for ep in range(1, epochs + 1):
            tl, tp, ty, _ = R.run_epoch(MODEL, model, train_loader, criterion, optimizer)
            vl, vp, vy, _ = R.run_epoch(MODEL, model, val_loader, criterion)
            tacc = R.metrics(tp, ty)[0]
            vacc, _, _, vf1 = R.metrics(vp, vy)
            w.writerow([ep, f"{tl:.5f}", f"{tacc:.4f}", f"{vl:.5f}", f"{vacc:.4f}", f"{vf1:.4f}"])
            f.flush()
            print(f"[{name} f{fold}] ep {ep:02d}/{epochs} train loss {tl:.4f} acc {tacc:.4f} | "
                  f"held-out loss {vl:.4f} acc {vacc:.4f} ({time.time() - t0:.0f}s)")
    with open(done, "w") as f:
        json.dump(dict(lr=lr, hidden=hidden, fold=fold, epochs=epochs, n_train=len(tr_idx), n_val=len(val_idx),
                       minutes=round((time.time() - t0) / 60, 1)), f, indent=2)


def read_log(path):
    with open(path, newline="") as f:
        return {int(r["epoch"]): r for r in csv.DictReader(f)}


def summarize():
    rows, curves = [], {}
    for lr in LRS:
        for hidden in HIDDENS:
            name = cfg_name(lr, hidden)
            logs = []
            for fold in range(N_FOLDS):
                p = os.path.join(OUT_DIR, "runs", f"{name}_fold{fold}", "epoch_log.csv")
                if os.path.exists(os.path.join(OUT_DIR, "runs", f"{name}_fold{fold}", "DONE.json")):
                    logs.append(read_log(p))
            if len(logs) < N_FOLDS:
                print(f"  {name}: only {len(logs)}/{N_FOLDS} folds finished, not summarised")
                continue
            max_ep = min(max(l) for l in logs)
            curves[name] = [np.mean([float(l[e]["val_acc"]) for l in logs]) * 100 for e in range(1, max_ep + 1)]
            for ep in EPOCH_CHOICES:
                if ep > max_ep:
                    continue
                accs = np.array([float(l[ep]["val_acc"]) for l in logs]) * 100
                losses = np.array([float(l[ep]["val_loss"]) for l in logs])
                f1s = np.array([float(l[ep]["val_f1"]) for l in logs]) * 100
                rows.append(dict(config=name, lr=lr, hidden=hidden, epochs=ep,
                                 cv_acc_mean=round(accs.mean(), 2), cv_acc_sd=round(accs.std(ddof=1), 2),
                                 cv_f1_mean=round(f1s.mean(), 2), cv_loss_mean=round(losses.mean(), 4),
                                 fold_accs=" / ".join(f"{a:.2f}" for a in accs)))
    if not rows:
        print("Nothing to summarise yet.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "cv_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "cv_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        names = list(curves)
        w.writerow(["epoch"] + names)
        for e in range(max(len(c) for c in curves.values())):
            w.writerow([e + 1] + [f"{curves[n][e]:.2f}" if e < len(curves[n]) else "" for n in names])

    best = sorted(rows, key=lambda r: (-r["cv_acc_mean"], r["cv_loss_mean"]))[0]
    original = next((r for r in rows if r["lr"] == 1e-4 and r["hidden"] == 256 and r["epochs"] == 60), None)
    with open(os.path.join(OUT_DIR, "cv_selected.json"), "w") as f:
        json.dump(dict(selected=best, original_config=original), f, indent=2)

    print("\n" + "=" * 92)
    print(f"{'config':<14}{'epochs':>7}{'CV acc % (mean ± SD)':>24}{'CV F1 %':>10}{'CV loss':>10}   per fold")
    for r in sorted(rows, key=lambda r: (-r["cv_acc_mean"], r["cv_loss_mean"])):
        mark = "  <== selected" if r is best else ""
        print(f"{r['config']:<14}{r['epochs']:>7}{r['cv_acc_mean']:>15.2f} ± {r['cv_acc_sd']:<6.2f}"
              f"{r['cv_f1_mean']:>10.2f}{r['cv_loss_mean']:>10.4f}   {r['fold_accs']}{mark}")
    print("=" * 92)
    print(f"Selected by CV on training data: lr {best['lr']:.0e}, hidden {best['hidden']}, {best['epochs']} epochs "
          f"(CV acc {best['cv_acc_mean']:.2f}%)")
    if original:
        same = best["lr"] == 1e-4 and best["hidden"] == 256 and best["epochs"] == 60
        print(f"Original configuration (lr 1e-4, hidden 256, 60 epochs): CV acc {original['cv_acc_mean']:.2f}% "
              f"-> {'SAME as selected' if same else 'differs from selected'}")
    print(f"Saved {OUT_DIR}\\cv_summary.csv, cv_curves.csv, cv_selected.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.summary_only:
        summarize()
        return

    global OUT_DIR
    if args.quick:
        OUT_DIR = os.path.join(HERE, "cv_results_quick")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {R.DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if R.DEVICE.type == "cuda" else ""))
    print(f"Training data: {R.TRAIN_DIR}")
    print("[load] training data into memory (test sets are NOT loaded) ...")
    ds = R.DualDataset(R.TRAIN_DIR)
    labels = [it[2] for it in ds.items]
    folds = stratified_folds(labels, N_FOLDS, SPLIT_SEED)
    print(f"[load] {len(ds)} training sequences | per class {np.bincount(labels).tolist()}")
    with open(os.path.join(OUT_DIR, "folds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "class", "video"])
        for k, idx in enumerate(folds):
            for i in idx:
                w.writerow([k, R.CLASSES[ds.items[i][2]], ds.items[i][3]])
    for k, idx in enumerate(folds):
        print(f"  fold {k}: {len(idx)} sequences, per class {np.bincount([labels[i] for i in idx], minlength=4).tolist()}")

    configs = [(lr, h) for h in HIDDENS for lr in LRS]
    configs.sort(key=lambda c: (c != (1e-4, 256)))          # original configuration first
    if args.quick:
        run_fold(ds, folds, 0, 1e-4, 256, 1)
        print("Quick test finished. Delete the cv_results_quick folder and run without --quick.")
        return
    t0 = time.time()
    for lr, hidden in configs:
        for fold in range(N_FOLDS):
            run_fold(ds, folds, fold, lr, hidden, EPOCHS)
        print(f"--- elapsed {(time.time() - t0) / 3600:.1f} h")
    summarize()


if __name__ == "__main__":
    main()
