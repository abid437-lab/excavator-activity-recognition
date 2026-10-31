"""
Retraining + statistical evaluation for the multimodal excavator activity paper.

Answers reviewer comments R1-3.1 (McNemar), R1-3.2 (no selection on evaluation
data), R1-3.3 (true per-sample loss), R1-3.4 (95% CIs), R2-1, R2-2 (per-type
breakdown) and R2-4(iii) (full results matrix).

Protocol
  * Train on the clean training set (98 sequences/class).
  * The 168 validation sequences are used ONLY for evaluation, in two versions:
      - normal   : clean validation set
      - occluded : same set where 5 seq/class were replaced by naturally
                   visually-occluded clips and 5 seq/class had sensor faults
                   injected (never both).
  * Fixed hyperparameters, fixed number of epochs, results reported at the
    FINAL epoch (no checkpoint selection), several fixed seeds per model.
  * Every model reads the SAME multimodal folders (class/video/{Images,GAF}),
    so all models are evaluated on identical sequences.

Models
  multimodal : Model3 TwoBranchCNNLSTM (ResNet-18 + GAF CNN -> LSTMCell 256)
  img_branch : ResNet-18 branch only     -> LSTMCell 256  (same settings as multimodal)
  gaf_branch : GAF CNN branch only       -> LSTMCell 256  (same settings as multimodal)
  img_small  : original single-stream baseline CNNLSTM on 256x256 RGB frames
  gaf_small  : original single-stream baseline CNNLSTM on 256x256 GADF images

Usage
  python retrain_eval.py --quick                      # 1 epoch, 1 seed smoke test
  python retrain_eval.py                              # full run, all models
  python retrain_eval.py --models multimodal gaf_branch img_branch
  python retrain_eval.py --summary-only               # recompute statistics only
Re-running skips finished runs, so the script can be stopped and restarted.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

# =====================================================================
# CONFIGURATION  (edit these paths)
# =====================================================================
TRAIN_DIR = r"E:\Kamran Space\Abid_Project\12. Dual Stream 4 act - without occulsion\Training"
TEST_NORMAL_DIR = r"E:\Kamran Space\Abid_Project\12. Dual Stream 4 act - without occulsion\Validation"
TEST_OCCLUDED_DIR = r"E:\Kamran Space\Abid_Project\12. Dual Stream 4 act with full occulsuion\Validation - PF_Occul in images and GAF 7"
OUT_DIR = r".\retrain_results"

CLASSES = ["Digging", "Dumping", "Idle", "Leveling"]      # matched case-insensitively
SEEDS = [0, 1, 2, 3, 4]
T = 14
IMG_EXT = (".jpg", ".jpeg", ".png")

# Hyperparameters (Table 2 for multimodal/branches, Mainsingle_1 for small baselines)
CFG = {
    "multimodal": dict(epochs=60, batch=8, lr=1e-4, wd=0.0, hidden=256, kind="dual"),
    "img_branch": dict(epochs=60, batch=8, lr=1e-4, wd=0.0, hidden=256, kind="dual"),
    "gaf_branch": dict(epochs=60, batch=8, lr=1e-4, wd=0.0, hidden=256, kind="dual"),
    # original paper baselines (single-stream CNN-LSTM, 256x256, no regularisation, 30 epochs)
    "img_small":  dict(epochs=30, batch=8, lr=1e-4, wd=0.0, hidden=256, kind="small", stream="Images",
                       size=256, dropout=False, noise=0.0),
    "gaf_small":  dict(epochs=30, batch=8, lr=1e-4, wd=0.0, hidden=256, kind="small", stream="GAF",
                       size=256, dropout=False, noise=0.0),
}
MODEL_ORDER = ["multimodal", "gaf_branch", "img_branch", "gaf_small", "img_small"]
REFERENCE = "multimodal"
PRETRAINED = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# UTILITIES
# =====================================================================
class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def write(self, m):
        sys.__stdout__.write(m)
        self.f.write(m)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_dirs(root):
    """Return {class_index: actual_folder_path}, matching CLASSES case-insensitively."""
    existing = {d.lower(): d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    out = {}
    for i, c in enumerate(CLASSES):
        if c.lower() not in existing:
            sys.exit(f"Class folder '{c}' not found in {root}. Found: {sorted(existing.values())}")
        out[i] = os.path.join(root, existing[c.lower()])
    return out


def list_images(path):
    return sorted(f for f in os.listdir(path) if f.lower().endswith(IMG_EXT))


def list_videos(root):
    """[(label, video_name, video_path)] for folders that contain both Images and GAF."""
    vids = []
    for lab, cdir in class_dirs(root).items():
        for v in sorted(os.listdir(cdir)):
            vp = os.path.join(cdir, v)
            if not os.path.isdir(vp):
                continue
            if os.path.isdir(os.path.join(vp, "Images")) and os.path.isdir(os.path.join(vp, "GAF")):
                vids.append((lab, v, vp))
            else:
                print(f"WARNING: skipped {vp} (needs both 'Images' and 'GAF' subfolders; found {os.listdir(vp)})")
    return vids


def dual_indices(n):
    """Identical to dataset_dual2.CustomDataset sampling."""
    if n >= T:
        step = n // T
        return [i * step for i in range(T)]
    return [i % n for i in range(T)]


def folder_hash(path):
    h = hashlib.md5()
    for f in list_images(path):
        with open(os.path.join(path, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def degradation_types(normal_dir, occl_dir):
    """Classify every occluded-test sequence as clean / visual / sensor by comparing folders."""
    normal = {(lab, v): vp for lab, v, vp in list_videos(normal_dir)}
    types = {}
    for lab, v, vp in list_videos(occl_dir):
        key = (lab, v)
        if key not in normal:
            types[key] = "visual"
            continue
        img_same = folder_hash(os.path.join(vp, "Images")) == folder_hash(os.path.join(normal[key], "Images"))
        gaf_same = folder_hash(os.path.join(vp, "GAF")) == folder_hash(os.path.join(normal[key], "GAF"))
        if img_same and gaf_same:
            types[key] = "clean"
        elif img_same and not gaf_same:
            types[key] = "sensor"
        elif gaf_same and not img_same:
            types[key] = "visual"
        else:
            types[key] = "both"
    return types


# =====================================================================
# DATASETS (preloaded into RAM)
# =====================================================================
DUAL_TF = transforms.Resize((224, 224))
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class DualDataset(Dataset):
    """Same preprocessing as dataset_dual2: 14 sampled frames, 224x224, ImageNet norm."""

    def __init__(self, root):
        self.items = []
        for lab, v, vp in list_videos(root):
            rgb = list_images(os.path.join(vp, "Images"))
            gaf = list_images(os.path.join(vp, "GAF"))
            idx = dual_indices(min(len(rgb), len(gaf)))
            r = np.stack([np.array(DUAL_TF(Image.open(os.path.join(vp, "Images", rgb[i])).convert("RGB"))) for i in idx])
            g = np.stack([np.array(DUAL_TF(Image.open(os.path.join(vp, "GAF", gaf[i])).convert("RGB"))) for i in idx])
            self.items.append((g, r, lab, v))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _to_tensor(a):
        x = torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0
        return NORM(x)

    def __getitem__(self, i):
        g, r, lab, _ = self.items[i]
        return self._to_tensor(g), self._to_tensor(r), lab, i


class SmallDataset(Dataset):
    """Same preprocessing as the original single-stream baseline: cv2 (BGR), resize, /255,
    random 14-frame window when training (optional noise). Evaluation uses fixed frames."""

    def __init__(self, root, stream, train, size, noise):
        self.train, self.noise = train, noise
        self.items = []
        for lab, v, vp in list_videos(root):
            files = list_images(os.path.join(vp, stream))
            if len(files) < T:
                raise ValueError(f"Not enough frames in {os.path.join(vp, stream)}")
            frames = [np.transpose(cv2.resize(cv2.imread(os.path.join(vp, stream, f)), (size, size)), (2, 0, 1))
                      for f in files]
            self.items.append((np.stack(frames), lab, v))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        frames, lab, _ = self.items[i]
        n = len(frames)
        if self.train:
            s = random.randint(0, n - T)
            x = torch.from_numpy(frames[s:s + T].astype(np.float32) / 255.0)
            if self.noise > 0:
                x = torch.clamp(x + torch.randn_like(x) * self.noise, 0.0, 1.0)
        else:
            idx = dual_indices(n)                       # same frames as the dual models
            x = torch.from_numpy(frames[idx].astype(np.float32) / 255.0)
        return x, lab, i


# =====================================================================
# MODELS
# =====================================================================
def resnet18_trunk():
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
        net = models.resnet18(weights=weights)
    except AttributeError:
        net = models.resnet18(pretrained=PRETRAINED)
    return nn.Sequential(*list(net.children())[:-1])


def gaf_cnn(in_ch=3):
    """Identical to Model3._make_cnn_branch."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True),
    )


class DualStreamNet(nn.Module):
    """mode='both' reproduces Model3 exactly; 'img' / 'gaf' are the single-branch ablations."""

    def __init__(self, mode, hidden=256, num_classes=4):
        super().__init__()
        self.mode, self.hidden = mode, hidden
        in_dim = 0
        if mode in ("both", "gaf"):
            self.cnn_gaf = gaf_cnn()
            in_dim += 128
        if mode in ("both", "img"):
            self.cnn_img = resnet18_trunk()
            in_dim += 512
        self.lstm = nn.LSTMCell(in_dim, hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, gaf_seq, img_seq):
        b, n = gaf_seq.shape[:2]
        h = torch.zeros(b, self.hidden, device=gaf_seq.device)
        c = torch.zeros(b, self.hidden, device=gaf_seq.device)
        for t in range(n):
            feats = []
            if self.mode in ("both", "gaf"):
                feats.append(self.cnn_gaf(gaf_seq[:, t]))
            if self.mode in ("both", "img"):
                feats.append(self.cnn_img(img_seq[:, t]).reshape(b, -1))
            h, c = self.lstm(torch.cat(feats, dim=1), (h, c))
        return self.classifier(h)


class SmallCNNLSTM(nn.Module):
    """Original single-stream baseline CNNLSTM (dropout=False, size=256 reproduces it exactly;
    dropout=True, size=100 reproduces Mainsingle_1)."""

    def __init__(self, hidden=256, num_classes=4, img_size=256, dropout=False):
        super().__init__()
        layers = [nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
        if dropout:
            layers.append(nn.Dropout2d(p=0.3))
        layers += [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
        if dropout:
            layers.append(nn.Dropout2d(p=0.3))
        self.cnn = nn.Sequential(*layers)
        n_in = self.cnn(torch.zeros(1, 3, img_size, img_size)).numel()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.dropout_fc = nn.Dropout(p=0.5) if dropout else nn.Identity()
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        b, n = x.shape[:2]
        feats = torch.stack([self.cnn(x[:, t]).reshape(b, -1) for t in range(n)], dim=1)
        out, _ = self.lstm(feats)
        return self.fc(self.dropout_fc(out[:, -1, :]))


def build_model(name, hidden):
    if name == "multimodal":
        return DualStreamNet("both", hidden)
    if name == "img_branch":
        return DualStreamNet("img", hidden)
    if name == "gaf_branch":
        return DualStreamNet("gaf", hidden)
    cfg = CFG[name]
    return SmallCNNLSTM(hidden, img_size=cfg["size"], dropout=cfg["dropout"])


def forward_batch(name, model, batch):
    if CFG[name]["kind"] == "dual":
        g, r, y, i = batch
        return model(g.to(DEVICE), r.to(DEVICE)), y.to(DEVICE), i
    x, y, i = batch
    return model(x.to(DEVICE)), y.to(DEVICE), i


# =====================================================================
# TRAIN / EVALUATE
# =====================================================================
def run_epoch(name, model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum, n, preds, labels, idxs = 0.0, 0, [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            if train:
                optimizer.zero_grad()
            out, y, i = forward_batch(name, model, batch)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)            # true per-sample mean
            n += y.size(0)
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()
            idxs += i.tolist()
    return loss_sum / n, preds, labels, idxs


def metrics(preds, labels):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(4)), average="macro", zero_division=0)
    return float(np.mean(np.array(preds) == np.array(labels))), p, r, f


def save_preds(path, ds, idxs, labels, preds, types=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "video", "true", "pred", "correct", "type"])
        for i, t, p in sorted(zip(idxs, labels, preds)):
            vid = ds.items[i][-1]
            typ = types.get((t, vid), "") if types else ""
            w.writerow([CLASSES[t], vid, t, p, int(t == p), typ])


def run_one(name, seed, train_ds, tests, types, quick):
    cfg = CFG[name]
    run_dir = os.path.join(OUT_DIR, "runs", f"{name}_seed{seed}")
    done = os.path.join(run_dir, "DONE.json")
    if os.path.exists(done):
        print(f"[skip] {name} seed {seed} already finished")
        return
    os.makedirs(run_dir, exist_ok=True)
    tee = Tee(os.path.join(run_dir, "console.log"))
    sys.stdout = tee
    try:
        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        epochs = 1 if quick else cfg["epochs"]
        train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, drop_last=True,
                                  generator=g, num_workers=0)
        test_loaders = {k: DataLoader(d, batch_size=cfg["batch"], shuffle=False, num_workers=0) for k, d in tests.items()}

        model = build_model(name, cfg["hidden"]).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        with open(os.path.join(run_dir, "run_info.json"), "w") as f:
            json.dump(dict(cfg, model=name, seed=seed, epochs=epochs, params=n_params, device=str(DEVICE)), f, indent=2)
        print(f"=== {name} seed {seed} | params {n_params:,} | {DEVICE} | {epochs} epochs")

        t0 = time.time()
        with open(os.path.join(run_dir, "epoch_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc"] +
                       [f"{k}_{m}" for k in tests for m in ("loss", "acc", "f1")])
            for ep in range(1, epochs + 1):
                tl, tp, ty, _ = run_epoch(name, model, train_loader, criterion, optimizer)
                row = [ep, f"{tl:.5f}", f"{metrics(tp, ty)[0]:.4f}"]
                msg = f"[{name} s{seed}] ep {ep:02d}/{epochs} train loss {tl:.4f} acc {metrics(tp, ty)[0]:.4f}"
                final = {}
                for k, ld in test_loaders.items():
                    vl, vp, vy, vi = run_epoch(name, model, ld, criterion)
                    acc, _, _, f1 = metrics(vp, vy)
                    row += [f"{vl:.5f}", f"{acc:.4f}", f"{f1:.4f}"]
                    msg += f" | {k} loss {vl:.4f} acc {acc:.4f}"
                    final[k] = (vl, vp, vy, vi)
                w.writerow(row)
                f.flush()
                print(msg + f" ({time.time() - t0:.0f}s)")

        torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pth"))   # final-epoch weights
        result = dict(model=name, seed=seed, params=n_params, minutes=round((time.time() - t0) / 60, 1))
        for k, (vl, vp, vy, vi) in final.items():          # FINAL epoch only, no selection
            acc, p, r, f1 = metrics(vp, vy)
            result.update({f"{k}_loss": vl, f"{k}_acc": acc, f"{k}_precision": p, f"{k}_recall": r, f"{k}_f1": f1})
            save_preds(os.path.join(run_dir, f"pred_{k}.csv"), tests[k], vi, vy, vp, types if k == "occluded" else None)
            np.savetxt(os.path.join(run_dir, f"cm_{k}.csv"), confusion_matrix(vy, vp, labels=list(range(4))),
                       fmt="%d", delimiter=",")
        with open(done, "w") as f:
            json.dump(result, f, indent=2)
        print(f"=== {name} seed {seed} done: normal {result['normal_acc']:.4f} | occluded {result['occluded_acc']:.4f}")
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


# =====================================================================
# STATISTICS
# =====================================================================
def t_ci(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return v.mean(), 0.0, v.mean(), v.mean()
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return m, sd, m - h, m + h


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def read_preds(path):
    with open(path, encoding="utf-8") as f:
        return {(r["class"], r["video"]): r for r in csv.DictReader(f)}


def mcnemar(a, b, keys):
    n01 = sum(1 for k in keys if a[k]["correct"] == "1" and b[k]["correct"] == "0")
    n10 = sum(1 for k in keys if a[k]["correct"] == "0" and b[k]["correct"] == "1")
    p = 1.0 if n01 + n10 == 0 else stats.binomtest(n01, n01 + n10, 0.5).pvalue
    return n01, n10, p


def summarize():
    run_root = os.path.join(OUT_DIR, "runs")
    if not os.path.isdir(run_root):
        sys.exit("No runs found.")
    results = []
    for d in sorted(os.listdir(run_root)):
        p = os.path.join(run_root, d, "DONE.json")
        if os.path.exists(p):
            with open(p) as f:
                results.append(json.load(f))
    if not results:
        sys.exit("No finished runs.")
    names = [m for m in MODEL_ORDER if any(r["model"] == m for r in results)]

    # ---- per-type accuracy from prediction files ----
    for r in results:
        pr = read_preds(os.path.join(run_root, f"{r['model']}_seed{r['seed']}", "pred_occluded.csv"))
        for typ in ("clean", "visual", "sensor"):
            rows = [x for x in pr.values() if x["type"] == typ]
            r[f"occ_{typ}_acc"] = float(np.mean([int(x["correct"]) for x in rows])) if rows else np.nan
            r[f"occ_{typ}_n"] = len(rows)
        deg = [x for x in pr.values() if x["type"] in ("visual", "sensor")]
        r["occ_degraded_acc"] = float(np.mean([int(x["correct"]) for x in deg])) if deg else np.nan
        r["drop_pp"] = (r["normal_acc"] - r["occluded_acc"]) * 100

    with open(os.path.join(OUT_DIR, "all_runs.csv"), "w", newline="") as f:
        keys = sorted({k for r in results for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    # ---- summary table: mean, SD, 95% t-CI across seeds ----
    metrics_list = ["normal_acc", "normal_f1", "normal_loss", "occluded_acc", "occluded_f1", "occluded_loss",
                    "occ_clean_acc", "occ_visual_acc", "occ_sensor_acc", "occ_degraded_acc", "drop_pp"]
    rows = []
    print("\n" + "=" * 100)
    print(f"{'model':<12}{'seeds':<7}{'normal acc %':<24}{'occluded acc %':<24}{'drop (pp)':<22}{'params':>10}")
    for m in names:
        rs = [r for r in results if r["model"] == m]
        row = {"model": m, "n_seeds": len(rs), "params": rs[0]["params"]}
        for k in metrics_list:
            vals = [r[k] * (100 if k.endswith("acc") or k.endswith("f1") else 1) for r in rs]
            mean, sd, lo, hi = t_ci(vals)
            row.update({f"{k}_mean": round(mean, 3), f"{k}_sd": round(sd, 3),
                        f"{k}_ci_low": round(lo, 3), f"{k}_ci_high": round(hi, 3)})
        rows.append(row)
        fmt = lambda k: f"{row[k + '_mean']:.2f} [{row[k + '_ci_low']:.2f},{row[k + '_ci_high']:.2f}]"
        print(f"{m:<12}{len(rs):<7}{fmt('normal_acc'):<24}{fmt('occluded_acc'):<24}{fmt('drop_pp'):<22}{rs[0]['params']:>10,}")
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nPer-type accuracy on the occluded test set (mean over seeds, %):")
    print(f"{'model':<12}{'clean':>10}{'visual':>10}{'sensor':>10}{'degraded':>10}")
    for row in rows:
        print(f"{row['model']:<12}{row['occ_clean_acc_mean']:>10.2f}{row['occ_visual_acc_mean']:>10.2f}"
              f"{row['occ_sensor_acc_mean']:>10.2f}{row['occ_degraded_acc_mean']:>10.2f}")

    # ---- paired tests vs reference model ----
    tests = []
    if REFERENCE in names:
        for m in names:
            if m == REFERENCE:
                continue
            seeds = sorted({r["seed"] for r in results if r["model"] == m} &
                           {r["seed"] for r in results if r["model"] == REFERENCE})
            for split in ("normal", "occluded"):
                diffs = []
                for s in seeds:
                    a = read_preds(os.path.join(run_root, f"{REFERENCE}_seed{s}", f"pred_{split}.csv"))
                    b = read_preds(os.path.join(run_root, f"{m}_seed{s}", f"pred_{split}.csv"))
                    subsets = {"all": sorted(set(a) & set(b))}
                    if split == "occluded":
                        subsets["degraded"] = [k for k in subsets["all"] if a[k]["type"] in ("visual", "sensor")]
                    for sub, keys in subsets.items():
                        n01, n10, p = mcnemar(a, b, keys)
                        acc_a = np.mean([int(a[k]["correct"]) for k in keys])
                        acc_b = np.mean([int(b[k]["correct"]) for k in keys])
                        tests.append(dict(reference=REFERENCE, model=m, split=split, subset=sub, seed=s, n=len(keys),
                                          ref_acc=round(acc_a * 100, 2), model_acc=round(acc_b * 100, 2),
                                          ref_right_model_wrong=n01, ref_wrong_model_right=n10,
                                          mcnemar_p=round(p, 5)))
                        if sub == "all":
                            diffs.append((acc_a - acc_b) * 100)
                if len(diffs) >= 2:
                    t, p = stats.ttest_1samp(diffs, 0.0)
                    mean, sd, lo, hi = t_ci(diffs)
                    tests.append(dict(reference=REFERENCE, model=m, split=split, subset="all", seed="paired_t",
                                      n=len(diffs), ref_acc="", model_acc=f"diff {mean:.2f} [{lo:.2f},{hi:.2f}]",
                                      ref_right_model_wrong="", ref_wrong_model_right="", mcnemar_p=round(p, 5)))
        with open(os.path.join(OUT_DIR, "paired_tests.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
            w.writeheader()
            w.writerows(tests)
        print(f"\nPaired tests vs {REFERENCE} (McNemar per seed; paired t-test across seeds):")
        for t in tests:
            if t["seed"] == "paired_t":
                print(f"  {t['model']:<11}{t['split']:<9} paired t-test on accuracy difference: "
                      f"{t['model_acc']}  p = {t['mcnemar_p']}")
            elif t["split"] == "occluded":
                print(f"  {t['model']:<11}occluded/{t['subset']:<9} seed {t['seed']}: "
                      f"{t['ref_acc']:.2f} vs {t['model_acc']:.2f}  discordant {t['ref_right_model_wrong']}/"
                      f"{t['ref_wrong_model_right']}  McNemar p = {t['mcnemar_p']}")
    print(f"\nFiles: {OUT_DIR}\\summary.csv, all_runs.csv, paired_tests.csv")


# =====================================================================
# MAIN
# =====================================================================
def main():
    global OUT_DIR, PRETRAINED, TRAIN_DIR, TEST_NORMAL_DIR, TEST_OCCLUDED_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--quick", action="store_true", help="1 epoch, seed 0, into retrain_results_TEST")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="testing only")
    ap.add_argument("--train-dir")
    ap.add_argument("--normal-dir")
    ap.add_argument("--occluded-dir")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    TRAIN_DIR = args.train_dir or TRAIN_DIR
    TEST_NORMAL_DIR = args.normal_dir or TEST_NORMAL_DIR
    TEST_OCCLUDED_DIR = args.occluded_dir or TEST_OCCLUDED_DIR
    OUT_DIR = args.out_dir or (OUT_DIR + "_TEST" if args.quick else OUT_DIR)
    PRETRAINED = not args.no_pretrained
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.summary_only:
        summarize()
        return

    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else
                                 "  WARNING: no GPU, training will be very slow"))
    types = degradation_types(TEST_NORMAL_DIR, TEST_OCCLUDED_DIR)
    print("Occluded test set composition (detected by comparing with the normal test set):")
    for lab, c in enumerate(CLASSES):
        cnt = {t: sum(1 for (l, _), v in types.items() if l == lab and v == t) for t in ("clean", "visual", "sensor", "both")}
        print(f"  {c:<10} {cnt}")
    with open(os.path.join(OUT_DIR, "occluded_composition.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "video", "type"])
        for (lab, v), t in sorted(types.items()):
            w.writerow([CLASSES[lab], v, t])

    seeds = [0] if args.quick else args.seeds
    cache = {}
    for name in args.models:
        kind = CFG[name]["kind"]
        key = kind if kind == "dual" else f"small_{CFG[name]['stream']}"
        if key not in cache:
            cache.clear()                                   # keep RAM use low
            print(f"\n[load] {key} data into memory ...")
            if kind == "dual":
                cache[key] = (DualDataset(TRAIN_DIR),
                              {"normal": DualDataset(TEST_NORMAL_DIR), "occluded": DualDataset(TEST_OCCLUDED_DIR)})
            else:
                st = CFG[name]["stream"]
                sz, nz = CFG[name]["size"], CFG[name]["noise"]
                cache[key] = (SmallDataset(TRAIN_DIR, st, True, sz, nz),
                              {"normal": SmallDataset(TEST_NORMAL_DIR, st, False, sz, nz),
                               "occluded": SmallDataset(TEST_OCCLUDED_DIR, st, False, sz, nz)})
            tr, te = cache[key]
            print(f"[load] train {len(tr)} | normal {len(te['normal'])} | occluded {len(te['occluded'])} sequences")
        tr, te = cache[key]
        for s in seeds:
            run_one(name, s, tr, te, types, args.quick)

    summarize()


if __name__ == "__main__":
    main()