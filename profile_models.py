r"""
profile_models.py  (self-contained - no other script needed)

Measures inference time and peak GPU memory of the five models of the revision
(multimodal, img_branch, gaf_branch, img_small, gaf_small) for one sequence of
14 frames. The model definitions are copied from retrain_eval.py; the trained
seed-0 weights are loaded from RUNS_DIR if they exist.

Run in PyCharm (Run 'profile_models') or:  python profile_models.py
Output: profile_results.csv next to this script + a table in the console.
"""
import argparse
import csv
import os
import statistics
import time

import torch
import torch.nn as nn
import torchvision.models as models

RUNS_DIR = r"E:\Kamran Space\Kamran_Stuff\Kamran_Stuff\retrain_results_2\runs"
T = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same settings as retrain_eval.py
CFG = {
    "multimodal": dict(hidden=256, kind="dual", mode="both"),
    "img_branch": dict(hidden=256, kind="dual", mode="img"),
    "gaf_branch": dict(hidden=256, kind="dual", mode="gaf"),
    "img_small":  dict(hidden=256, kind="small", size=256),
    "gaf_small":  dict(hidden=256, kind="small", size=256),
}
MODEL_ORDER = ["multimodal", "img_branch", "gaf_branch", "img_small", "gaf_small"]


# ---------------- model definitions (identical to retrain_eval.py) ----------------
def resnet18_trunk():
    try:
        net = models.resnet18(weights=None)
    except TypeError:
        net = models.resnet18(pretrained=False)
    return nn.Sequential(*list(net.children())[:-1])


def gaf_cnn(in_ch=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True),
    )


class DualStreamNet(nn.Module):
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
    def __init__(self, hidden=256, num_classes=4, img_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        n_in = self.cnn(torch.zeros(1, 3, img_size, img_size)).numel()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.dropout_fc = nn.Identity()
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        b, n = x.shape[:2]
        feats = torch.stack([self.cnn(x[:, t]).reshape(b, -1) for t in range(n)], dim=1)
        out, _ = self.lstm(feats)
        return self.fc(self.dropout_fc(out[:, -1, :]))


def build_model(name):
    cfg = CFG[name]
    if cfg["kind"] == "dual":
        return DualStreamNet(cfg["mode"], cfg["hidden"])
    return SmallCNNLSTM(cfg["hidden"], img_size=cfg["size"])


def make_input(name):
    cfg = CFG[name]
    if cfg["kind"] == "dual":
        g = torch.rand(1, T, 3, 224, 224, device=DEVICE)
        r = torch.rand(1, T, 3, 224, 224, device=DEVICE)
        return lambda m: m(g, r)
    x = torch.rand(1, T, 3, cfg["size"], cfg["size"], device=DEVICE)
    return lambda m: m(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()

    if DEVICE.type != "cuda":
        print("WARNING: no GPU found - times are CPU times and peak GPU memory is not available.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))
    print("Weights folder:", args.runs_dir, "| exists:", os.path.isdir(args.runs_dir))

    rows = []
    for name in MODEL_ORDER:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = build_model(name).to(DEVICE).eval()
        w = os.path.join(args.runs_dir, f"{name}_seed{args.seed}", "model_final.pth")
        loaded = os.path.exists(w)
        if loaded:
            model.load_state_dict(torch.load(w, map_location=DEVICE))
        params = sum(p.numel() for p in model.parameters())
        run = make_input(name)

        with torch.no_grad():
            for _ in range(args.warmup):
                run(model)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                run(model)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else float("nan")

        row = dict(model=name, params=params, weights_loaded=loaded,
                   mean_ms=round(statistics.mean(times), 2), sd_ms=round(statistics.stdev(times), 2),
                   min_ms=round(min(times), 2), max_ms=round(max(times), 2), peak_gpu_mb=round(peak, 1))
        rows.append(row)
        print(f"{name:<11} params {params:>12,}  {row['mean_ms']:>8.2f} ± {row['sd_ms']:<7.2f} ms   "
              f"peak GPU {row['peak_gpu_mb']:>8.1f} MB   weights {'loaded' if loaded else 'random'}")
        del model, run

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_results.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\nWritten:", out)


if __name__ == "__main__":
    main()