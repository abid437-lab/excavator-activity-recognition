# Multimodal Excavator Activity Recognition (RGB + GADF CNN-LSTM)

Code, trained weights, results and a sample dataset for the manuscript
**"Multimodal Excavator Activity Recognition Using GAF-Encoded Kinematic Sensors and RGB Vision Under Occlusion"**, submitted to *Automation in Construction* (AUTCON-D-26-05142).

The proposed model (`TwoBranchCNNLSTM`, `Model3.py`) fuses RGB video frames (ResNet-18 branch) with
Gramian Angular Difference Field (GADF) images of the excavator kinematic sensor signals (custom CNN branch).
An LSTM cell (256 hidden units) models the frame sequence and classifies four activities:
**Digging, Dumping, Idle, Leveling**.

## Repository contents

```
├── Model3.py               # proposed two-branch CNN-LSTM
├── retrain_eval.py         # training + evaluation of all 5 models, 5 seeds, statistics
├── cv_tune.py              # hyperparameter selection by 3-fold CV on the training set only
├── profile_models.py       # parameter count and inference time per model
├── requirements.txt
├── results/
│   ├── retrain_results_2/  # summary.csv, all_runs.csv, paired_tests.csv, occluded_composition.csv
│   │   └── runs/<model>_seed<k>/   # epoch logs, confusion matrices, per-sequence predictions
│   │                                # model_final.pth included for seed 0 (see "Trained weights")
│   ├── cv_results/         # cross-validation summary and selected configuration
│   └── profile_results.csv
└── sample_data/
    ├── Training/<class>/<VideoID>/{Images,GAF}/
    ├── Validation_normal/<class>/<VideoID>/{Images,GAF}/
    ├── Validation_occluded/<class>/<VideoID>/{Images,GAF}/
    └── sample_index.csv    # list of the sample sequences and occlusion type
```

## Sample dataset

The full dataset is not included because of its size. `sample_data/` contains a representative subset of
**10 sequences per activity class** from each of the three sets:

| Set | Sequences | Description |
|---|---|---|
| Training | 40 | clean training sequences |
| Validation_normal | 40 | clean evaluation sequences |
| Validation_occluded | 40 | evaluation sequences with visual occlusion, sensor faults, both, or clean (type listed in `sample_index.csv`) |

Where possible, the normal and occluded samples use the same video IDs, so the clean and degraded versions
of a sequence can be compared directly. Each sequence folder contains 15 RGB frames (`Images/`) and the 15 matching GADF images (`GAF/`)
with identical file names; the data loader samples 14 frames per sequence.

The complete dataset is available from the corresponding author on reasonable request.

## Trained weights

Final-epoch weights (seed 0) are provided for:

| Model | File | Size |
|---|---|---|
| multimodal (proposed) | `results/retrain_results_2/runs/multimodal_seed0/model_final.pth` | 47 MB |
| img_branch | `results/retrain_results_2/runs/img_branch_seed0/model_final.pth` | 46 MB |
| gaf_branch | `results/retrain_results_2/runs/gaf_branch_seed0/model_final.pth` | 2 MB |
| img_small, gaf_small (baselines) | download from the [Releases](../../releases) page | 1.07 GB each |

Load the proposed model:

```python
import torch
from Model3 import TwoBranchCNNLSTM

model = TwoBranchCNNLSTM(input_channels=3, hidden_size=256, num_ops=4)
model.load_state_dict(torch.load("results/retrain_results_2/runs/multimodal_seed0/model_final.pth",
                                 map_location="cpu"))
model.eval()
```

## Results (5 seeds, mean ± SD, final epoch, no checkpoint selection)

| Model | Params | Normal acc. (%) | Occluded acc. (%) | Drop (pp) |
|---|---:|---:|---:|---:|
| **multimodal (proposed)** | 12.2 M | **98.33 ± 0.50** | **93.10 ± 1.55** | 5.24 |
| img_branch | 12.0 M | 97.26 ± 2.13 | 89.64 ± 2.55 | 7.62 |
| gaf_branch | 0.55 M | 84.88 ± 2.93 | 80.71 ± 3.08 | 4.17 |
| img_small | 268.7 M | 97.14 ± 1.42 | 88.21 ± 1.07 | 8.93 |
| gaf_small | 268.7 M | 91.07 ± 2.42 | 84.41 ± 2.61 | 6.67 |

Full statistics (95% CIs, F1, loss, per-occlusion-type accuracy, McNemar paired tests) are in
`results/retrain_results_2/`.

## Installation

Python 3.9+ and a CUDA GPU are recommended.

```
pip install -r requirements.txt
```

## Usage

Quick test on the sample data (1 epoch, 1 seed):

```
python retrain_eval.py --quick ^
  --train-dir sample_data/Training ^
  --normal-dir sample_data/Validation_normal ^
  --occluded-dir sample_data/Validation_occluded
```

Full training and evaluation (requires the full dataset):

```
python retrain_eval.py --train-dir <Training> --normal-dir <Validation> --occluded-dir <Validation_occluded>
python retrain_eval.py --summary-only          # recompute statistics from finished runs
python cv_tune.py                              # hyperparameter selection (training set only)
python profile_models.py --runs-dir results/retrain_results_2/runs
```

Training on the 40-sequence sample will not reproduce the reported accuracies; it only demonstrates the pipeline.

## Citation

```
<add BibTeX after acceptance>
```

## License

Code: MIT License (see `LICENSE`). Sample data and weights: CC BY 4.0.
