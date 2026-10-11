# Aerial Waste Detection & Dumping Severity Scoring

Research codebase for a paper on top-down/aerial waste classification and
detection — extending standard ground-level waste classification
(TrashNet-style) to drone imagery of open dumps, landfill sorting yards,
and illegal dumping sites, where occlusion, small object scale, and
variable lighting/altitude make the task meaningfully harder.

Full context, the literature gap, and exactly what's baseline vs. novel
contribution: see [CLAUDE.md](CLAUDE.md) and
[docs/contribution-notes.md](docs/contribution-notes.md).

- **Base dataset**: KACHARA aerial-waste dataset —
  https://onlinelibrary.wiley.com/doi/10.1155/acis/9106130
- **Baseline**: YOLOv8 / Faster R-CNN, no tiling, no fusion, no severity
  scoring.
- **Contribution 1**: tiled, scale-and-occlusion-aware detection with a
  multi-scale feature fusion head.
- **Contribution 2**: dumping severity scoring — per-zone coverage/density
  and an overall priority score per site.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Data

Place the KACHARA export (YOLO format) under `data/raw/`:

```
data/raw/
  data.yaml           # names: [...], nc: N
  train/images/  train/labels/
  valid/images/  valid/labels/
  test/images/   test/labels/
```

Verify the layout and check class distribution before training:

```bash
python scripts/verify_dataset.py --root data/raw
```

(Optional) pre-generate tiled patches to disk instead of tiling on the fly:

```bash
python scripts/tile_dataset.py --root data/raw --split train --out data/processed/train --tile-size 640 --overlap 0.2
```

## Training

Pipeline (`baseline_yolo` / `baseline_frcnn` / `scale_fusion`) is selected
by the config file — same script for baseline and contribution runs so
results are directly comparable.

```bash
python -m src.train --config experiments/configs/baseline_yolo.yaml
python -m src.train --config experiments/configs/baseline_frcnn.yaml
python -m src.train --config experiments/configs/scale_fusion.yaml
```

Logs go to `experiments/logs/`, checkpoints to `experiments/checkpoints/<run_name>/`.

## Evaluation

```bash
python -m src.evaluate --config experiments/configs/baseline_yolo.yaml --split test
python -m src.evaluate --config experiments/configs/baseline_frcnn.yaml --split test --weights experiments/checkpoints/baseline_frcnn/final.pt
python -m src.evaluate --config experiments/configs/scale_fusion.yaml --split test --weights experiments/checkpoints/scale_fusion/final.pt
```

Reports mAP@[.5:.95], mAP@.5, and IoU@.5 precision/recall; metrics are
written to `results/tables/<run_name>_<split>_metrics.json`.

## Inference (tiling -> detection -> severity scoring)

Runs the full pipeline on one image and writes an annotated image +
severity report:

```bash
python -m src.inference --config experiments/configs/scale_fusion.yaml --weights experiments/checkpoints/scale_fusion/final.pt --image path/to/site.jpg
```

Output goes to `results/qualitative/`: `<name>_annotated.jpg` (detections +
per-zone severity overlay) and `<name>_severity_report.json` (per-zone
coverage/severity + overall score).

## Tests

```bash
pytest
```

## Repo layout

See [CLAUDE.md](CLAUDE.md) for the full directory breakdown and the
baseline-vs-contribution boundary that should be kept consistent across
code, configs, and the eventual paper.
