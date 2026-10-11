# Aerial Waste Detection and Dumping Severity Scoring

Detects waste in top-down aerial (drone) images and scores how severe each zone is.
Built around a YOLOv8 baseline, with a tiling plus multi-scale fusion detector and a
zone-based severity module for small and occluded waste.

## Status

| Component | State |
|---|---|
| YOLOv8n baseline (full dataset, GPU) | Trained and evaluated |
| Streamlit demo (`app.py`) | Working |
| Severity scoring (`src/models/severity_scoring`) | Implemented and tested |
| Tiling-only and scale-fusion detectors (`src/models/scale_fusion`) | Implemented, configs ready, not trained yet |
| CLIP input check (`src/domain_guard.py`) | Experimental, not fully verified |

## Baseline results

YOLOv8n trained on the full dataset (Colab T4, 50 epochs, image size 640, batch 32, about 1.5 hours).

| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| 0.609 | 0.347 | 0.673 | 0.551 |

Earlier CPU runs on small subsets (900 and 2000 images) levelled off around 0.42 mAP50,
so the limit was the amount of data, not the number of epochs.

## Dataset

Single class (`waste`), YOLO format, 640x640 images, 6039 train / 1695 valid / 951 test.
About half of all boxes are smaller than 32 px, so small-object detection is the hard part.

Source: Kaggle `nikitabreskanu/waste-detection-yolo` (CC BY-SA 4.0). It is not included in
this repo. Download it into `data/raw/` so the folder contains `train/`, `valid/`, `test/`
and `data.yaml`:

```bash
kaggle datasets download -d nikitabreskanu/waste-detection-yolo -p data/raw --unzip
python scripts/verify_dataset.py
```

Make sure the paths inside `data/raw/data.yaml` point to your local `train/images`,
`valid/images` and `test/images` folders.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.12 is recommended. NumPy is pinned below 2 because PyTorch builds for NumPy 1.x.

## Usage

Run the demo (upload an aerial image to see detections and a severity overlay):

```bash
python -m streamlit run app.py
```

It needs the trained weights at
`experiments/checkpoints/baseline_yolo_gpu_full/weights/best.pt`, which are not stored in
this repo.

Train the baseline (a GPU is recommended):

```bash
python -m src.train --config experiments/configs/baseline_yolo_gpu_full.yaml
```

Run detection on one image:

```bash
python -m src.inference --config experiments/configs/baseline_yolo_gpu_full.yaml --weights experiments/checkpoints/baseline_yolo_gpu_full/weights/best.pt --image path/to/image.jpg
```

Run the tests:

```bash
pytest
```

## Project layout

```
app.py                      Streamlit demo
src/
  train.py  evaluate.py  inference.py  domain_guard.py
  models/
    baseline/               YOLOv8 and Faster R-CNN baselines
    scale_fusion/           tiling detector and fusion head
    severity_scoring/       zone scoring and overlays
scripts/                    dataset checks, subsets and tiling
experiments/configs/        training configs
tests/
```

## Severity scoring

Each image is split into a 3x3 grid. Every zone gets a score from its detections, and is
labelled low (below 0.10), medium (below 0.30) or high.

## Limitations

- Trained only on aerial waste scenes. Other kinds of images give unreliable results.
- Very dense, overlapping piles of waste are under-detected, and the severity score
  depends on the detections, so it is too low in those cases.
- The tiling and fusion variants have not been trained, so their benefit is unproven.

## Licenses

The dataset is CC BY-SA 4.0. Ultralytics YOLO is AGPL-3.0, so check its terms before
reusing this code in a public or commercial project.