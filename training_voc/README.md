# Dataset Swap Guide — COCO ↔ Pascal VOC

This project ships two complete, parallel dataset implementations.
Switch between them by swapping two files.  Nothing else changes.

---

## File map

| Purpose | COCO version | VOC version |
|---|---|---|
| Config / hyperparameters | `config.py` | `config_voc.py` |
| Dataset pipeline | `dataset.py` | `dataset_voc.py` |
| Evaluation | `evaluate.py` | `evaluate_voc.py` |
| Labelmap export | `export_labelmap.py` | `export_labelmap_voc.py` |

`train.py`, `export_tflite.py`, all model files, all loss files, and
`anchors.py` are **not touched** — they work with both datasets.

---

## Switch COCO → VOC  (activate VOC)

```bash
# 1. Back up the current COCO files
cp config.py  config_coco.py
cp dataset.py dataset_coco.py

# 2. Activate the VOC files
cp config_voc.py  config.py
cp dataset_voc.py dataset.py

# Done — train.py, evaluate_voc.py, and export_tflite.py all work now.
```

---

## Switch VOC → COCO  (restore COCO)

```bash
cp config_coco.py  config.py
cp dataset_coco.py dataset.py
```

---

## VOC dataset setup

### Download

```bash
mkdir -p data/voc/VOCdevkit

# VOC 2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar -C data/voc/
tar -xf VOCtest_06-Nov-2007.tar     -C data/voc/

# VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar -C data/voc/
```

### Expected layout after extraction

```
data/voc/
  VOCdevkit/
    VOC2007/
      Annotations/       ← 000001.xml  000002.xml  …
      ImageSets/
        Main/            ← train.txt  val.txt  trainval.txt  test.txt
      JPEGImages/        ← 000001.jpg  000002.jpg  …
    VOC2012/
      Annotations/
      ImageSets/
        Main/            ← train.txt  val.txt  trainval.txt
      JPEGImages/
```

### Split sizes

| Split | Source | Images |
|---|---|---|
| train | VOC2007 trainval + VOC2012 trainval | 16,551 |
| val / test | VOC2007 test | 4,952 |

---

## Full VOC workflow

### 1 — Activate VOC

```bash
cp config.py  config_coco.py   # backup COCO config
cp dataset.py dataset_coco.py  # backup COCO dataset

cp config_voc.py  config.py
cp dataset_voc.py dataset.py
```

### 2 — Train

Identical commands to the COCO workflow:

```bash
python train.py --model mobilenet_ssd
python train.py --model retinanet
python train.py --model yolov3
# Extra flags
python train.py --model retinanet --epochs 200 --batch 16 --lr 0.01
# etc.
```

Checkpoints and logs go to the same per-model directories:
```
checkpoints/<model>/best_model.weights.h5
logs/<model>/train/  val/
```

### 3 — Evaluate

Use `evaluate_voc.py` instead of `evaluate.py`:

```bash
python evaluate_voc.py --model mobilenet_ssd
python evaluate_voc.py --model retinanet --conf 0.05
```

Results saved to:
```
results/<model>/map_results_voc.txt
results/<model>/eval_voc.log
```

### 4 — Export to TFLite

`export_tflite.py` is unchanged — it works with both datasets:

```bash
python export_tflite.py --model mobilenet_ssd
```

Use `export_labelmap_voc.py` instead of `export_labelmap.py` to generate
the correct 20-class labelmap:

```bash
python export_labelmap_voc.py --model mobilenet_ssd
```

Output:
```
outputs/mobilenet_ssd/detect_voc.tflite
outputs/mobilenet_ssd/labelmap_voc.txt      ← 21 lines: ??? + 20 VOC classes
```

### 5 — Copy to Flutter

```bash
cp outputs/mobilenet_ssd/detect_voc.tflite  <flutter>/assets/models/
cp outputs/mobilenet_ssd/labelmap_voc.txt   <flutter>/assets/models/
```

---

## Key differences between COCO and VOC datasets

| Property | COCO 2017 | Pascal VOC |
|---|---|---|
| Classes | 80 | 20 |
| Train images | 118,287 | 16,551 (2007+2012 trainval) |
| Val/test images | 5,000 | 4,952 (2007 test) |
| Annotation format | JSON (instances) | XML (per image) |
| Category IDs | Non-contiguous (1–90 with gaps) | Name strings in XML |
| Difficult flag | Not present | Yes — excluded from GT counts |
| Crowd flag | Yes — skipped | Not present |
| Download size | ~25 GB (train+val images) | ~2.5 GB (2007+2012) |

### Difficult objects

VOC XML annotations include a `<difficult>` flag for objects that are
ambiguous, heavily occluded, or very small.  `evaluate_voc.py` handles
this correctly:

- A difficult GT box is **not counted** toward `num_gt` in the AP
  calculation.
- A detection that matches only a difficult GT box is **neither TP nor FP**
  — it is simply ignored.
- A detection that matches a non-difficult GT box that was already matched
  is counted as **FP** (duplicate detection).

This matches the official VOC devkit evaluation protocol exactly.

---

## Config differences at a glance

The only values that differ between `config.py` (COCO) and `config_voc.py`:

| Constant | COCO | VOC |
|---|---|---|
| `COCO_CLASSES` / `VOC_CLASSES` | 80 classes | 20 classes |
| `NUM_CLASSES` | 80 | 20 |
| `NUM_CLASSES_WITH_BG` | 81 | 21 |
| `MAX_GT` | 100 | 50 |
| `DATA_DIR` | `data/coco` | `data/voc` |
| `BATCH_SIZE` | 16 | 32 |
| `NUM_EPOCHS` | 200 | 120 |
| `LR_STEPS` | `[120, 160]` | `[80, 100]` |

Everything else — anchor configs, model architecture constants, loss
weights — is identical.

---

## Expected mAP on VOC 2007 test (mAP@0.50)

| Model | mAP | Notes |
|---|---|---|
| `mobilenet_ssd` | ~68–72 | Fast, good baseline |
| `mobilenetv2_ssd` | ~70–74 | Slightly better than V1 |
| `vgg_ssd` | ~77–79 | Original SSD paper result |
| `resnet_ssd` | ~79–81 | Stronger backbone |
| `retinanet` | ~81–83 | Focal loss benefit clear on VOC |
| `yolov3` | ~82–85 | Fast and accurate |
| `fcos` | ~82–84 | Anchor-free, no tuning of anchors |
| `centernet` | ~79–82 | Fastest inference |

VOC numbers are higher than COCO because VOC has fewer and larger objects,
and the 20-class problem is simpler than 80 classes.

