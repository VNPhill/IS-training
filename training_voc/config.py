"""
config_voc.py — Hyperparameters and constants for Pascal VOC training.

DROP-IN REPLACEMENT for config.py.
To switch from COCO → VOC:
    cp config.py config_coco_backup.py
    cp config_voc.py config.py

Pascal VOC classes:  20 foreground + 1 background = 21 total.
Training split  :  VOC2007 trainval + VOC2012 trainval  (16,551 images)
Validation split:  VOC2007 test                         ( 4,952 images)
Test split      :  VOC2007 test  (same — no held-out test with annotations)

All architecture constants (anchor configs, loss weights, etc.) are
identical to the COCO config.  Only the class list, class counts, and
DATA_DIR differ.
"""

# ─────────────────────────── Model Selection ─────────────────────────────────

MODEL_TYPE  = 'mobilenet_ssd'   # change this to pick your architecture
MODEL_WIDTH = 1.0               # channel-width multiplier

# ─────────────────────────── Dataset / Classes ───────────────────────────────

# Pascal VOC 20 categories (alphabetical order used in the official devkit)
VOC_CLASSES = [
    'aeroplane',   'bicycle',  'bird',       'boat',        'bottle',
    'bus',         'car',      'cat',        'chair',       'cow',
    'diningtable', 'dog',      'horse',      'motorbike',   'person',
    'pottedplant', 'sheep',    'sofa',       'train',       'tvmonitor',
]

NUM_CLASSES         = len(VOC_CLASSES)     # 20 foreground classes
NUM_CLASSES_WITH_BG = NUM_CLASSES + 1      # 21  (index 0 = background)

CLASS_TO_IDX = {cls: idx + 1 for idx, cls in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {idx + 1: cls for idx, cls in enumerate(VOC_CLASSES)}

# Aliases kept for compatibility with train.py / evaluate.py which reference
# COCO_CLASSES as a generic "class name list"
COCO_CLASSES = VOC_CLASSES

# VOC annotations do NOT use a numeric category-ID → label map;
# class names appear directly in the XML.  The dummy dict below keeps any
# code that imports COCO_ID_TO_LABEL from crashing.
COCO_ID_TO_LABEL = {}   # unused for VOC — see dataset_voc.py

# ─────────────────────────── Raw-GT Dataset Format ───────────────────────────

# VOC images rarely have more than 50 objects; 50 is a safe upper bound.
MAX_GT = 50

# ──────────────────────────── SSD Anchor Config ──────────────────────────────
# Identical to the COCO config — SSD anchors are input-resolution-dependent,
# not dataset-dependent.

FEATURE_MAP_SIZES  = [38, 19, 10, 5, 3, 1]
ANCHORS_PER_CELL   = [4,  6,  6,  6, 4, 4]
NUM_ANCHORS        = sum(a * f * f for a, f in zip(ANCHORS_PER_CELL, FEATURE_MAP_SIZES))
ANCHOR_SCALES      = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075]
ANCHOR_ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
ENCODE_VARIANCES   = (0.1, 0.2)

# ─────────────────────────── RetinaNet Config ────────────────────────────────

RETINA_FPN_CHANNELS     = 256
RETINA_NUM_CONVS        = 4
RETINA_ANCHOR_SCALES    = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]
RETINA_ANCHOR_RATIOS    = [0.5, 1.0, 2.0]
RETINA_ANCHORS_PER_CELL = len(RETINA_ANCHOR_SCALES) * len(RETINA_ANCHOR_RATIOS)  # 9
RETINA_ANCHOR_BASE_SIZES = [0.04, 0.08, 0.16, 0.32, 0.64]
RETINA_IOU_POS          = 0.5
RETINA_IOU_NEG          = 0.4
RETINA_FOCAL_ALPHA      = 0.25
RETINA_FOCAL_GAMMA      = 2.0

# ─────────────────────────── YOLOv3 Config ───────────────────────────────────

YOLO_ANCHORS = [
    [(10, 13),   (16, 30),   (33, 23)],
    [(30, 61),   (62, 45),   (59, 119)],
    [(116, 90),  (156, 198), (373, 326)],
]
YOLO_STRIDES      = [8, 16, 32]
YOLO_IOU_IGNORE   = 0.5
YOLO_LAMBDA_OBJ   = 1.0
YOLO_LAMBDA_NOOBJ = 0.5
YOLO_LAMBDA_CLASS = 1.0
YOLO_LAMBDA_BOX   = 5.0

# ─────────────────────────── FCOS Config ─────────────────────────────────────

FCOS_FPN_CHANNELS      = 256
FCOS_NUM_CONVS         = 4
FCOS_STRIDES           = [8, 16, 32, 64, 128]
FCOS_REGRESS_RANGES    = [(-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)]
FCOS_CENTERNESS_ON_REG = True

# ─────────────────────────── CenterNet Config ────────────────────────────────

CENTERNET_OUTPUT_STRIDE   = 8
CENTERNET_HEATMAP_SIZE    = 38
CENTERNET_MIN_OVERLAP     = 0.7
CENTERNET_LAMBDA_HMAP     = 1.0
CENTERNET_LAMBDA_SIZE     = 0.1
CENTERNET_LAMBDA_OFFSET   = 1.0
CENTERNET_DECONV_CHANNELS = [256, 128, 64]

# ─────────────────────────── Training Hyperparameters ────────────────────────
# VOC is ~7× smaller than COCO so we use a lighter schedule:
#   • Fewer epochs (120 vs 200)
#   • LR milestones adjusted proportionally
#   • Smaller shuffle buffer (the full dataset fits in ~6 GB RAM)

INPUT_SIZE   = 300
BATCH_SIZE   = 16       # can use larger batch since VOC is smaller
NUM_EPOCHS   = 80
LR_INIT      = 1e-2
LR_STEPS     = [20, 40, 60, 70] # divide 10x after each epoch
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4

NEG_POS_RATIO    = 3
IOU_MATCH_THRESH = 0.5

# ─────────────────────────── Paths ───────────────────────────────────────────
# Expected layout:
#   data/voc/
#     VOCdevkit/
#       VOC2007/
#         Annotations/   ← xxxxxx.xml
#         ImageSets/Main/← trainval.txt, test.txt …
#         JPEGImages/    ← xxxxxx.jpg
#       VOC2012/
#         Annotations/
#         ImageSets/Main/← trainval.txt …
#         JPEGImages/

DATA_DIR       = '../data/voc'
CHECKPOINT_DIR = f'checkpoints/{MODEL_TYPE}'
LOG_DIR        = f'logs/{MODEL_TYPE}'
TFLITE_PATH    = f'outputs/{MODEL_TYPE}.tflite'

# ── VOC-specific split configuration ─────────────────────────────────────────
# Each entry is (year, split) — the dataset loader will look for
#   data/voc/VOCdevkit/VOC<year>/ImageSets/Main/<split>.txt
#
# Standard SSD / VOC protocol:
#   Training  → 2007 trainval  +  2012 trainval  (16,551 images)
#   Val/Test  → 2007 test                         ( 4,952 images)

VOC_TRAIN_SETS = [('2007', 'trainval'), ('2012', 'trainval')]
VOC_VAL_SETS   = [('2007', 'test')]
VOC_TEST_SETS  = [('2007', 'test')]

