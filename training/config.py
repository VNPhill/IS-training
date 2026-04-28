"""
config.py — Hyperparameters and constants for all detection models.

Supported MODEL_TYPE values:
    SSD family   : 'mobilenet_ssd' | 'mobilenetv2_ssd' | 'vgg_ssd' | 'resnet_ssd'
    RetinaNet    : 'retinanet'
    YOLOv3       : 'yolov3'
    FCOS         : 'fcos'
    CenterNet    : 'centernet'

To switch models, change MODEL_TYPE (or pass --model on the CLI).
Checkpoints, logs, and results are saved under per-model subdirectories.
"""

# ─────────────────────────── Model Selection ─────────────────────────────────

MODEL_TYPE  = 'mobilenet_ssd'   # change this to pick your architecture
MODEL_WIDTH = 1.0               # channel-width multiplier

# ─────────────────────────── Dataset / Classes ───────────────────────────────

COCO_CLASSES = [
    'person',         'bicycle',       'car',           'motorcycle',    'airplane',
    'bus',            'train',         'truck',         'boat',          'traffic light',
    'fire hydrant',   'stop sign',     'parking meter', 'bench',         'bird',
    'cat',            'dog',           'horse',         'sheep',         'cow',
    'elephant',       'bear',          'zebra',         'giraffe',       'backpack',
    'umbrella',       'handbag',       'tie',           'suitcase',      'frisbee',
    'skis',           'snowboard',     'sports ball',   'kite',          'baseball bat',
    'baseball glove', 'skateboard',    'surfboard',     'tennis racket', 'bottle',
    'wine glass',     'cup',           'fork',          'knife',         'spoon',
    'bowl',           'banana',        'apple',         'sandwich',      'orange',
    'broccoli',       'carrot',        'hot dog',       'pizza',         'donut',
    'cake',           'chair',         'couch',         'potted plant',  'bed',
    'dining table',   'toilet',        'tv',            'laptop',        'mouse',
    'remote',         'keyboard',      'cell phone',    'microwave',     'oven',
    'toaster',        'sink',          'refrigerator',  'book',          'clock',
    'vase',           'scissors',      'teddy bear',    'hair drier',    'toothbrush',
]

COCO_CAT_IDS = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

COCO_ID_TO_LABEL = {cat_id: idx + 1 for idx, cat_id in enumerate(COCO_CAT_IDS)}
NUM_CLASSES         = len(COCO_CLASSES)         # 80 foreground classes
NUM_CLASSES_WITH_BG = NUM_CLASSES + 1           # 81
CLASS_TO_IDX        = {cls: idx + 1 for idx, cls in enumerate(COCO_CLASSES)}
IDX_TO_CLASS        = {idx + 1: cls for idx, cls in enumerate(COCO_CLASSES)}

# ─────────────────────────── Raw-GT Dataset Format ───────────────────────────
# Anchor-free models receive raw (padded) ground-truth instead of pre-encoded
# anchor targets.  MAX_GT is the maximum GT boxes per image in a batch.

MAX_GT = 100

# ──────────────────────────── SSD Anchor Config ──────────────────────────────

FEATURE_MAP_SIZES  = [38, 19, 10, 5, 3, 1]
ANCHORS_PER_CELL   = [4,  6,  6,  6, 4, 4]
NUM_ANCHORS        = sum(a * f * f for a, f in zip(ANCHORS_PER_CELL, FEATURE_MAP_SIZES))
ANCHOR_SCALES      = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075]
ANCHOR_ASPECT_RATIOS = [[2], [2,3], [2,3], [2,3], [2], [2]]
ENCODE_VARIANCES   = (0.1, 0.2)

# ─────────────────────────── RetinaNet Config ────────────────────────────────
# Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
#
# FPN levels P3–P7; 9 anchors per cell (3 scales × 3 ratios).
# Focal loss replaces hard-negative mining.

RETINA_FPN_CHANNELS   = 256       # channels in every FPN / subnet layer
RETINA_NUM_CONVS      = 4         # conv layers in class + box subnets
RETINA_ANCHOR_SCALES  = [1.0, 2**(1/3), 2**(2/3)]   # 3 scales per level
RETINA_ANCHOR_RATIOS  = [0.5, 1.0, 2.0]              # 3 aspect ratios
RETINA_ANCHORS_PER_CELL = len(RETINA_ANCHOR_SCALES) * len(RETINA_ANCHOR_RATIOS)  # 9
# Base anchor sizes (in normalised coords) for FPN levels P3–P7
RETINA_ANCHOR_BASE_SIZES = [0.04, 0.08, 0.16, 0.32, 0.64]  # strides 8,16,32,64,128
RETINA_IOU_POS     = 0.5      # IoU ≥ this → positive
RETINA_IOU_NEG     = 0.4      # IoU <  this → negative (between = ignore)
RETINA_FOCAL_ALPHA = 0.25
RETINA_FOCAL_GAMMA = 2.0

# ─────────────────────────── YOLOv3 Config ───────────────────────────────────
# Reference: Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018.
#
# Three detection scales (strides 8, 16, 32) with 3 anchors each.
# Anchor WH in pixels at INPUT_SIZE=300.

YOLO_ANCHORS = [
    # stride 8  — small objects   (feature map ~38×38)
    [(10, 13),  (16, 30),   (33, 23)],
    # stride 16 — medium objects  (feature map ~19×19)
    [(30, 61),  (62, 45),   (59, 119)],
    # stride 32 — large objects   (feature map ~10×10)
    [(116, 90), (156, 198), (373, 326)],
]
YOLO_STRIDES       = [8, 16, 32]
YOLO_IOU_IGNORE    = 0.5     # anchors with IoU > this to any GT are ignored
YOLO_LAMBDA_OBJ    = 1.0
YOLO_LAMBDA_NOOBJ  = 0.5
YOLO_LAMBDA_CLASS  = 1.0
YOLO_LAMBDA_BOX    = 5.0

# ─────────────────────────── FCOS Config ─────────────────────────────────────
# Reference: Tian et al., "FCOS: Fully Convolutional One-Stage Object
#            Detection", ICCV 2019.
#
# Anchor-free: every FPN point predicts (l, t, r, b) distances to box edges
# plus a centerness score that suppresses low-quality predictions.

FCOS_FPN_CHANNELS = 256
FCOS_NUM_CONVS    = 4
FCOS_STRIDES      = [8, 16, 32, 64, 128]     # P3–P7
# Maximum regression distance allowed at each FPN level.
# Points whose GT box falls outside the level's range are ignored.
FCOS_REGRESS_RANGES = [
    (-1,    64),    # P3 — small objects
    (64,   128),    # P4
    (128,  256),    # P5
    (256,  512),    # P6
    (512,  1e8),    # P7 — large objects
]
FCOS_CENTERNESS_ON_REG = True   # predict centerness on the regression branch

# ─────────────────────────── CenterNet Config ────────────────────────────────
# Reference: Zhou et al., "Objects as Points", 2019.
#
# Anchor-free: object centers are predicted as Gaussian heatmap peaks.
# Output stride 8 → heatmap size 38×38 for 300×300 input.

CENTERNET_OUTPUT_STRIDE = 8     # 300 / 8 = ~38
CENTERNET_HEATMAP_SIZE  = 38    # (INPUT_SIZE // OUTPUT_STRIDE)
CENTERNET_MIN_OVERLAP   = 0.7   # min IoU for Gaussian radius (TTFNet criterion)
CENTERNET_LAMBDA_HMAP   = 1.0
CENTERNET_LAMBDA_SIZE   = 0.1
CENTERNET_LAMBDA_OFFSET = 1.0
# Decoder: number of upsampling blocks in the ResNet deconv head
CENTERNET_DECONV_CHANNELS = [256, 128, 64]   # channels at each upsample

# ─────────────────────────── Training Hyperparameters ────────────────────────

INPUT_SIZE   = 300
BATCH_SIZE   = 16
NUM_EPOCHS   = 150
LR_INIT      = 1e-2
LR_STEPS     = [30, 70, 120]
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4

NEG_POS_RATIO    = 3     # SSD hard-negative mining
IOU_MATCH_THRESH = 0.5   # SSD anchor assignment

# ─────────────────────────── Paths ───────────────────────────────────────────

DATA_DIR       = '../data/coco'
CHECKPOINT_DIR = f'checkpoints/{MODEL_TYPE}'
LOG_DIR        = f'logs/{MODEL_TYPE}'
TFLITE_PATH    = f'outputs/{MODEL_TYPE}.tflite'
