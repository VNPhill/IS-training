"""
dataset_voc.py — Pascal VOC dataset pipeline.

DROP-IN REPLACEMENT for dataset.py.
To switch from COCO → VOC:
    cp dataset.py dataset_coco_backup.py
    cp dataset_voc.py dataset.py

Requires config_voc.py to be active as config.py.

Expected directory layout (DATA_DIR = 'data/voc'):

    data/voc/
      VOCdevkit/
        VOC2007/
          Annotations/         ← 000001.xml  …
          ImageSets/Main/      ← trainval.txt, test.txt, train.txt, val.txt
          JPEGImages/          ← 000001.jpg  …
        VOC2012/
          Annotations/
          ImageSets/Main/      ← trainval.txt, train.txt, val.txt
          JPEGImages/

Two output formats (identical API to dataset.py):

    'ssd'  Pre-encoded anchor targets.
           Yields: (images [B,300,300,3],
                    loc_targets [B,8732,4],
                    cls_targets [B,8732])

    'raw'  Padded raw GT for anchor-free models.
           Yields: (images [B,300,300,3],
                    gt_boxes  [B,MAX_GT,4],
                    gt_labels [B,MAX_GT],
                    num_valid [B])

Annotation notes:
    • Boxes in VOC XML are [xmin, ymin, xmax, ymax] in pixel coordinates.
    • Difficult objects are SKIPPED during training but included (and
      properly handled) during evaluation.
    • Truncated objects are kept.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from config import (
    DATA_DIR, INPUT_SIZE, BATCH_SIZE,
    CLASS_TO_IDX, NUM_ANCHORS, MAX_GT,
    VOC_TRAIN_SETS, VOC_VAL_SETS,
)
from anchors import generate_anchors, encode_boxes

_ANCHORS = generate_anchors()   # [8732, 4]


# ─────────────────────────── Annotation Loading ──────────────────────────────

def load_voc_image_list(data_dir: str,
                        year_split_pairs: list) -> List[Tuple[str, str]]:
    """
    Build a list of (image_path, xml_path) pairs from one or more
    VOC year/split combinations.

    Args:
        data_dir        : root data directory  (e.g. 'data/voc')
        year_split_pairs: list of (year_str, split_str)
                          e.g. [('2007', 'trainval'), ('2012', 'trainval')]

    Returns:
        List of (jpeg_path, xml_path) tuples, deduplicated.
    """
    seen   = set()
    result = []

    for year, split in year_split_pairs:
        devkit  = os.path.join(data_dir, 'VOCdevkit', f'VOC{year}')
        ids_file = os.path.join(devkit, 'ImageSets', 'Main', f'{split}.txt')

        if not os.path.exists(ids_file):
            raise FileNotFoundError(
                f"VOC split file not found: {ids_file}\n"
                f"Check that DATA_DIR='{data_dir}' is correct and the "
                f"VOCdevkit is extracted there."
            )

        with open(ids_file) as f:
            ids = [line.strip() for line in f if line.strip()]

        for img_id in ids:
            key = (year, img_id)
            if key in seen:
                continue
            seen.add(key)
            jpg = os.path.join(devkit, 'JPEGImages', f'{img_id}.jpg')
            xml = os.path.join(devkit, 'Annotations', f'{img_id}.xml')
            result.append((jpg, xml))

    return result


def parse_voc_xml(xml_path: str,
                  skip_difficult: bool = True
                  ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Parse a VOC annotation XML file.

    Args:
        xml_path       : path to the .xml annotation file
        skip_difficult : if True, objects marked difficult=1 are excluded

    Returns:
        boxes   : [N, 4] float32  [xmin, ymin, xmax, ymax]  pixel coords
        labels  : [N]    int32    1-based class indices  (0 = background)
        img_w   : image width  in pixels
        img_h   : image height in pixels
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find('size')
    img_w = int(size_node.find('width').text)
    img_h = int(size_node.find('height').text)

    boxes, labels = [], []

    for obj in root.findall('object'):
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        if skip_difficult and difficult:
            continue

        name = obj.find('name').text.strip().lower()
        if name not in CLASS_TO_IDX:
            continue        # unknown class — skip silently

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Skip degenerate boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[name])

    if not boxes:
        return (np.zeros((0, 4), np.float32),
                np.zeros(0, np.int32),
                img_w, img_h)

    return (np.array(boxes,  np.float32),
            np.array(labels, np.int32),
            img_w, img_h)


def parse_voc_boxes(xml_path: str,
                    skip_difficult: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a VOC XML annotation and return normalised [cx, cy, w, h] boxes
    and 1-based labels — the same output format as parse_coco_boxes().

    This is the function called by dataset loading AND by evaluate_voc.py.

    Args:
        xml_path       : path to .xml file
        skip_difficult : skip difficult objects (True for training)

    Returns:
        gt_boxes  : [N, 4] float32  [cx, cy, w, h]  normalised to [0, 1]
        gt_labels : [N]    int32    1-based class indices
    """
    boxes_px, labels, img_w, img_h = parse_voc_xml(
        xml_path, skip_difficult=skip_difficult)

    if len(boxes_px) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.int32))

    xmin, ymin, xmax, ymax = (boxes_px[:, 0], boxes_px[:, 1],
                               boxes_px[:, 2], boxes_px[:, 3])

    cx = np.clip((xmin + xmax) / 2.0 / img_w, 0.0, 1.0)
    cy = np.clip((ymin + ymax) / 2.0 / img_h, 0.0, 1.0)
    w  = np.clip((xmax - xmin) / img_w,        0.0, 1.0)
    h  = np.clip((ymax - ymin) / img_h,        0.0, 1.0)

    return (np.stack([cx, cy, w, h], axis=-1).astype(np.float32), labels)


# ─────────────────────────── Augmentation ────────────────────────────────────
# Identical to dataset.py — shared augmentation helpers.

def _random_flip(img: np.ndarray, boxes: np.ndarray):
    if np.random.random() < 0.5:
        img = img[:, ::-1, :]
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 0] = 1.0 - boxes[:, 0]
    return img, boxes


def _photo_distortion(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.1)
    return tf.clip_by_value(img, 0.0, 255.0)


def _random_crop(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                 min_iou: float = 0.5):
    if len(boxes) == 0:
        return img, boxes, labels
    h, w = img.shape[:2]
    for _ in range(50):
        scale  = np.random.uniform(0.3, 1.0)
        aspect = np.random.uniform(0.5, 2.0)
        nh = max(1, min(int(h * scale), h))
        nw = max(1, min(int(w * scale * aspect), w))
        top  = np.random.randint(0, h - nh + 1)
        left = np.random.randint(0, w - nw + 1)
        cx1, cy1 = left / w,         top / h
        cx2, cy2 = (left + nw) / w,  (top + nh) / h
        bx1 = boxes[:, 0] - boxes[:, 2] / 2
        by1 = boxes[:, 1] - boxes[:, 3] / 2
        bx2 = boxes[:, 0] + boxes[:, 2] / 2
        by2 = boxes[:, 1] + boxes[:, 3] / 2
        inter = (np.maximum(0, np.minimum(bx2, cx2) - np.maximum(bx1, cx1)) *
                 np.maximum(0, np.minimum(by2, cy2) - np.maximum(by1, cy1)))
        area  = (bx2 - bx1) * (by2 - by1)
        iou   = inter / (area + 1e-10)
        if np.all(iou >= min_iou):
            cropped   = img[top:top + nh, left:left + nw]
            cx = (boxes[:, 0] - cx1) / (cx2 - cx1)
            cy = (boxes[:, 1] - cy1) / (cy2 - cy1)
            bw = boxes[:, 2] / (cx2 - cx1)
            bh = boxes[:, 3] / (cy2 - cy1)
            new_boxes = np.stack([cx, cy, bw, bh], axis=-1)
            valid = (cx > 0) & (cx < 1) & (cy > 0) & (cy < 1)
            if valid.any():
                return cropped, new_boxes[valid], labels[valid]
    return img, boxes, labels


# ─────────────────────────── Dataset Class ───────────────────────────────────

class VOCDataset:
    """
    Pascal VOC dataset supporting both 'ssd' and 'raw' target formats.

    Args:
        samples       : list of (jpeg_path, xml_path) tuples
        augment       : whether to apply training augmentation
        target_format : 'ssd' or 'raw'
    """

    def __init__(self,
                 samples: List[Tuple[str, str]],
                 augment: bool = False,
                 target_format: str = 'ssd'):
        self.samples       = samples
        self.augment       = augment
        self.target_format = target_format
        print(f"[Dataset] VOC  {len(samples)} images  "
              f"(format='{target_format}'  augment={augment})")

    def _load_raw(self, jpg_path: str, xml_path: str):
        """Load, augment, and preprocess one image + GT boxes."""
        raw = tf.io.read_file(jpg_path)
        img = tf.image.decode_jpeg(raw, channels=3)
        gt_boxes, gt_labels = parse_voc_boxes(xml_path, skip_difficult=True)

        if self.augment:
            img    = _photo_distortion(tf.cast(img, tf.float32))
            img_np = img.numpy().astype(np.uint8)
            img_np, gt_boxes              = _random_flip(img_np, gt_boxes)
            img_np, gt_boxes, gt_labels   = _random_crop(img_np, gt_boxes, gt_labels)
            img = tf.constant(img_np, dtype=tf.float32)
        else:
            img = tf.cast(img, tf.float32)

        img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
        img = img / 127.5 - 1.0
        return img, gt_boxes, gt_labels

    def load_ssd_sample(self, jpg_path: str, xml_path: str):
        """Return (image, loc_targets, cls_targets) for SSD training."""
        img, gt_boxes, gt_labels = self._load_raw(jpg_path, xml_path)
        loc_t, cls_t = encode_boxes(gt_boxes, gt_labels, _ANCHORS)
        return img, loc_t, cls_t

    def load_raw_sample(self, jpg_path: str, xml_path: str):
        """Return (image, padded_boxes, padded_labels, num_valid)."""
        img, gt_boxes, gt_labels = self._load_raw(jpg_path, xml_path)
        n      = len(gt_boxes)
        actual = min(n, MAX_GT)
        padded_boxes  = np.zeros((MAX_GT, 4), np.float32)
        padded_labels = np.zeros((MAX_GT,),   np.int32)
        if actual > 0:
            padded_boxes[:actual]  = gt_boxes[:actual]
            padded_labels[:actual] = gt_labels[:actual]
        return img, padded_boxes, padded_labels, np.int32(actual)

    def as_tf_dataset(self) -> tf.data.Dataset:
        if self.target_format == 'ssd':
            def _gen():
                for jpg, xml in self.samples:
                    try:
                        yield self.load_ssd_sample(jpg, xml)
                    except Exception as e:
                        print(f"[Dataset] skip {jpg}: {e}")

            return tf.data.Dataset.from_generator(
                _gen,
                output_signature=(
                    tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 3), tf.float32),
                    tf.TensorSpec((NUM_ANCHORS, 4),            tf.float32),
                    tf.TensorSpec((NUM_ANCHORS,),              tf.int32),
                ),
            )
        else:   # 'raw'
            def _gen():
                for jpg, xml in self.samples:
                    try:
                        yield self.load_raw_sample(jpg, xml)
                    except Exception as e:
                        print(f"[Dataset] skip {jpg}: {e}")

            return tf.data.Dataset.from_generator(
                _gen,
                output_signature=(
                    tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 3), tf.float32),
                    tf.TensorSpec((MAX_GT, 4),                 tf.float32),
                    tf.TensorSpec((MAX_GT,),                   tf.int32),
                    tf.TensorSpec((),                          tf.int32),
                ),
            )


# ─────────────────────────── Public API ──────────────────────────────────────

def build_dataset(split: str = 'train',
                  batch_size: int = BATCH_SIZE,
                  data_dir: str = DATA_DIR,
                  target_format: str = 'ssd') -> tf.data.Dataset:
    """
    Build a batched, prefetched tf.data.Dataset for Pascal VOC.

    Identical signature to the COCO version — drop-in replacement.

    Args:
        split         : 'train'  → VOC2007 trainval + VOC2012 trainval
                        'val'    → VOC2007 test
                        'test'   → VOC2007 test  (same as val)
        batch_size    : samples per batch
        data_dir      : root path  (must contain VOCdevkit/)
        target_format : 'ssd' or 'raw'

    Returns:
        tf.data.Dataset
    """
    if split == 'train':
        year_splits = VOC_TRAIN_SETS
        augment     = True
    else:
        year_splits = VOC_VAL_SETS
        augment     = False

    samples = load_voc_image_list(data_dir, year_splits)
    print(f"[Dataset] split='{split}'  "
          f"years={[f'VOC{y}/{s}' for y, s in year_splits]}  "
          f"{len(samples)} images")

    ds = VOCDataset(samples,
                    augment=augment,
                    target_format=target_format).as_tf_dataset()

    if split == 'train':
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────── Compatibility shims ─────────────────────────────────
# evaluate.py imports load_coco_annotations / parse_coco_boxes by name.
# These shims mean evaluate.py keeps working after the swap without any edits.

def load_coco_annotations(ann_json: str):
    """
    Compatibility shim — not used by VOC evaluate_voc.py directly, but
    prevents ImportError if the old evaluate.py is accidentally run.
    Raises a clear error pointing to evaluate_voc.py instead.
    """
    raise RuntimeError(
        "load_coco_annotations() is not available in VOC mode.\n"
        "Use evaluate_voc.py instead of evaluate.py."
    )


def parse_coco_boxes(anns, img_w, img_h):
    """Compatibility shim — same as above."""
    raise RuntimeError(
        "parse_coco_boxes() is not available in VOC mode.\n"
        "Use evaluate_voc.py instead of evaluate.py."
    )