"""
evaluate_voc.py — mAP evaluation on Pascal VOC 2007 test set.

DROP-IN REPLACEMENT for evaluate.py when using Pascal VOC.

Usage:
    python evaluate_voc.py                          # uses config.MODEL_TYPE
    python evaluate_voc.py --model retinanet
    python evaluate_voc.py --model yolov3 --conf 0.25
    python evaluate_voc.py --model centernet --ckpt checkpoints/centernet/best_model.weights.h5

Evaluation protocol:
    • IoU threshold: 0.50  (VOC standard)
    • Difficult objects are EXCLUDED from both GT counts and detection
      matching — a detection that overlaps only a difficult GT is not
      penalised as a false positive.
    • AP computed with the VOC 2010+ area-under-curve method (same as
      evaluate.py, not the 11-point interpolation used in VOC 2007 paper).

Requires config_voc.py to be active as config.py
and      dataset_voc.py to be active as dataset.py.
"""

import os
import argparse
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import tensorflow as tf

from models           import get_detector, AVAILABLE_MODELS
from dataset          import load_voc_image_list, parse_voc_xml
from utils.logger     import setup_logging
from config import (
    DATA_DIR, INPUT_SIZE,
    NUM_CLASSES, NUM_CLASSES_WITH_BG,
    VOC_CLASSES, MODEL_TYPE, MODEL_WIDTH,
    VOC_TEST_SETS,
)


# ──────────────────────────── CLI ────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate any detection model on Pascal VOC 2007 test.')
    p.add_argument('--model',    default=MODEL_TYPE, choices=AVAILABLE_MODELS)
    p.add_argument('--width',    type=float, default=MODEL_WIDTH)
    p.add_argument('--ckpt',     default=None,
                   help='Explicit checkpoint path '
                        '(default: checkpoints/<model>/best_model.weights.h5)')
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--iou',      type=float, default=0.5,
                   help='IoU threshold for TP matching (default: 0.50)')
    p.add_argument('--conf',     type=float, default=0.05,
                   help='Confidence threshold before NMS (default: 0.05)')
    p.add_argument('--nms_iou',  type=float, default=0.45)
    return p.parse_args()


# ──────────────────────────── IoU (scalar) ───────────────────────────────────

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter    = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a   = (a[2] - a[0]) * (a[3] - a[1])
    area_b   = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-10)


# ──────────────────────────── AP Computation ─────────────────────────────────

def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC 2010+ AP: area under the max-smoothed PR curve."""
    mrec = np.concatenate(([0.0], recall,    [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    pts = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[pts + 1] - mrec[pts]) * mpre[pts + 1]))


# ──────────────────────────── Main mAP loop ──────────────────────────────────

def compute_map(model,
                detector,
                model_type:     str,
                data_dir:       str   = DATA_DIR,
                iou_threshold:  float = 0.5,
                conf_threshold: float = 0.05,
                nms_iou:        float = 0.45) -> float:
    """
    Run inference on VOC 2007 test, compute per-class AP and mAP.

    Saves results to  results/<model_type>/map_results_voc.txt .

    Difficult objects are excluded from ground-truth counts and from
    detection matching so they don't affect the final numbers.

    Args:
        model          : loaded tf.keras.Model
        detector       : corresponding DetectionModel instance
        model_type     : string key used for results path
        data_dir       : path to data/voc/  (contains VOCdevkit/)
        iou_threshold  : IoU threshold for TP matching
        conf_threshold : minimum confidence score to keep a detection
        nms_iou        : NMS IoU threshold

    Returns:
        mAP scalar (float)
    """
    # ── Load VOC 2007 test image list ─────────────────────────────────────────
    samples = load_voc_image_list(data_dir, VOC_TEST_SETS)
    print(f"\n[Eval] Model       : {model_type}")
    print(f"[Eval] Test images : {len(samples)}  (VOC 2007 test)")
    print(f"[Eval] IoU={iou_threshold}  Conf={conf_threshold}  NMS={nms_iou}\n")

    # cls_idx → [(score, img_id, box_xyxy_normalised)]
    detections    = defaultdict(list)
    # cls_idx → [(img_id, box_xyxy_normalised, is_difficult)]
    ground_truths = defaultdict(list)

    for n, (jpg_path, xml_path) in tqdm(enumerate(samples), desc="Evaluate", total=len(samples), leave=False):
        if (n + 1) % 500 == 0:
            print(f"  [{n + 1}/{len(samples)}] …")

        img_id = os.path.splitext(os.path.basename(jpg_path))[0]

        # ── Preprocess ───────────────────────────────────────────────────────
        try:
            raw = tf.io.read_file(jpg_path)
            img = tf.image.decode_jpeg(raw, channels=3)
        except Exception as exc:
            print(f"  [Eval] Skip {img_id}: {exc}")
            continue

        img_t = tf.image.resize(tf.cast(img, tf.float32),
                                [INPUT_SIZE, INPUT_SIZE])
        img_t = img_t / 127.5 - 1.0
        img_t = img_t[tf.newaxis]                      # [1, 300, 300, 3]

        # ── Inference ────────────────────────────────────────────────────────
        raw_preds = model(img_t, training=False)

        if isinstance(raw_preds, (list, tuple)):
            preds_one = [p[0] for p in raw_preds]
        else:
            preds_one = raw_preds[0]

        det_boxes, det_scores, det_labels = detector.postprocess(
            preds_one,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
        )

        for box, score, label in zip(det_boxes, det_scores, det_labels):
            detections[int(label)].append((float(score), img_id, box))

        # ── Ground-truth (include difficult objects with a flag) ─────────────
        # parse_voc_xml with skip_difficult=False so we load ALL objects
        boxes_px, labels_all, img_w, img_h = parse_voc_xml(
            xml_path, skip_difficult=False)
        # Re-parse with skip_difficult=False means difficult objects ARE in the
        # list; we need to know which ones are difficult so they can be ignored
        # in matching without affecting the FP count.

        # Build difficult mask by re-reading the XML
        tree    = __import__('xml.etree.ElementTree', fromlist=['ElementTree']).parse(xml_path)
        root    = tree.getroot()
        size    = root.find('size')
        iw      = int(size.find('width').text)
        ih      = int(size.find('height').text)

        for obj in root.findall('object'):
            name      = obj.find('name').text.strip().lower()
            from config import CLASS_TO_IDX
            if name not in CLASS_TO_IDX:
                continue
            label_idx = CLASS_TO_IDX[name]
            difficult = int(obj.find('difficult').text) \
                        if obj.find('difficult') is not None else 0

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / iw
            ymin = float(bndbox.find('ymin').text) / ih
            xmax = float(bndbox.find('xmax').text) / iw
            ymax = float(bndbox.find('ymax').text) / ih

            xyxy = np.array([xmin, ymin, xmax, ymax], np.float32)
            ground_truths[label_idx].append((img_id, xyxy, bool(difficult)))

    # ── Per-class AP ─────────────────────────────────────────────────────────
    aps   = []
    lines = [
        f"Model          : {model_type}",
        f"Dataset        : Pascal VOC 2007 test",
        f"IoU threshold  : {iou_threshold}",
        f"Conf threshold : {conf_threshold}",
        f"NMS IoU        : {nms_iou}",
        "-" * 50,
    ]

    print()
    for cls_idx in range(1, NUM_CLASSES_WITH_BG):
        cls_name = VOC_CLASSES[cls_idx - 1]
        cls_dets = sorted(detections[cls_idx], key=lambda x: -x[0])
        cls_gts_all = ground_truths[cls_idx]

        if not cls_gts_all:
            msg = f"  {cls_name:<14s}  (no GT, skipped)"
            print(msg);  lines.append(msg.strip())
            continue

        # Group GTs by image; track difficult flag per GT box
        gt_by_img = defaultdict(list)
        for (img_id, box, is_diff) in cls_gts_all:
            gt_by_img[img_id].append({
                'box': box, 'difficult': is_diff, 'matched': False})

        # num_gt counts ONLY non-difficult objects
        num_gt = sum(1 for (_, _, d) in cls_gts_all if not d)
        if num_gt == 0:
            msg = f"  {cls_name:<14s}  (only difficult GT, skipped)"
            print(msg);  lines.append(msg.strip())
            continue

        tp = np.zeros(len(cls_dets), np.float32)
        fp = np.zeros(len(cls_dets), np.float32)

        for d, (score, img_id, det_box) in enumerate(cls_dets):
            gts = gt_by_img.get(img_id, [])
            if not gts:
                fp[d] = 1
                continue

            ious     = [_iou_xyxy(det_box, g['box']) for g in gts]
            best_idx = int(np.argmax(ious))
            best_iou = ious[best_idx]

            if best_iou >= iou_threshold:
                if not gts[best_idx]['matched']:
                    if gts[best_idx]['difficult']:
                        # Overlaps a difficult GT — neither TP nor FP
                        pass
                    else:
                        tp[d] = 1
                    gts[best_idx]['matched'] = True
                else:
                    # Already matched (duplicate detection)
                    if not gts[best_idx]['difficult']:
                        fp[d] = 1
            else:
                fp[d] = 1

        cum_tp    = np.cumsum(tp)
        cum_fp    = np.cumsum(fp)
        recall    = cum_tp / (num_gt + 1e-10)
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        ap        = _voc_ap(recall, precision)
        aps.append(ap)

        msg = f"  {cls_name:<14s}  AP={ap:.4f}  ({num_gt} GT non-difficult)"
        print(msg);  lines.append(msg.strip())

    mAP     = float(np.mean(aps)) if aps else 0.0
    summary = (f"\n  mAP@{iou_threshold:.2f}: {mAP:.4f}  "
               f"({len(aps)}/{NUM_CLASSES} classes evaluated)")
    print(summary)
    lines.append(summary.strip())

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir  = os.path.join('results', model_type)
    results_path = os.path.join(results_dir, 'map_results_voc.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(results_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"\n[Eval] Results → {results_path}")

    return mAP


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == '__main__':
    args = _parse_args()

    results_dir = os.path.join('results', args.model)
    log_path    = setup_logging(log_dir=results_dir, filename='eval_voc.log')
    print(f"[Eval] Terminal output saved to {log_path}")

    ckpt_path = args.ckpt or os.path.join(
        'checkpoints', args.model, 'best_model.weights.h5')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train with:  python train.py --model {args.model}"
        )

    detector = get_detector(args.model)
    model    = detector.build(num_classes=NUM_CLASSES_WITH_BG,
                              width=args.width)

    print(f"[Eval] Loading weights from {ckpt_path}")
    model.load_weights(ckpt_path)

    compute_map(
        model          = model,
        detector       = detector,
        model_type     = args.model,
        data_dir       = args.data_dir,
        iou_threshold  = args.iou,
        conf_threshold = args.conf,
        nms_iou        = args.nms_iou,
    )
    
    