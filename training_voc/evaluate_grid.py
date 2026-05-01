"""
evaluate_voc_grid.py — Grid search over post-processing thresholds.

Strategy (two-phase):
    Phase 1 — Batched inference  (GPU, runs ONCE)
        Load all VOC2007 test images through a tf.data pipeline,
        run the model in batches, and cache every image's raw
        predictions in RAM as numpy arrays.

    Phase 2 — Grid sweep  (CPU, parallelised across classes)
        For each (conf, nms_iou, iou_threshold) combination apply
        NMS + confidence filtering to the cached predictions, then
        compute per-class AP with a ThreadPoolExecutor.
        No GPU is needed and no images are read from disk again.

This means a 5×5×3 = 75-point grid costs ≈ 2× a single evaluation run.

Usage:
    python evaluate_voc_grid.py --model mobilenet_ssd

    # Custom grid
    python evaluate_voc_grid.py --model retinanet \\
        --conf_values  0.01 0.05 0.1 0.3 0.5 \\
        --nms_values   0.35 0.45 0.50 0.55 \\
        --iou_values   0.50

    # Larger batch for faster inference on a big GPU
    python evaluate_voc_grid.py --model vgg_ssd --batch 32

Note on --iou_values:
    Changing the IoU matching threshold makes your mAP incomparable to
    published VOC numbers.  Keep it at 0.50 for reporting.  Multiple
    values are supported for research / debugging only.
"""

import os
import argparse
import itertools
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models       import get_detector, AVAILABLE_MODELS
from dataset      import load_voc_image_list, parse_voc_xml
from utils.logger import setup_logging
from config import (
    DATA_DIR, INPUT_SIZE,
    NUM_CLASSES, NUM_CLASSES_WITH_BG,
    VOC_CLASSES, MODEL_TYPE, MODEL_WIDTH,
    VOC_TEST_SETS,
)


# ──────────────────────────── CLI ────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Grid search over post-processing thresholds on VOC 2007 test.')

    p.add_argument('--model',   default=MODEL_TYPE, choices=AVAILABLE_MODELS)
    p.add_argument('--width',   type=float, default=MODEL_WIDTH)
    p.add_argument('--ckpt',    default=None,
                   help='Checkpoint path '
                        '(default: checkpoints/<model>/best_model.weights.h5)')
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--batch',   type=int, default=4,
                   help='Inference batch size (default: 16)')
    p.add_argument('--workers', type=int, default=8,
                   help='Parallel workers for AP computation (default: 8)')

    # Grid axes
    p.add_argument('--conf_values',  type=float, nargs='+',
                   default=[0.005, 0.01, 0.05, 0.1, 0.5],
                   help='Confidence threshold values to sweep')
    p.add_argument('--nms_values',   type=float, nargs='+',
                   default=[0.25, 0.30, 0.35, 0.40, 0.45],
                   help='NMS IoU threshold values to sweep')
    p.add_argument('--iou_values',   type=float, nargs='+',
                   default=[0.50],
                   help='AP matching IoU threshold values to sweep '
                        '(keep at 0.50 for comparable VOC results)')

    p.add_argument('--out_dir', default=None,
                   help='Where to save results  '
                        '(default: results/<model>/)')
    return p.parse_args()


# ─────────────────────── Phase 1: Batched Inference ──────────────────────────

def _build_inference_dataset(samples: List[Tuple[str, str]],
                              batch_size: int) -> tf.data.Dataset:
    """
    Build a tf.data pipeline that loads, resizes, and normalises images
    in parallel and returns them in batches.

    Yields batches of (images [B, 300, 300, 3], indices [B])
    where `indices` maps each image back to its position in `samples`.
    """
    paths  = [jpg for jpg, _ in samples]
    indices = list(range(len(paths)))

    path_ds = tf.data.Dataset.from_tensor_slices(
        (paths, indices))

    def _load(path, idx):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img, idx

    return (path_ds
            .map(_load,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 deterministic=True)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))


def run_batched_inference(model,
                          detector,
                          samples: List[Tuple[str, str]],
                          batch_size: int = 16
                          ) -> List[object]:
    """
    Run model inference on all images in batches.

    Returns a list of raw per-image predictions (one entry per sample),
    where each entry is whatever detector.postprocess() accepts as its
    first argument (architecture-specific).

    The list is indexed to match `samples` order.
    """
    ds          = _build_inference_dataset(samples, batch_size)
    n_images    = len(samples)
    raw_preds   = [None] * n_images   # pre-allocate in order

    n_batches   = (n_images + batch_size - 1) // batch_size
    t0          = time.time()

    print(f"[Inference] {n_images} images  "
          f"batch={batch_size}  "
          f"({n_batches} batches)")

    inf_bar = tqdm(enumerate(ds), total=n_batches,
                   desc="[Inference] Batches", unit="batch", leave=False)
    for batch_num, (images, batch_indices) in inf_bar:
        outputs = model(images, training=False)

        # outputs is a list/tuple of tensors, each [B, ...]
        # Unbatch per image
        if isinstance(outputs, (list, tuple)):
            batch_size_actual = images.shape[0]
            for local_i in range(batch_size_actual):
                global_i = int(batch_indices[local_i].numpy())
                raw_preds[global_i] = [
                    out[local_i] for out in outputs]
        else:
            # Single-output model (unlikely but safe)
            for local_i in range(images.shape[0]):
                global_i = int(batch_indices[local_i].numpy())
                raw_preds[global_i] = outputs[local_i]

        elapsed = time.time() - t0
        fps     = (batch_num + 1) * batch_size / max(elapsed, 1e-6)
        inf_bar.set_postfix(fps=f"{fps:.1f}", elapsed=f"{elapsed:.0f}s")

    print(f"[Inference] Done in {time.time() - t0:.1f}s\n")
    return raw_preds


# ─────────────────────── Phase 1b: Load Ground Truth ─────────────────────────

def load_all_ground_truths(
        samples: List[Tuple[str, str]]
) -> Dict[int, List[dict]]:
    """
    Parse all VOC XML annotations once and store in a dict.

    Returns:
        gt[cls_idx] = list of {'img_id': str, 'box': np.ndarray [4],
                                'difficult': bool}
    """
    gt: Dict[int, List[dict]] = defaultdict(list)

    for jpg_path, xml_path in tqdm(samples, desc="[GT] Parsing XML", unit="file", leave=False):
        img_id = os.path.splitext(os.path.basename(jpg_path))[0]

        tree = __import__(
            'xml.etree.ElementTree', fromlist=['parse']).parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        iw   = int(size.find('width').text)
        ih   = int(size.find('height').text)

        from config import CLASS_TO_IDX
        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            if name not in CLASS_TO_IDX:
                continue
            difficult = int(obj.find('difficult').text) \
                        if obj.find('difficult') is not None else 0
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / iw
            ymin = float(bndbox.find('ymin').text) / ih
            xmax = float(bndbox.find('xmax').text) / iw
            ymax = float(bndbox.find('ymax').text) / ih
            gt[CLASS_TO_IDX[name]].append({
                'img_id':    img_id,
                'box':       np.array([xmin, ymin, xmax, ymax], np.float32),
                'difficult': bool(difficult),
            })

    return gt


# ─────────────────────── Phase 2: NMS + Detection List ───────────────────────

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter    = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a   = (a[2] - a[0]) * (a[3] - a[1])
    area_b   = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-10)


def apply_postprocess(detector,
                      raw_preds: List,
                      samples: List[Tuple[str, str]],
                      conf: float,
                      nms_iou: float
                      ) -> Dict[int, List[Tuple]]:
    """
    Apply detector.postprocess() with specific conf and nms_iou to
    all cached raw predictions.

    Returns:
        detections[cls_idx] = [(score, img_id, box_xyxy), ...]
    """
    detections: Dict[int, List] = defaultdict(list)

    for i, (jpg_path, _) in tqdm(enumerate(samples),
                               total=len(samples),
                               desc="  Postprocess", unit="img",
                               leave=False):
        img_id   = os.path.splitext(os.path.basename(jpg_path))[0]
        pred_one = raw_preds[i]
        if pred_one is None:
            continue

        boxes, scores, labels = detector.postprocess(
            pred_one,
            conf_threshold=conf,
            nms_iou=nms_iou,
        )
        for box, score, label in zip(boxes, scores, labels):
            detections[int(label)].append(
                (float(score), img_id, box))

    return detections


# ─────────────────────── Phase 2b: Per-class AP ──────────────────────────────

def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall,    [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    pts = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[pts + 1] - mrec[pts]) * mpre[pts + 1]))


def _compute_cls_ap(cls_idx: int,
                    cls_name: str,
                    cls_dets: List[Tuple],
                    cls_gts: List[dict],
                    iou_threshold: float) -> Tuple[str, float, int]:
    """
    Compute AP for one class.  Designed to run in a thread.

    Returns (cls_name, ap, num_gt_non_difficult).
    """
    if not cls_gts:
        return cls_name, 0.0, 0

    # Group GTs by image
    gt_by_img: Dict[str, List[dict]] = defaultdict(list)
    for gt in cls_gts:
        gt_by_img[gt['img_id']].append({
            'box': gt['box'], 'difficult': gt['difficult'],
            'matched': False,
        })

    num_gt = sum(1 for g in cls_gts if not g['difficult'])
    if num_gt == 0:
        return cls_name, 0.0, 0

    cls_dets_sorted = sorted(cls_dets, key=lambda x: -x[0])
    tp = np.zeros(len(cls_dets_sorted), np.float32)
    fp = np.zeros(len(cls_dets_sorted), np.float32)

    for d, (score, img_id, det_box) in enumerate(cls_dets_sorted):
        gts = gt_by_img.get(img_id, [])
        if not gts:
            fp[d] = 1
            continue

        ious     = [_iou_xyxy(det_box, g['box']) for g in gts]
        best_idx = int(np.argmax(ious))
        best_iou = ious[best_idx]

        if best_iou >= iou_threshold:
            if not gts[best_idx]['matched']:
                if not gts[best_idx]['difficult']:
                    tp[d] = 1
                gts[best_idx]['matched'] = True
            else:
                if not gts[best_idx]['difficult']:
                    fp[d] = 1
        else:
            fp[d] = 1

    cum_tp    = np.cumsum(tp)
    cum_fp    = np.cumsum(fp)
    recall    = cum_tp / (num_gt + 1e-10)
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)
    ap        = _voc_ap(recall, precision)
    return cls_name, ap, num_gt


def compute_map_parallel(detections: Dict[int, List],
                          ground_truths: Dict[int, List],
                          iou_threshold: float,
                          n_workers: int = 8
                          ) -> Tuple[float, Dict[str, float]]:
    """
    Compute per-class AP in parallel using a thread pool.

    Returns:
        (mAP, {cls_name: ap})
    """
    futures = {}
    ap_by_cls: Dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for cls_idx in range(1, NUM_CLASSES_WITH_BG):
            cls_name = VOC_CLASSES[cls_idx - 1]
            fut = pool.submit(
                _compute_cls_ap,
                cls_idx,
                cls_name,
                list(detections.get(cls_idx, [])),
                list(ground_truths.get(cls_idx, [])),
                iou_threshold,
            )
            futures[fut] = cls_name

        for fut in as_completed(futures):
            cls_name, ap, num_gt = fut.result()
            if num_gt > 0:
                ap_by_cls[cls_name] = ap

    aps = list(ap_by_cls.values())
    mAP = float(np.mean(aps)) if aps else 0.0
    return mAP, ap_by_cls


# ─────────────────────── Grid Search ─────────────────────────────────────────

def grid_search(detector,
                raw_preds:    List,
                ground_truths: Dict[int, List],
                samples:      List[Tuple[str, str]],
                conf_values:  List[float],
                nms_values:   List[float],
                iou_values:   List[float],
                n_workers:    int = 8
                ) -> List[dict]:
    """
    Sweep all combinations of (conf, nms_iou, iou_threshold).

    For each combination:
        1. Apply postprocessing to cached predictions (fast, CPU-only)
        2. Compute mAP in parallel across classes

    Returns a list of result dicts sorted by mAP descending.
    """
    grid     = list(itertools.product(conf_values, nms_values, iou_values))
    n_points = len(grid)
    results  = []

    print(f"[Grid] {len(conf_values)} conf × "
          f"{len(nms_values)} nms_iou × "
          f"{len(iou_values)} iou  =  {n_points} combinations\n")

    t_grid = time.time()

    grid_bar = tqdm(grid, total=n_points,
                    desc="[Grid] Sweep", unit="combo", leave=False)
    for conf, nms_iou, iou_thr in grid_bar:
        t0 = time.time()

        # Apply NMS + confidence filtering to cached predictions
        detections = apply_postprocess(
            detector, raw_preds, samples, conf, nms_iou)

        # Compute mAP in parallel
        mAP, ap_by_cls = compute_map_parallel(
            detections, ground_truths, iou_thr, n_workers)

        elapsed = time.time() - t0
        grid_bar.set_postfix(
            conf=conf, nms=nms_iou, iou=iou_thr,
            mAP=f"{mAP:.4f}")
        tqdm.write(f"  conf={conf:.2f}  nms={nms_iou:.2f}  "
                   f"iou={iou_thr:.2f}  →  mAP={mAP:.4f}  ({elapsed:.1f}s)")

        results.append({
            'conf':       conf,
            'nms_iou':    nms_iou,
            'iou_thr':    iou_thr,
            'mAP':        mAP,
            'ap_by_cls':  ap_by_cls,
        })

    print(f"\n[Grid] Finished {n_points} combinations "
          f"in {time.time() - t_grid:.1f}s\n")

    return sorted(results, key=lambda r: r['mAP'], reverse=True)


# ─────────────────────── Save Results ────────────────────────────────────────

def save_results(results: List[dict],
                 out_dir: str,
                 model_type: str,
                 iou_values: List[float]) -> None:
    """
    Save two files:
        grid_search_summary.txt   — ranked table of all grid points
        grid_search_best.txt      — full per-class AP for the best setting
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, 'grid_search_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Grid search results — {model_type}\n")
        f.write(f"Ranked by mAP (descending)\n")

        if len(iou_values) > 1:
            f.write("\nWARNING: Multiple iou_values used. "
                    "Only iou=0.50 results are comparable to published VOC numbers.\n")

        f.write("\n")
        f.write(f"{'Rank':<6} {'conf':<7} {'nms_iou':<10} "
                f"{'iou_thr':<10} {'mAP':<10}\n")
        f.write("-" * 46 + "\n")

        for rank, r in enumerate(results, 1):
            f.write(f"{rank:<6} {r['conf']:<7.2f} {r['nms_iou']:<10.2f} "
                    f"{r['iou_thr']:<10.2f} {r['mAP']:<10.4f}\n")

    print(f"[Results] Summary saved  → {summary_path}")

    # ── Best point — full per-class AP ────────────────────────────────────────
    best       = results[0]
    best_path  = os.path.join(out_dir, 'grid_search_best.txt')
    with open(best_path, 'w') as f:
        f.write(f"Best configuration — {model_type}\n")
        f.write(f"  conf    = {best['conf']}\n")
        f.write(f"  nms_iou = {best['nms_iou']}\n")
        f.write(f"  iou_thr = {best['iou_thr']}\n")
        f.write(f"  mAP     = {best['mAP']:.4f}\n")
        f.write("\nPer-class AP:\n")
        f.write("-" * 35 + "\n")
        for cls_name in VOC_CLASSES:
            ap = best['ap_by_cls'].get(cls_name, 0.0)
            f.write(f"  {cls_name:<15s}  AP={ap:.4f}\n")

    print(f"[Results] Best config    → {best_path}")

    # ── Print summary to terminal ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Top 5 configurations")
    print(f"{'='*55}")
    print(f"  {'Rank':<5} {'conf':<7} {'nms_iou':<10} "
          f"{'iou_thr':<10} {'mAP'}")
    print(f"  {'-'*50}")
    for rank, r in enumerate(results[:5], 1):
        marker = '  ←  best' if rank == 1 else ''
        print(f"  {rank:<5} {r['conf']:<7.2f} {r['nms_iou']:<10.2f} "
              f"{r['iou_thr']:<10.2f} {r['mAP']:.4f}{marker}")

    print(f"\n  Best: conf={best['conf']}  "
          f"nms_iou={best['nms_iou']}  "
          f"iou_thr={best['iou_thr']}  "
          f"mAP={best['mAP']:.4f}")
    print(f"{'='*55}\n")


# ──────────────────────────── Main ───────────────────────────────────────────

def main():
    args = _parse_args()

    out_dir  = args.out_dir or os.path.join('results', args.model)
    log_path = setup_logging(log_dir=out_dir, filename='grid_search.log')
    print(f"[Grid] Log saved to {log_path}\n")

    # ── Load model ────────────────────────────────────────────────────────────
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
    print(f"[Grid] Loading weights from {ckpt_path}")
    model.load_weights(ckpt_path)

    # ── Load test image list ──────────────────────────────────────────────────
    samples = load_voc_image_list(args.data_dir, VOC_TEST_SETS)
    print(f"[Grid] Test images: {len(samples)}\n")

    # ── Phase 1a: batched inference ───────────────────────────────────────────
    print("=" * 55)
    print("  Phase 1 — Batched inference (runs once)")
    print("=" * 55)
    raw_preds = run_batched_inference(
        model, detector, samples, batch_size=args.batch)

    # ── Phase 1b: load all ground truths ─────────────────────────────────────
    print("[GT] Parsing all XML annotations …")
    t0 = time.time()
    ground_truths = load_all_ground_truths(samples)
    print(f"[GT] Done in {time.time() - t0:.1f}s\n")

    # ── Phase 2: grid search ──────────────────────────────────────────────────
    print("=" * 55)
    print("  Phase 2 — Grid search (CPU, parallel AP)")
    print("=" * 55)

    # Warn if iou_values != [0.50]
    if args.iou_values != [0.50]:
        print(f"\n  WARNING: iou_values={args.iou_values}")
        print("  Only iou=0.50 is comparable to published VOC results.")
        print("  Use other values for research/debugging only.\n")

    results = grid_search(
        detector      = detector,
        raw_preds     = raw_preds,
        ground_truths = ground_truths,
        samples       = samples,
        conf_values   = args.conf_values,
        nms_values    = args.nms_values,
        iou_values    = args.iou_values,
        n_workers     = args.workers,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(results, out_dir, args.model, args.iou_values)

    # ── Recommended settings for training and Flutter ─────────────────────────
    best = results[0]
    print("Recommended settings based on grid search:")
    print(f"  evaluate_voc.py  →  "
          f"--conf {best['conf']} --nms_iou {best['nms_iou']} --iou 0.50")
    print(f"  export_tflite.py →  "
          f"--conf {best['conf']} --nms_iou {best['nms_iou']}")
    print()


if __name__ == '__main__':
    main()