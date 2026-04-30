"""
export_labelmap_voc.py — Generate Flutter-compatible labelmap for Pascal VOC.

DROP-IN REPLACEMENT for export_labelmap.py when using Pascal VOC.

The Flutter DetectionService reads the labelmap as:
    _labels = raw.split('\\n').where((l) => l.trim().isNotEmpty).toList();
    final label = classId + 1 < _labels.length
                  ? _labels[classId + 1] : 'unknown';

So index 0 is the background dummy and indices 1–20 are the 20 VOC classes.

Output files written to outputs/<model>/ :
    labelmap_voc_<model>.txt   — named copy for reference
    labelmap_voc.txt           — Flutter drop-in name (matches pubspec.yaml)

Usage:
    python export_labelmap_voc.py                        # default model from config
    python export_labelmap_voc.py --model retinanet
    python export_labelmap_voc.py --out_dir my_assets/
"""

import os
import argparse

from config import VOC_CLASSES, MODEL_TYPE

AVAILABLE_MODELS = [
    'mobilenet_ssd', 'mobilenetv2_ssd', 'vgg_ssd', 'resnet_ssd',
    'retinanet', 'yolov3', 'fcos', 'centernet',
]


# ──────────────────────────── CLI ────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Generate Flutter-compatible VOC labelmap files.')
    p.add_argument('--model',   default=MODEL_TYPE,
                   choices=AVAILABLE_MODELS,
                   help='Used only to determine the output subdirectory.')
    p.add_argument('--out_dir', default=None,
                   help='Output directory (default: outputs/<model>/)')
    return p.parse_args()


# ──────────────────────────── Writer ─────────────────────────────────────────

def write_voc_labelmap(out_dir: str, model_name: str) -> None:
    """
    Write two identical labelmap files into out_dir.

    Format:
        Line 0  : ???             ← background dummy  (classId + 1 offset)
        Line 1  : aeroplane       ← VOC class 0  (classId=0  → _labels[1])
        Line 2  : bicycle
        ...
        Line 20 : tvmonitor       ← VOC class 19 (classId=19 → _labels[20])
    """
    os.makedirs(out_dir, exist_ok=True)

    lines   = ['???'] + list(VOC_CLASSES)
    content = '\n'.join(lines) + '\n'

    named_path   = os.path.join(out_dir, f'labelmap_voc_{model_name}.txt')
    flutter_path = os.path.join(out_dir, 'labelmap_voc.txt')

    for path in (named_path, flutter_path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[Labelmap] Written → {path}  ({len(lines)} entries)")

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n[Labelmap] Full label list:")
    for i, lbl in enumerate(lines):
        role = '← background dummy  (never returned by model)' \
               if i == 0 else f'← classId {i - 1}'
        print(f"  [{i:2d}] {lbl:<15s} {role}")

    print(f"\n[Labelmap] Total entries  : {len(lines)}"
          f"  (1 dummy + {len(lines) - 1} classes)")
    print(f"[Labelmap] Dart lookup    : "
          f"_labels[classId + 1]  e.g. classId=14 → '{lines[15]}'")


# ──────────────────────────── Main ───────────────────────────────────────────

def main():
    args    = _parse_args()
    out_dir = args.out_dir or os.path.join('outputs', args.model)

    print(f"\n[Labelmap] Model   : {args.model}")
    print(f"[Labelmap] Dataset : Pascal VOC  ({len(VOC_CLASSES)} classes)")
    print(f"[Labelmap] Out dir : {out_dir}\n")

    write_voc_labelmap(out_dir, args.model)

    print(f"\n[Labelmap] Done.  Copy to your Flutter assets folder:")
    print(f"  cp {os.path.join(out_dir, 'labelmap_voc.txt')} "
          f"<flutter_project>/assets/models/labelmap_voc.txt\n")


if __name__ == '__main__':
    main()
    
    