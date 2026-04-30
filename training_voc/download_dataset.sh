#!/usr/bin/env bash
# =============================================================================
# download_voc.sh — Download and verify Pascal VOC 2007 + 2012
#
# Usage:
#   bash download_voc.sh              # downloads to data/voc/ (default)
#   bash download_voc.sh /my/path     # downloads to a custom path
#
# What this script does:
#   1. Creates the target directory
#   2. Downloads VOC2007 trainval + test  (~460 MB total)
#   3. Downloads VOC2012 trainval         (~2.0 GB)
#   4. Extracts all archives into VOCdevkit/
#   5. Verifies the expected file counts so you know it worked
#   6. Prints the exact directory layout the training code expects
#
# Requirements:  wget  tar  (both available on Linux / macOS / WSL)
# Disk space   : ~3.5 GB for downloads + ~5.5 GB extracted  (~9 GB total)
#                Delete the .tar files after extraction to reclaim ~3.5 GB.
# =============================================================================

set -euo pipefail   # exit on error, unset variable, or pipe failure

# ─────────────────────────── Configuration ───────────────────────────────────

DATA_ROOT="${1:-../data/voc}"          # override by passing a path as $1

VOC_BASE="http://host.robots.ox.ac.uk/pascal/VOC"

# Archives to download: (url, filename, expected_extracted_folder)
declare -a URLS=(
    "${VOC_BASE}/voc2007/VOCtrainval_06-Nov-2007.tar"
    "${VOC_BASE}/voc2007/VOCtest_06-Nov-2007.tar"
    "${VOC_BASE}/voc2012/VOCtrainval_11-May-2012.tar"
)

declare -a FILENAMES=(
    "VOCtrainval_06-Nov-2007.tar"
    "VOCtest_06-Nov-2007.tar"
    "VOCtrainval_11-May-2012.tar"
)

# Expected image counts after extraction (for verification)
declare -A EXPECTED_IMAGES=(
    ["VOC2007"]=9963    # 5011 trainval + 4952 test
    ["VOC2012"]=17125
)

# Expected annotation counts (same as image counts — one XML per image)
declare -A EXPECTED_ANNOTS=(
    ["VOC2007"]=9963
    ["VOC2012"]=17125
)

# ─────────────────────────── Helpers ─────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'   # no colour

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

check_cmd() {
    command -v "$1" &>/dev/null || \
        error "'$1' is not installed. Install it with:  sudo apt-get install $1"
}

human_size() {
    # Print a file size in human-readable form
    du -sh "$1" 2>/dev/null | cut -f1
}

# ─────────────────────────── Preflight checks ────────────────────────────────

echo ""
echo "============================================================"
echo "  Pascal VOC 2007 + 2012 downloader"
echo "  Target directory: ${DATA_ROOT}"
echo "============================================================"
echo ""

check_cmd wget
check_cmd tar

# ─────────────────────────── Create directories ──────────────────────────────

info "Creating directory: ${DATA_ROOT}"
mkdir -p "${DATA_ROOT}"

DEVKIT="${DATA_ROOT}/VOCdevkit"

# ─────────────────────────── Download ────────────────────────────────────────

info "Starting downloads (total ~2.5 GB) …"
echo ""

for i in "${!URLS[@]}"; do
    URL="${URLS[$i]}"
    FILENAME="${FILENAMES[$i]}"
    DEST="${DATA_ROOT}/${FILENAME}"

    if [[ -f "${DEST}" ]]; then
        warn "${FILENAME} already exists ($(human_size "${DEST}")), skipping download."
    else
        info "Downloading ${FILENAME} …"
        wget \
            --progress=bar:force \
            --retry-connrefused \
            --waitretry=5 \
            --tries=5 \
            --timeout=60 \
            -O "${DEST}" \
            "${URL}" \
            || error "Failed to download ${URL}"
        success "Downloaded ${FILENAME}  ($(human_size "${DEST}"))"
    fi
done

echo ""

# ─────────────────────────── Extract ─────────────────────────────────────────

info "Extracting archives into ${DATA_ROOT}/ …"
echo ""

for FILENAME in "${FILENAMES[@]}"; do
    DEST="${DATA_ROOT}/${FILENAME}"

    # Determine which VOC year this archive belongs to
    if [[ "${FILENAME}" == *"2007"* ]]; then
        YEAR_DIR="${DEVKIT}/VOC2007"
    else
        YEAR_DIR="${DEVKIT}/VOC2012"
    fi

    # # Skip if the target directory already has images
    # if [[ -d "${YEAR_DIR}/JPEGImages" ]] && \
    #    [[ "$(ls -1 "${YEAR_DIR}/JPEGImages" | wc -l)" -gt 100 ]]; then
    #     warn "${FILENAME} already extracted — skipping."
    # else
    #     info "Extracting ${FILENAME} …"
    #     tar -xf "${DEST}" -C "${DATA_ROOT}/"
    #     success "Extracted ${FILENAME}"
    # fi

    info "Extracting ${FILENAME} …"
    tar -xf "${DEST}" -C "${DATA_ROOT}/"
    success "Extracted ${FILENAME}"
done

echo ""

# ─────────────────────────── Verify ──────────────────────────────────────────

info "Verifying extracted files …"
echo ""

ALL_OK=true

for YEAR in "VOC2007" "VOC2012"; do
    YEAR_DIR="${DEVKIT}/${YEAR}"

    # Check required subdirectories exist
    for SUBDIR in "Annotations" "JPEGImages" "ImageSets/Main"; do
        if [[ ! -d "${YEAR_DIR}/${SUBDIR}" ]]; then
            error "${YEAR}/${SUBDIR} directory not found — extraction may have failed."
        fi
    done

    # Count images and annotations
    IMG_COUNT=$(ls -1 "${YEAR_DIR}/JPEGImages/"*.jpg 2>/dev/null | wc -l)
    ANN_COUNT=$(ls -1 "${YEAR_DIR}/Annotations/"*.xml 2>/dev/null | wc -l)

    EXPECTED_IMG="${EXPECTED_IMAGES[$YEAR]}"
    EXPECTED_ANN="${EXPECTED_ANNOTS[$YEAR]}"

    if [[ "${IMG_COUNT}" -ge "${EXPECTED_IMG}" ]]; then
        success "${YEAR}  images     : ${IMG_COUNT}  (expected ≥ ${EXPECTED_IMG})"
    else
        warn    "${YEAR}  images     : ${IMG_COUNT}  (expected ≥ ${EXPECTED_IMG}) — may be incomplete"
        ALL_OK=false
    fi

    if [[ "${ANN_COUNT}" -ge "${EXPECTED_ANN}" ]]; then
        success "${YEAR}  annotations: ${ANN_COUNT}  (expected ≥ ${EXPECTED_ANN})"
    else
        warn    "${YEAR}  annotations: ${ANN_COUNT}  (expected ≥ ${EXPECTED_ANN}) — may be incomplete"
        ALL_OK=false
    fi

    # Check the split files that the training code needs
    for SPLIT in "trainval" "train" "val"; do
        SPLIT_FILE="${YEAR_DIR}/ImageSets/Main/${SPLIT}.txt"
        if [[ -f "${SPLIT_FILE}" ]]; then
            COUNT=$(wc -l < "${SPLIT_FILE}")
            success "${YEAR}  ImageSets/Main/${SPLIT}.txt  (${COUNT} entries)"
        else
            warn    "${YEAR}  ImageSets/Main/${SPLIT}.txt  NOT FOUND"
            ALL_OK=false
        fi
    done

    # VOC2007 also needs test.txt
    if [[ "${YEAR}" == "VOC2007" ]]; then
        TEST_FILE="${YEAR_DIR}/ImageSets/Main/test.txt"
        if [[ -f "${TEST_FILE}" ]]; then
            COUNT=$(wc -l < "${TEST_FILE}")
            success "VOC2007  ImageSets/Main/test.txt  (${COUNT} entries)"
        else
            warn    "VOC2007  ImageSets/Main/test.txt  NOT FOUND"
            ALL_OK=false
        fi
    fi

    echo ""
done

# ─────────────────────────── Print layout ────────────────────────────────────

echo "============================================================"
echo "  Directory layout (matches dataset_voc.py expectations)"
echo "============================================================"
echo ""
echo "  ${DATA_ROOT}/"
echo "    VOCdevkit/"
echo "      VOC2007/"
echo "        Annotations/         $(ls -1 "${DEVKIT}/VOC2007/Annotations/" 2>/dev/null | wc -l) XML files"
echo "        ImageSets/Main/      $(ls -1 "${DEVKIT}/VOC2007/ImageSets/Main/"*.txt 2>/dev/null | wc -l) split .txt files"
echo "        JPEGImages/          $(ls -1 "${DEVKIT}/VOC2007/JPEGImages/" 2>/dev/null | wc -l) JPEG images"
echo "      VOC2012/"
echo "        Annotations/         $(ls -1 "${DEVKIT}/VOC2012/Annotations/" 2>/dev/null | wc -l) XML files"
echo "        ImageSets/Main/      $(ls -1 "${DEVKIT}/VOC2012/ImageSets/Main/"*.txt 2>/dev/null | wc -l) split .txt files"
echo "        JPEGImages/          $(ls -1 "${DEVKIT}/VOC2012/JPEGImages/" 2>/dev/null | wc -l) JPEG images"
echo ""

# ─────────────────────────── Disk usage ──────────────────────────────────────

TOTAL_SIZE=$(du -sh "${DATA_ROOT}" 2>/dev/null | cut -f1)
TAR_SIZE=$(du -sh "${DATA_ROOT}"/*.tar 2>/dev/null | awk '{sum += $1} END {print sum}' || echo "?")

info "Total disk usage: ${TOTAL_SIZE}"
info "Archives (.tar):  $(du -sh "${DATA_ROOT}"/*.tar 2>/dev/null | awk '{print $2": "$1}' | tr '\n' '  ')"
echo ""
info "Tip: delete the .tar files to free ~2.5 GB once you have verified the data:"
echo ""
echo "  rm ${DATA_ROOT}/*.tar"
echo ""

# ─────────────────────────── Next steps ──────────────────────────────────────

echo "============================================================"
echo "  Next steps"
echo "============================================================"
echo ""
if $ALL_OK; then
    success "All files verified.  You are ready to train."
else
    warn "Some files were not found — check warnings above."
fi
echo ""
echo "  1. Activate the VOC config and dataset:"
echo ""
echo "       cp config.py  config_coco.py   # backup"
echo "       cp dataset.py dataset_coco.py  # backup"
echo "       cp config_voc.py  config.py"
echo "       cp dataset_voc.py dataset.py"
echo ""
echo "  2. Train:"
echo ""
echo "       python train.py --model mobilenet_ssd"
echo ""
echo "  3. Evaluate:"
echo ""
echo "       python evaluate_voc.py --model mobilenet_ssd"
echo ""
echo "  4. Export to TFLite:"
echo ""
echo "       python export_labelmap_voc.py --model mobilenet_ssd"
echo "       python export_tflite.py       --model mobilenet_ssd"
echo ""
echo "  See SWAP_GUIDE.md for full documentation."
echo ""