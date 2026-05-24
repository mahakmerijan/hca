"""
COCO 2017 Dataset Downloader
==============================
Downloads the COCO 2017 dataset via FiftyOne Zoo and exports it to the
local 'datasets/' folder in standard COCO directory layout.

Stages
------
1. Load 50 random validation samples (detections only)
2. Load 25 validation samples with cat/dog segmentations
3. Load the full validation split with both detections + segmentations
4. Export the final dataset to datasets/coco-2017-validation/
"""

import os
import fiftyone as fo
import fiftyone.zoo as foz

# ── Output directory ──────────────────────────────────────────────────────────
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
EXPORT_DIR = os.path.join(DATASETS_DIR, "coco-2017-validation")
os.makedirs(DATASETS_DIR, exist_ok=True)

print("=" * 60)
print("  COCO 2017 Dataset Downloader")
print("=" * 60)

# ── Stage 1: 50 random validation samples (detections) ───────────────────────
print("\n[Stage 1/3] Loading 50 random validation samples (detections)…")
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=50,
    shuffle=True,
)
print(f"  ✓ Loaded {len(dataset)} samples")

# ── Stage 2: 25 samples with cat/dog segmentations ───────────────────────────
print("\n[Stage 2/3] Loading 25 cat/dog samples with segmentations…")
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["segmentations"],
    classes=["cat", "dog"],
    max_samples=25,
)
print(f"  ✓ Loaded {len(dataset)} samples")

# ── Stage 3: Full validation split — detections + segmentations ──────────────
print("\n[Stage 3/3] Loading full validation split (detections + segmentations)…")
print("  (This is the large download — may take a while on first run)")
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections", "segmentations"],
)
print(f"  ✓ Loaded {len(dataset)} samples")

# ── Export to datasets/ folder ────────────────────────────────────────────────
print(f"\n[Export] Saving dataset to: {EXPORT_DIR}")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Inspect actual label fields present in the loaded dataset
print(f"  Dataset fields: {dataset.get_field_schema().keys()}")

# The COCO zoo dataset stores labels under 'detections' and 'segmentations'
# Export detections as the primary label field
dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.COCODetectionDataset,
    label_field="detections",
    overwrite=True,
)

print(f"\n✅ Dataset saved to: {EXPORT_DIR}")
print("\nDirectory layout:")
for root, dirs, files in os.walk(EXPORT_DIR):
    depth = root.replace(EXPORT_DIR, "").count(os.sep)
    indent = "  " * depth
    print(f"{indent}{os.path.basename(root)}/")
    if depth < 2:
        for f in files[:5]:
            print(f"{indent}  {f}")
        if len(files) > 5:
            print(f"{indent}  … ({len(files)} files total)")

print("\nDone!")
