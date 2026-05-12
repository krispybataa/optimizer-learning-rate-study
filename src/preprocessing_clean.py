"""
preprocessing.py

Prepares the STRAMPN ovarian cancer dataset for training:
  - Inventories raw images and verifies expected counts
  - Performs a stratified train / val / test split (52.5% / 22.5% / 25%)
  - Copies files into dataset/processed/{split}/{class}/ directories
  - Prints a verification report and checks for cross-split leakage

Run directly:  python src/preprocessing.py
"""

import os
import shutil
import collections
from pathlib import Path

from sklearn.model_selection import train_test_split


# ── Directory layout ──────────────────────────────────────────────────────────

# Resolve paths relative to this file's location so the script works from
# C:\dev-work\optimizer-learning-rate-study (Windows) or the WSL2 mount path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_RAW_BASE = _PROJECT_ROOT / "dataset" / "raw" / "STRAMPN Dataset" / "Dataset"
_RAW_DIRS = {
    "cancer":    _RAW_BASE / "Ovarian_Cancer",
    "no_cancer": _RAW_BASE / "Ovarian_Non_Cancer",
}

_PROCESSED_BASE = _PROJECT_ROOT / "dataset" / "processed"

_EXPECTED_COUNTS = {"cancer": 481, "no_cancer": 506}
_EXPECTED_TOTAL  = 987


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_files(directory: Path) -> list[Path]:
    """Return a sorted list of all files (non-recursive) in directory."""
    return sorted(p for p in directory.iterdir() if p.is_file())


def _copy_file(src: Path, dst_dir: Path) -> str:
    """Copy src to dst_dir/src.name. Returns 'copied' or 'skipped'."""
    dst = dst_dir / src.name
    if dst.exists():
        return "skipped"
    shutil.copy2(src, dst)
    return "copied"


# ── Main function ─────────────────────────────────────────────────────────────

def run_preprocessing(random_seed: int = 42) -> None:
    """
    Full preprocessing pipeline: inventory -> split -> copy -> verify.

    Args:
        random_seed: Controls the stratified split for reproducibility.
                     Must be identical to the seed used in all 84 training runs.
    """

    # ==========================================================================
    # Step 1 - Inventory
    # ==========================================================================
    print("=" * 60)
    print("STEP 1 - RAW DATASET INVENTORY")
    print("=" * 60)

    all_pairs: list[tuple[Path, str]] = []  # (filepath, label)

    for label, raw_dir in _RAW_DIRS.items():
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"Raw directory not found: {raw_dir}\n"
                "Download the STRAMPN dataset and place it at:\n"
                f"  {_RAW_BASE}"
            )

        files = _all_files(raw_dir)
        non_jpg = [f for f in files if f.suffix.lower() != ".jpg"]

        print(f"\n  [{label}]  path: {raw_dir}")
        print(f"    Total files : {len(files)}")

        if non_jpg:
            print(f"    [WARNING] Non-JPG files found ({len(non_jpg)}):")
            for f in non_jpg:
                print(f"        {f.name}")
        else:
            print(f"    All files   : .jpg [ok]")

        jpg_files = [f for f in files if f.suffix.lower() == ".jpg"]
        expected  = _EXPECTED_COUNTS[label]
        match_sym = "[ok]" if len(jpg_files) == expected else "[x] (expected {expected})"
        print(f"    JPG count   : {len(jpg_files)}  {match_sym}")

        all_pairs.extend((f, label) for f in jpg_files)

    total = len(all_pairs)
    total_sym = "[ok]" if total == _EXPECTED_TOTAL else f"[x] (expected {_EXPECTED_TOTAL})"
    print(f"\n  Total images across both classes: {total}  {total_sym}")

    # ==========================================================================
    # Step 2 - Stratified Split
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 2 - STRATIFIED SPLIT")
    print("=" * 60)

    filepaths = [p for p, _ in all_pairs]
    labels    = [l for _, l in all_pairs]

    # First split: 75% train+val, 25% test
    paths_trainval, paths_test, labels_trainval, labels_test = train_test_split(
        filepaths, labels,
        test_size=0.25,
        random_state=random_seed,
        stratify=labels,
    )

    # Second split: 70% train, 30% val  (of the 75% slice -> ~52.5% / 22.5% overall)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_trainval, labels_trainval,
        test_size=0.30,
        random_state=random_seed,
        stratify=labels_trainval,
    )

    splits = {
        "train": (paths_train, labels_train),
        "val":   (paths_val,   labels_val),
        "test":  (paths_test,  labels_test),
    }

    print(f"\n  Seed: {random_seed}")
    print(f"  {'Split':<8}  {'cancer':>8}  {'no_cancer':>10}  {'total':>7}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}")
    for split_name, (paths, lbls) in splits.items():
        c  = lbls.count("cancer")
        nc = lbls.count("no_cancer")
        print(f"  {split_name:<8}  {c:>8}  {nc:>10}  {len(lbls):>7}")
    print(f"  {'total':<8}  {labels.count('cancer'):>8}  {labels.count('no_cancer'):>10}  {total:>7}")

    # ==========================================================================
    # Step 3 - Copy Files
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 3 - COPYING FILES TO dataset/processed/")
    print("=" * 60)

    for split_name, (paths, lbls) in splits.items():
        copied_total  = 0
        skipped_total = 0

        for src, label in zip(paths, lbls):
            dst_dir = _PROCESSED_BASE / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            result = _copy_file(src, dst_dir)
            if result == "copied":
                copied_total  += 1
            else:
                skipped_total += 1

        print(
            f"  {split_name:<6}  copied: {copied_total:>4}   "
            f"skipped (already exist): {skipped_total:>4}"
        )

    # ==========================================================================
    # Step 4 - Verification Report
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 4 - VERIFICATION REPORT")
    print("=" * 60)

    class_names = ["cancer", "no_cancer"]
    grand_total = 0

    print(f"\n  {'Split':<8}  {'Class':<12}  {'Count':>6}  {'% of total':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*6}  {'-'*10}")

    for split_name in ["train", "val", "test"]:
        for cls in class_names:
            cls_dir = _PROCESSED_BASE / split_name / cls
            n = len(list(cls_dir.glob("*.jpg"))) if cls_dir.exists() else 0
            pct = n / total * 100
            grand_total += n
            print(f"  {split_name:<8}  {cls:<12}  {n:>6}  {pct:>9.1f}%")

    print(f"\n  Grand total in processed/: {grand_total}  "
          f"({'[ok] matches raw' if grand_total == total else '[x] MISMATCH with raw'})")

    # Cross-split leakage check - compare stem names across splits
    print("\n  Checking for cross-split filename leakage...")
    split_stems: dict[str, set[str]] = {}
    for split_name in ["train", "val", "test"]:
        stems: set[str] = set()
        for cls in class_names:
            cls_dir = _PROCESSED_BASE / split_name / cls
            if cls_dir.exists():
                stems.update(p.stem for p in cls_dir.glob("*.jpg"))
        split_stems[split_name] = stems

    violations: list[str] = []
    split_list = list(split_stems.items())
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            a_name, a_stems = split_list[i]
            b_name, b_stems = split_list[j]
            overlap = a_stems & b_stems
            if overlap:
                violations.append(
                    f"    [x] {a_name} intersect {b_name}: {len(overlap)} shared filename(s): "
                    + ", ".join(sorted(overlap)[:5])
                    + ("..." if len(overlap) > 5 else "")
                )

    print()
    if violations:
        print("  [WARNING] INTEGRITY VIOLATIONS FOUND:")
        for v in violations:
            print(v)
    else:
        print("  PREPROCESSING COMPLETE - No integrity violations found")

    print("\n" + "=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing()
