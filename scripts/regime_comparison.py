"""Equal-budget comparison of the three training regimes.

The published fine-tuning run (training_results/finetuning) trained for 172
epochs on the augmented split. Re-running the other two regimes to the same
length costs ~20 h each, so this script trains **all three** regimes under an
identical, smaller budget (same split, same epochs, same image fraction) so
the comparison is fair even though the absolute numbers are lower.

Usage:
    python scripts/regime_comparison.py --epochs 25 --fraction 0.25
Outputs:
    training_results/regime_comparison/<regime>/   (Ultralytics run dirs)
    training_results/regime_comparison/summary.csv (best mAP per regime)
"""
import argparse
import csv
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
REGIMES = {
    "sgd_from_scratch":       {"cfg": "yamls/sgd_from_scratch.yaml",       "model": "yolo11n.yaml"},
    "adamw_transfer_learning": {"cfg": "yamls/adamw_transfer_learning.yaml", "model": "yolo11n.pt"},
    "adamw_finetuning":        {"cfg": "yamls/adamw_finetuning.yaml",        "model": "yolo11n.pt"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--fraction", type=float, default=0.25, help="fraction of the training split to use")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data", default="dataset/yolo_base_dataset.yaml", help="non-augmented split")
    ap.add_argument("--regimes", nargs="*", default=list(REGIMES))
    args = ap.parse_args()

    out_dir = ROOT / "training_results" / "regime_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.regimes:
        r = REGIMES[name]
        t0 = time.time()
        model = YOLO(r["model"])
        model.train(
            data=str(ROOT / args.data), cfg=str(ROOT / r["cfg"]),
            epochs=args.epochs, patience=args.epochs,        # no early stop: equal budget
            fraction=args.fraction, imgsz=args.imgsz, batch=args.batch,
            project=str(out_dir), name=name, exist_ok=True, seed=0, deterministic=True,
            plots=True, verbose=False,
        )
        results = list(csv.DictReader(open(out_dir / name / "results.csv")))
        best = max(results, key=lambda x: float(x["metrics/mAP50-95(B)"]))
        rows.append({
            "regime": name, "epochs": len(results), "hours": round((time.time() - t0) / 3600, 2),
            "best_epoch": best["epoch"].strip(),
            "precision": round(float(best["metrics/precision(B)"]), 4),
            "recall": round(float(best["metrics/recall(B)"]), 4),
            "mAP50": round(float(best["metrics/mAP50(B)"]), 4),
            "mAP50-95": round(float(best["metrics/mAP50-95(B)"]), 4),
        })
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(rows[-1])


if __name__ == "__main__":
    main()
