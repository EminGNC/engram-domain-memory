"""Kosu loglarindan A/B/C/D/E karsilastirma tablosu uretir.

Kullanim:
    python eval/report.py runs/C_python/log.txt runs/B_lora/log.txt
    python eval/report.py --all   # runs/ altindaki tum loglari tarar
"""

import argparse
import math
import re
import sys
from pathlib import Path


def parse_log(path: Path):
    """log.txt -> [(step, val_loss)]"""
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"step\s+(\d+)\s+\|\s+VAL LOSS ([\d.]+)", line)
        if m:
            vals.append((int(m.group(1)), float(m.group(2))))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    paths = []
    if args.all:
        paths = sorted(Path("runs").glob("*/log.txt"))
    else:
        paths = [Path(p) for p in args.logs]

    if not paths:
        print("Log bulunamadi.")
        return

    print(f"{'kosu':<28} {'step':>6} {'val loss':>9} {'ppl':>8}")
    print("-" * 55)
    for p in paths:
        vals = parse_log(p)
        name = p.parent.name
        if not vals:
            print(f"{name:<28} {'—':>6} {'—':>9} {'—':>8}")
            continue
        # Ilk ve son olcumu bas; aradakileri seyrek goster
        first = vals[0]
        print(f"{name:<28} {first[0]:>6} {first[1]:>9.4f} {math.exp(first[1]):>8.2f}  (ilk)")
        if len(vals) > 1:
            last = vals[-1]
            delta = first[1] - last[1]
            print(f"{'':<28} {last[0]:>6} {last[1]:>9.4f} {math.exp(last[1]):>8.2f}  (son, Δ={delta:+.4f})")


if __name__ == "__main__":
    main()
