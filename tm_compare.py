#!/usr/bin/env python
"""TM-score every sample in a condition against the dense run at matched seed.

Usage:
    python tm_compare.py <cond_dir> <dense_dir> [--tm /path/to/TMscore]

Prints per-length mean/median TM and the fraction of samples above 0.5
(same fold as dense).
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np


def find_tmscore(explicit=None):
    if explicit:
        return explicit
    for c in [shutil.which("TMscore"),
              os.path.expandvars("$PROJECT/genie3/packages/TMscore/TMscore"),
              os.path.expandvars("$PROJECT/genie3/packages/TMscore")]:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    sys.exit("TMscore binary not found; pass --tm /path/to/TMscore")


TM_RE = re.compile(r"TM-score\s*=\s*([0-9.]+)")


def tm(binary, a, b):
    out = subprocess.run([binary, a, b], capture_output=True, text=True).stdout
    m = TM_RE.search(out)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond_dir")
    ap.add_argument("dense_dir")
    ap.add_argument("--tm", default=None)
    args = ap.parse_args()

    binary = find_tmscore(args.tm)
    by_len = defaultdict(list)
    missing = 0

    for pa in sorted(glob.glob(os.path.join(args.cond_dir, "**", "sample.pdb"),
                               recursive=True)):
        rel = os.path.relpath(pa, args.cond_dir)
        pb = os.path.join(args.dense_dir, rel)
        if not os.path.exists(pb):
            missing += 1
            continue
        score = tm(binary, pa, pb)
        if score is None:
            missing += 1
            continue
        length = rel.split(os.sep)[0]          # e.g. "length_300"
        by_len[length].append(score)

    if not by_len:
        sys.exit("no paired samples found")

    print(f"TMscore: {binary}")
    print(f"{'length':<12}{'n':>4}{'mean':>8}{'median':>8}{'min':>8}{'>0.5':>8}")
    allv = []
    for length in sorted(by_len, key=lambda s: int(s.split("_")[1])):
        v = np.array(by_len[length])
        allv.append(v)
        print(f"{length:<12}{len(v):>4}{v.mean():>8.3f}{np.median(v):>8.3f}"
              f"{v.min():>8.3f}{(v > 0.5).mean():>8.2f}")
    v = np.concatenate(allv)
    print(f"{'ALL':<12}{len(v):>4}{v.mean():>8.3f}{np.median(v):>8.3f}"
          f"{v.min():>8.3f}{(v > 0.5).mean():>8.2f}")
    if missing:
        print(f"({missing} unpaired or unparsed)")


if __name__ == "__main__":
    main()
