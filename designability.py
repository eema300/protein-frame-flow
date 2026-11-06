#!/usr/bin/env python
"""Designability for a directory of generated backbones.

ProteinMPNN inverse-folds each backbone, ESMFold refolds each sequence,
scRMSD is the best CA RMSD over sequences. Designable = scRMSD < 2.0 A.

Run in the genie3 env (needs esm + openfold), not the fm env.

Usage:
    python designability.py <run_dir> --out results.csv \
        [--mpnn $PROJECT/genie3/packages/ProteinMPNN] [--n-seq 8]
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch


def ca_coords(path):
    xs = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
          for l in open(path)
          if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return np.array(xs)


def kabsch_rmsd(P, Q):
    """CA RMSD after optimal superposition. P, Q are [N, 3]."""
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    P = P @ (V @ D @ Wt)
    return float(np.sqrt(((P - Q) ** 2).sum(-1).mean()))


def run_mpnn(mpnn_dir, pdb, out_dir, n_seq, temp, seed):
    script = os.path.join(mpnn_dir, "protein_mpnn_run.py")
    if not os.path.isfile(script):
        sys.exit(f"protein_mpnn_run.py not found under {mpnn_dir}")
    cmd = [sys.executable, script,
           "--pdb_path", pdb,
           "--pdb_path_chains", "A",
           "--out_folder", out_dir,
           "--num_seq_per_target", str(n_seq),
           "--sampling_temp", str(temp),
           "--seed", str(seed),
           "--batch_size", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], file=sys.stderr)
        print(r.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"ProteinMPNN failed on {pdb}")
    fa = glob.glob(os.path.join(out_dir, "seqs", "*.fa"))
    if not fa:
        raise RuntimeError(f"no ProteinMPNN output for {pdb}")
    seqs, keep = [], False
    for line in open(fa[0]):
        line = line.strip()
        if line.startswith(">"):
            keep = "T=" in line          # skip the native/input record
            continue
        if keep and line:
            seqs.append(line)
    return seqs[:n_seq]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="designability.csv")
    ap.add_argument("--mpnn",
                    default=os.path.expandvars("$PROJECT/genie3/packages/ProteinMPNN"))
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--cutoff", type=float, default=2.0)
    args = ap.parse_args()

    pdbs = sorted(glob.glob(os.path.join(args.run_dir, "**", "sample.pdb"),
                            recursive=True))
    if not pdbs:
        sys.exit(f"no sample.pdb under {args.run_dir}")
    print(f"{len(pdbs)} backbones", flush=True)

    import esm
    model = esm.pretrained.esmfold_v1().eval().cuda()

    rows = []
    for i, pdb in enumerate(pdbs):
        rel = os.path.relpath(pdb, args.run_dir)
        ref = ca_coords(pdb)
        with tempfile.TemporaryDirectory() as td:
            seqs = run_mpnn(args.mpnn, pdb, td, args.n_seq, args.temp, args.seed)
        best, best_seq = np.inf, None
        for s in seqs:
            with torch.no_grad():
                out = model.infer_pdb(s)
            with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
                f.write(out)
                fold_path = f.name
            pred = ca_coords(fold_path)
            os.unlink(fold_path)
            if pred.shape != ref.shape:
                continue
            r = kabsch_rmsd(pred, ref)
            if r < best:
                best, best_seq = r, s
        rows.append({"sample": rel,
                     "length": rel.split(os.sep)[0],
                     "scrmsd": best,
                     "designable": best < args.cutoff,
                     "best_seq": best_seq})
        print(f"[{i+1}/{len(pdbs)}] {rel}  scRMSD={best:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print()
    print(df.groupby("length").agg(n=("scrmsd", "size"),
                                   mean_scrmsd=("scrmsd", "mean"),
                                   designable=("designable", "mean")).round(3))
    print(f"\noverall designable: {df.designable.mean():.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
