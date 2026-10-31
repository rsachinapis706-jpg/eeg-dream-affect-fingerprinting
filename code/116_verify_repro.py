"""
Check that 116_pcd_leaky.py reproduces its earlier 20-labelling run exactly
when labellings are batched: recompute labellings 0 and 1 together and
compare every rule's coverage and retained balanced accuracy with
out/stale_pcd_leaky_conditions_r20.csv. If they agree, the 20 finished labellings
can seed the resumable run instead of being recomputed.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "leaky", os.path.join(HERE, "116_pcd_leaky.py"))
lk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lk)

X, y, subj, unit = lk.load_deed()
parts = lk.partitions(subj)
batch = [(f"null {i}", lk.null_labels(y, unit, subj, i), lk.SEED + i)
         for i in (0, 1)]
res = lk.run(X, subj, unit, parts, batch)
new = pd.DataFrame([r for n, _, _ in batch for r in res[n][1]]).round(6)
old = pd.read_csv(os.path.join(lk.OUT, "stale_pcd_leaky_conditions_r20.csv"))
old = old[old.labelling.isin(["null 0", "null 1"])]
m = new.merge(old, on=["partition", "labelling", "rule"], suffixes=("_new", "_old"))
assert len(m) == len(new) == len(old), (len(m), len(new), len(old))
d = max(np.nanmax(np.abs(m.coverage_new - m.coverage_old)),
        np.nanmax(np.abs(m.bal_retained_new - m.bal_retained_old)))
print(f"{len(m)} rows compared; largest difference {d:.2e}")
print("REPRODUCED EXACTLY" if d < 1e-6 else "MISMATCH")
