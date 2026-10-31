"""
The cross-validated identity rule on 74's balanced subsamples, under both
partitions (companion to 81_identity_routes.py; no model is fitted).

74 reported the rule only under window-shuffled folds. With equal class counts
within each participant, the class most frequent among a participant's
training windows is the least frequent among their held-out windows, so the
rule should fall below chance under either partition.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ic = load("ic", "74_identity_controls.py")
ir = load("ir", "81_identity_routes.py")


def main():
    d, fc = ic.load()
    three = sorted(g for g, v in d.groupby("subject").y.nunique().items()
                   if v == 3)
    rows = []
    for i in range(ic.N_BAL):
        dd = ir.balanced_subsample(d, three, i)
        y, s, r = dd.y.to_numpy(), dd.subject.to_numpy(), dd.segment_id.to_numpy()
        for part, grp in (("window-shuffled", None), ("recording-grouped", r)):
            ide, _ = ic.identity_pred(y, s, ic.folds(y, grp, ic.SEEDS[0]))
            rows.append(dict(subsample=i, partition=part, windows=len(y),
                             id_acc=accuracy_score(y, ide)))
    a = pd.DataFrame(rows)
    a.round(4).to_csv(os.path.join(ic.OUT, "identity_routes_rule.csv"),
                      index=False)
    print(a.groupby("partition").id_acc.agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
