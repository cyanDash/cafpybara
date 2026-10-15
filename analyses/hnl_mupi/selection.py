"""HNL mu-pi cut-flow table and threshold scan.

Prototyped here, not in ``core/``, until the migrated plots match the
selection notebook. Both take a dict of loaded slice frames keyed by sample
name (e.g. ``{"bnb": bnb, "m300": hnl300}``) carrying the ``signal`` column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.selection import select
from .analysis import signal_categories, signal_dict

__all__ = ['cutflow', 'scan_cut']

_NU_BKG = [signal_dict[k] for k in ("NC", "CCQE", "CC2p2h", "CCRES", "CCDIS", "CCother")]


def _hnl_signal(df):
    return df[df.signal == signal_dict["hnl"]]


def cutflow(frames: dict, cuts, bkg: str = "bnb", weights: dict | None = None) -> pd.DataFrame:
    """Slices left after each cut.

    For ``bkg``: weighted count per background category and in total. For every
    other sample: the number of HNL-matched slices and the efficiency relative
    to the first row (the cafpyana preselection).
    """
    weights = weights or {}
    stages = {k: {"preselection": df, **select(df, cuts=cuts, savedict=True, check_preprocessed=False)}
              for k, df in frames.items()}
    labels = {"preselection": "cafpyana preselection", **{c.name: c.label for c in cuts}}
    rows = []
    for stage, label in labels.items():
        row = {"cut": label}
        for k, st in stages.items():
            df = st[stage]
            if k == bkg:
                w = weights.get(k, 1.0)
                for cat, entry in signal_categories.items():
                    if cat in ("hnl", "hnlcosmic", "offbeam"):
                        continue
                    row[entry["label"]] = (df.signal == entry["value"]).sum() * w
                row[f"{k} total"] = len(df) * w
            else:
                n = len(_hnl_signal(df))
                n0 = len(_hnl_signal(st["preselection"]))
                row[f"{k} sig. slices"] = n
                row[f"{k} eff."] = n / n0 if n0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("cut")


def scan_cut(frames: dict, getter, thresholds, op: str = ">", bkg: str = "bnb") -> pd.DataFrame:
    """Kept fractions for ``getter(df) op threshold`` at each threshold.

    ``op`` is ``'>'``, ``'>='``, ``'<'`` or ``'window'`` (threshold = ``(lo, hi)``,
    keeps ``lo <= v < hi``). Columns: HNL efficiency per signal sample, and the
    kept fraction of the ``bkg`` sample's cosmics, nu in FV and nu outside FV.
    Non-finite values and the -999/-9999 sentinels never pass.
    """
    def keep(df, t):
        v = np.asarray(getter(df), dtype=float)
        v[(v == -999) | (v == -9999)] = np.nan
        with np.errstate(invalid="ignore"):
            if op == "window":
                m = (v >= t[0]) & (v < t[1])
            else:
                m = {">": v > t, ">=": v >= t, "<": v < t}[op]
        return m & np.isfinite(v)

    b = frames[bkg]
    sig = b.signal.to_numpy()
    rows = []
    for t in thresholds:
        row = {"thr": t}
        for k, df in frames.items():
            if k != bkg:
                row[f"eff {k}"] = keep(_hnl_signal(df), t).mean()
        kb = keep(b, t)
        row["kept cosmic"] = kb[sig == signal_dict["cosmic"]].mean()
        row["kept nu in FV"] = kb[np.isin(sig, _NU_BKG)].mean()
        row["kept nu out FV"] = kb[sig == signal_dict["nonFV"]].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("thr").round(3)
