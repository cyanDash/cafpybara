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

__all__ = ['cutflow', 'scan_cut', 'eff_purity_flow', 'split_table']

_NU_BKG = [signal_dict[k] for k in ("NC", "CCQE", "CC2p2h", "CCRES", "CCDIS", "CCother")]


def _hnl_signal(df):
    return df[df.signal == signal_dict["hnl"]]


def _hnl_weights(df):
    """``weights_mc`` if the frame has it (HNL rate weights), else ones."""
    return df["weights_mc"].to_numpy(float).ravel() if "weights_mc" in df.columns else np.ones(len(df))


def cutflow(frames: dict, cuts, bkg: str = "bnb", weights: dict | None = None) -> pd.DataFrame:
    """Slices left after each cut.

    For ``bkg``: weighted count per background category and in total. For every
    other sample: the number of HNL-matched slices and the efficiency relative
    to the first row (the cafpyana preselection). When the HNL frame has
    ``weights_mc`` (see :func:`~cafpybara.analyses.hnl_mupi.io.load_mchnl`), also
    the expected HNL-matched slices, ``sum(weights_mc) * weights[k]`` (pass
    ``weights[k] = target_pot / hnl_pot``), and the rate-weighted efficiency.
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
                sig, sig0 = _hnl_signal(df), _hnl_signal(st["preselection"])
                n, n0 = len(sig), len(sig0)
                row[f"{k} sig. slices"] = n
                row[f"{k} eff."] = n / n0 if n0 else np.nan
                if "weights_mc" in df.columns:
                    w, w0 = _hnl_weights(sig).sum(), _hnl_weights(sig0).sum()
                    row[f"{k} exp. slices"] = w * weights.get(k, 1.0)
                    row[f"{k} eff. (wtd)"] = w / w0 if w0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("cut")


def scan_cut(frames: dict, getter, thresholds, op: str = ">", bkg: str = "bnb") -> pd.DataFrame:
    """Kept fractions for ``getter(df) op threshold`` at each threshold.

    ``op`` is ``'>'``, ``'>='``, ``'<'`` or ``'window'`` (threshold = ``(lo, hi)``,
    keeps ``lo <= v < hi``). Columns: HNL efficiency per signal sample, and the
    kept fraction of the ``bkg`` sample's cosmics, nu in FV and nu outside FV.
    The HNL efficiency is rate-weighted when the frame has ``weights_mc``.
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
                hs = _hnl_signal(df)
                w = _hnl_weights(hs)
                row[f"eff {k}"] = (keep(hs, t) * w).sum() / w.sum() if w.sum() else np.nan
        kb = keep(b, t)
        row["kept cosmic"] = kb[sig == signal_dict["cosmic"]].mean()
        row["kept nu in FV"] = kb[np.isin(sig, _NU_BKG)].mean()
        row["kept nu out FV"] = kb[sig == signal_dict["nonFV"]].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("thr").round(3)


def eff_purity_flow(signal: dict, background: dict, cuts, sig_scale: dict, bkg_scale: float = 1.0,
                    split: bool = False) -> pd.DataFrame:
    """HNL efficiency and purity after each cut, cumulative, starting at the preselection.

    ``signal`` maps a label (e.g. ``"m300"``) to an HNL frame; only its HNL-matched
    slices count (``hnlcosmic`` slices are dropped: those cosmics are already in the
    background samples). ``background`` maps a label to a background frame (BNB MC,
    off-beam); every slice counts. Each slice is weighted by its ``weights_mc`` (ones
    if absent), times ``sig_scale[label]`` for a signal frame and ``bkg_scale`` for
    every background frame.

    Returns one row per stage, with ``B`` (and ``B <label>`` per background frame) and,
    per signal label ``k``: ``S k`` (expected signal slices), ``eff k`` (``S`` over its
    preselection value), ``purity k`` = ``S / (S + B)`` and ``ssb_rel k`` = ``S/sqrt(B)``
    over its preselection value. Purity depends on the HNL coupling the weights were made
    at (rate ~ |U|^4); efficiency and ``ssb_rel`` do not.

    ``split=True`` (the last cut must be ``contained``, see ``track_cuts(containment_last=True)``):
    containment is the separator of two final samples, not a filter. The ``contained`` row is the
    contained branch, and an ``uncontained`` row (the stage before it minus the contained slices)
    is added; both have ``branch = True``, every other row ``False``.
    """
    if split and (not cuts or cuts[-1].name != "contained"):
        raise ValueError("split=True needs 'contained' as the last cut (track_cuts(containment_last=True))")

    def stages(df):
        st = {"preselection": df, **select(df, cuts=cuts, savedict=True, check_preprocessed=False)}
        if split:
            prev = st[cuts[-2].name] if len(cuts) > 1 else df
            st["uncontained"] = prev[~prev.index.isin(st["contained"].index)]
        return st

    sig_st = {k: stages(_hnl_signal(df)) for k, df in signal.items()}
    bkg_st = {k: stages(df) for k, df in background.items()}
    labels = {"preselection": "cafpyana preselection", **{c.name: c.label for c in cuts}}
    if split:
        labels["contained"] = "split: contained"
        labels["uncontained"] = "split: uncontained"
    rows = []
    for stage, label in labels.items():
        b_each = {k: _hnl_weights(st[stage]).sum() * bkg_scale for k, st in bkg_st.items()}
        b0 = sum(_hnl_weights(st["preselection"]).sum() for st in bkg_st.values()) * bkg_scale
        b = sum(b_each.values())
        row = {"stage": stage, "cut": label, "branch": split and stage in ("contained", "uncontained"),
               "B": b, **{f"B {k}": v for k, v in b_each.items()}}
        for k, st in sig_st.items():
            s = _hnl_weights(st[stage]).sum() * sig_scale.get(k, 1.0)
            s0 = _hnl_weights(st["preselection"]).sum() * sig_scale.get(k, 1.0)
            row[f"S {k}"] = s
            row[f"eff {k}"] = s / s0 if s0 else np.nan
            row[f"purity {k}"] = s / (s + b) if s + b else np.nan
            row[f"ssb_rel {k}"] = (s / np.sqrt(b)) / (s0 / np.sqrt(b0)) if b > 0 and s0 and b0 > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("stage")


def split_table(signal: dict, background: dict, cuts, sig_scale: dict, bkg_scale: float = 1.0) -> pd.DataFrame:
    """S, B and S/sqrt(B) after ``cuts``, split into contained and uncontained slices, plus their combination.

    Arguments as in :func:`eff_purity_flow`; a ``contained`` cut in ``cuts`` is ignored (it is the
    separator here). A slice is ``contained`` when ``n_uncontained == 0``
    (no good track ends within ``contain_margin`` of the TPC edge or in the high y-z region).
    Rows ``(k, category)`` per signal label ``k``, categories ``contained``, ``uncontained``,
    ``combined``; columns ``S``, ``B``, ``B_n`` (unweighted background slices), ``S/sqrt(B)``
    and ``rel. presel.`` (S/sqrt(B) over its value at the preselection, before any cut).
    The combined S/sqrt(B) is ``sqrt(sum_i S_i^2 / B_i)`` over the two independent categories.
    """
    cuts = [c for c in cuts if c.name != "contained"]

    def wsum(df, mask):
        return _hnl_weights(df)[mask].sum()

    def contained(df):
        return (df["n_uncontained"].to_numpy(float).ravel() == 0)

    bkg_sel = {k: select(df, cuts=cuts, check_preprocessed=False) for k, df in background.items()}
    b0 = sum(_hnl_weights(df).sum() for df in background.values()) * bkg_scale
    cats = {"contained": True, "uncontained": False}
    B = {c: sum(wsum(df, contained(df) == v) for df in bkg_sel.values()) * bkg_scale for c, v in cats.items()}
    Bn = {c: int(sum((contained(df) == v).sum() for df in bkg_sel.values())) for c, v in cats.items()}
    rows = []
    for k, df in signal.items():
        sig0 = _hnl_signal(df)
        sig = select(sig0, cuts=cuts, check_preprocessed=False)
        sc = sig_scale.get(k, 1.0)
        s0 = _hnl_weights(sig0).sum() * sc
        ref = s0 / np.sqrt(b0) if b0 > 0 else np.nan
        z2 = 0.0
        for c, v in cats.items():
            s = wsum(sig, contained(sig) == v) * sc
            z = s / np.sqrt(B[c]) if B[c] > 0 else np.nan
            z2 += z ** 2 if np.isfinite(z) else 0.0
            rows.append({"signal": k, "category": c, "S": s, "B": B[c], "B_n": Bn[c], "S/sqrt(B)": z, "rel. presel.": z / ref})
        zc = np.sqrt(z2)
        rows.append({"signal": k, "category": "combined", "S": sum(r["S"] for r in rows[-2:]), "B": sum(B.values()),
                     "B_n": sum(Bn.values()), "S/sqrt(B)": zc, "rel. presel.": zc / ref})
    return pd.DataFrame(rows).set_index(["signal", "category"])
