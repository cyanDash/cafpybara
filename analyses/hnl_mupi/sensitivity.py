"""Statistics-only expected sensitivity to |U_mu4|^2 for HNL -> mu pi.

Two stages, so the fit inputs are files like hnlpi0's:

1. :func:`pyhf_input_dict` histograms a selected frame into hnlpi0's pyhf-input
   schema, ``{'bins', 'counts', 'POT', 'total_err', 'stat_err'}`` (HNL dicts add
   ``'m'`` [MeV] and ``'U'``, the |U|^2 the counts correspond to), plus
   ``'mcstat_err'``. :func:`save_pyhf_inputs` / :func:`load_pyhf_inputs` write
   and read them with hnlpi0's file names (``mcbnb_dict_<suffix>.pkl``,
   ``mchnl_dict_<mass>_<suffix>.pkl``, ``dtbnb_dict_<suffix>.pkl``).
2. :func:`upper_limit_mu` / :func:`sensitivity_table` read those dicts and give
   the expected (median, +-1 sigma) CLs upper limit on the signal strength mu
   (pyhf, asymptotics, background-only Asimov data), and |U|^2 = U * sqrt(mu).

Error keys: ``stat_err`` is sqrt(counts), hnlpi0's convention (data-like
Poisson); ``mcstat_err`` is the MC statistical error, scale * sqrt(sum w^2);
``total_err`` is ``mcstat_err`` until systematics are added. ``var=None`` gives
one bin holding every slice (counting experiment).

Caveats: no systematics; no off-beam / in-time cosmic sample; the HNL samples
have no cosmic overlay yet; BNB MC statistics (not data statistics) dominate
the background error when the MC POT is below the target POT.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pyhf

from ...core.pyhf_helpers import make_model_stats_only

__all__ = ['hist_with_err', 'pyhf_input_dict', 'save_pyhf_inputs', 'load_pyhf_inputs',
           'upper_limit_mu', 'u2_limit', 'sensitivity_table']

pyhf.set_backend("numpy")


def hist_with_err(df: pd.DataFrame, var=None, bins=None, scale: float = 1.0, weighted: bool = False) -> dict:
    """Scaled counts and MC stat errors, ``{'bins', 'counts', 'stat_err'}``.

    ``var=None`` gives one bin holding every row (``bins`` = ``[0, 1]``);
    otherwise ``var`` is a column key and ``bins`` the bin edges.
    ``stat_err = scale * sqrt(sum w^2)``, with ``w = weights_mc`` if
    ``weighted`` else 1.
    """
    w = df["weights_mc"].to_numpy(float).ravel() if weighted else np.ones(len(df))
    if var is None:
        bins = np.array([0.0, 1.0])
        counts, sumw2 = np.array([w.sum()]), np.array([(w ** 2).sum()])
    else:
        x = df[var].to_numpy(float).ravel()
        counts, _ = np.histogram(x, bins=bins, weights=w)
        sumw2, _ = np.histogram(x, bins=bins, weights=w ** 2)
    return {"bins": np.asarray(bins, float), "counts": scale * counts, "stat_err": scale * np.sqrt(sumw2)}


def pyhf_input_dict(df: pd.DataFrame, pot: float, var=None, bins=None, scale: float = 1.0,
                    weighted: bool = False, m: float | None = None, U: float | None = None) -> dict:
    """One sample's pyhf input in hnlpi0's schema (see module docstring).

    ``pot`` is the POT the counts are normalised to (``scale`` already applied).
    For an HNL sample pass ``weighted=True``, ``m`` [MeV] and ``U``, the |U|^2
    the scaled counts correspond to (``simU * sqrt(display scale)`` if
    ``scale`` includes a display factor, as hnlpi0's ``plotU``).
    """
    h = hist_with_err(df, var, bins, scale, weighted)
    out = {"bins": h["bins"], "counts": h["counts"], "POT": pot,
           "total_err": h["stat_err"], "stat_err": np.sqrt(h["counts"]), "mcstat_err": h["stat_err"]}
    if m is not None:
        out["m"] = m
    if U is not None:
        out["U"] = U
    return out


def _files(suffix, masses, data):
    names = {"mcbnb": f"mcbnb_dict_{suffix}.pkl"}
    names.update({m: f"mchnl_dict_{int(m)}_{suffix}.pkl" for m in masses})
    if data:
        names["dtbnb"] = f"dtbnb_dict_{suffix}.pkl"
    return names


def save_pyhf_inputs(out_dir: str, suffix: str, mcbnb: dict, mchnl: dict, dtbnb: dict | None = None) -> list:
    """Pickle the dicts with hnlpi0's file names; ``mchnl`` is ``{mass: dict}``. Returns the paths.

    Refuses to overwrite an existing file (``~/scratch`` is write-once).
    """
    os.makedirs(out_dir, exist_ok=True)
    objs = {"mcbnb": mcbnb, **mchnl, **({"dtbnb": dtbnb} if dtbnb is not None else {})}
    paths = []
    for key, name in _files(suffix, list(mchnl), dtbnb is not None).items():
        path = os.path.join(out_dir, name)
        with open(path, "xb") as f:
            pickle.dump(objs[key], f)
        paths.append(path)
    return paths


def load_pyhf_inputs(in_dir: str, suffix: str, masses, data: bool = False) -> dict:
    """Read :func:`save_pyhf_inputs` output: ``{'mcbnb', <mass>..., ['dtbnb']}``."""
    out = {}
    for key, name in _files(suffix, masses, data).items():
        with open(os.path.join(in_dir, name), "rb") as f:
            out[key] = pickle.load(f)
    return out


def _model(sig, bkg, mc_stat):
    if mc_stat:
        return make_model_stats_only(sig, bkg)
    model = pyhf.Model({"channels": [{"name": "singlechannel", "samples": [
        {"name": "signal", "data": list(sig["counts"]),
         "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]},
        {"name": "background", "data": list(bkg["counts"]), "modifiers": []},
    ]}]})
    return model, list(bkg["counts"]) + model.config.auxdata


def upper_limit_mu(sig: dict, bkg: dict, cl: float = 0.95, mc_stat: bool = True,
                   err_key: str = "mcstat_err") -> dict:
    """Expected CLs upper limit on mu, ``{'median', 'minus1', 'plus1'}``.

    ``sig`` and ``bkg`` are pyhf-input dicts at the same POT. With ``mc_stat``
    the ``err_key`` errors (falling back to ``stat_err``) enter as stat
    modifiers; ``mc_stat=False`` is data statistics only. The signal is
    rescaled internally so the limit lies near mu = 1, inside pyhf's default
    normfactor bounds [0, 10].
    """
    err = lambda d: np.asarray(d.get(err_key, d["stat_err"]), float)
    s, b = np.asarray(sig["counts"], float), np.asarray(bkg["counts"], float)
    var_b = b + (err(bkg) ** 2 if mc_stat else 0)
    k = 1.96 * np.sqrt(var_b.sum()) / s.sum()          # rough limit, one-bin guess
    for _ in range(4):
        scaled = {"counts": list(s * k), "stat_err": list(err(sig) * k)}
        model, data = _model(scaled, {"counts": list(b), "stat_err": list(err(bkg))}, mc_stat)
        _, exp = pyhf.infer.intervals.upper_limits.upper_limit(
            data, model, scan=None, level=1 - cl, test_stat="qtilde")
        exp = np.asarray(exp, float)                    # -2, -1, 0, +1, +2 sigma
        if 0.05 < exp[2] < 5:
            return {"median": exp[2] * k, "minus1": exp[1] * k, "plus1": exp[3] * k}
        k *= exp[2]
    raise RuntimeError(f"limit did not converge near mu = 1 (last scaled median {exp[2]:.3g})")


def u2_limit(U: float, mu):
    """|U|^2 limit from a mu limit on counts made at |U|^2 = ``U`` (rate ~ |U|^4)."""
    return U * np.sqrt(mu)


def sensitivity_table(inputs: dict, cl: float = 0.95, mc_stat=(True, False)) -> pd.DataFrame:
    """Expected limits for every HNL mass in ``inputs`` (from :func:`load_pyhf_inputs`)."""
    bkg = inputs["mcbnb"]
    rows = []
    for key, sig in inputs.items():
        if key in ("mcbnb", "dtbnb"):
            continue
        s, b = np.sum(sig["counts"]), np.sum(bkg["counts"])
        for mc in mc_stat:
            lim = upper_limit_mu(sig, bkg, cl=cl, mc_stat=mc)
            rows.append({"mass [MeV]": sig.get("m", key), "mc_stat": mc, "S": s, "B": b,
                         "B MC err": np.sqrt(np.sum(np.square(bkg["mcstat_err"]))),
                         "S/sqrt(B)": s / np.sqrt(b),
                         "mu95 (-1s)": lim["minus1"], "mu95": lim["median"], "mu95 (+1s)": lim["plus1"],
                         "|U|^2 (-1s)": u2_limit(sig["U"], lim["minus1"]),
                         "|U|^2 limit": u2_limit(sig["U"], lim["median"]),
                         "|U|^2 (+1s)": u2_limit(sig["U"], lim["plus1"])})
    return pd.DataFrame(rows)
