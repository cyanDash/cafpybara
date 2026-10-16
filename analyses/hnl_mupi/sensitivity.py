"""Expected sensitivity to |U_mu4|^2 for HNL -> mu pi.

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
Poisson); ``mcstat_err`` is the MC statistical error, scale * sqrt(sum w^2).
With ``systs=`` (a :func:`~.funcs.systs_input`), :func:`pyhf_input_dict` also
runs :func:`~cafpybara.core.funcs.get_total_cov` and adds ``syst_cov`` (counts^2),
``syst_err`` and one ``<category>_err`` per source category (``DetVar_err``,
``BeamExposure_err``, ``NTargets_err``); ``total_err`` is
sqrt(mcstat_err^2 + syst_err^2), else ``mcstat_err``. ``var=None`` gives one
bin holding every slice (counting experiment).

:func:`upper_limit_mu` puts ``total_err`` in per-bin ``staterror`` /
``shapesys`` modifiers, so bin-to-bin correlations in ``syst_cov`` are dropped
(exact for one bin).

Caveats: no multisim (flux/xsec/G4) or cosmic systematics; no off-beam /
in-time cosmic sample; the HNL samples have no cosmic overlay yet; the detvar
sample is ~7e18 POT.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pyhf

from ...core.funcs import get_total_cov
from ...core.physics import integrated_flux
from ...core.pyhf_helpers import make_model_stats_only

__all__ = ['hist_with_err', 'syst_errors', 'pyhf_input_dict', 'save_pyhf_inputs', 'load_pyhf_inputs',
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


# get_total_cov needs a real column; for var=None any always-filled column in
# the bins [0, 1] works, since get_hist1d folds under/overflow into the edge bins.
_ONE_BIN_VAR = ('slc', 'vertex', 'x', '', '', '')


def syst_errors(df: pd.DataFrame, systs, var=None, bins=None, scale: float = 1.0) -> dict:
    """Systematic errors in scaled counts from ``get_total_cov(df, **systs.to_kwargs())``.

    Returns ``{'syst_cov', 'syst_err', '<category>_err', ...}``. ``df`` needs
    ``weights_mc``; ``systs.mcbnb_pot`` must be the POT of ``df`` (before
    ``scale``).
    """
    expect = hist_with_err(df, var, bins, scale, weighted=True)["counts"]
    if var is None:
        var, bins = _ONE_BIN_VAR, np.array([0.0, 1.0])
    out = get_total_cov(df, reco_var=var, bins=bins, **systs.to_kwargs())
    to_counts = integrated_flux * (systs.mcbnb_pot / 1e6) * scale
    cv = out.rate_hist_cv * to_counts
    if not np.allclose(cv, expect, rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"get_total_cov CV {cv} != histogram {expect}. Either systs.mcbnb_pot is not "
            "the POT of df, or some rows of var are NaN or outside bins: get_total_cov "
            "folds those into the edge bins, hist_with_err drops them. Select rows with "
            "var inside bins first (the same cut must then go into systs' cuts).")
    res = {"syst_cov": out.rate_cov * to_counts ** 2}
    res["syst_err"] = np.sqrt(np.clip(np.diag(res["syst_cov"]), 0, None))
    cats = {}
    for key, cat in zip(out.rate_syst_df["key"], out.rate_syst_df["category"]):
        entry = out.rate_syst_dict.get(key, out.rate_syst_dict.get(f"{cat}_{key}"))
        if entry is not None:                           # Datastat is not in rate_syst_dict
            cats[cat] = cats.get(cat, 0) + entry["cov"]
    for cat, cov in cats.items():
        res[f"{cat}_err"] = np.sqrt(np.clip(np.diag(cov), 0, None)) * to_counts
    return res


def pyhf_input_dict(df: pd.DataFrame, pot: float, var=None, bins=None, scale: float = 1.0,
                    weighted: bool = False, m: float | None = None, U: float | None = None,
                    systs=None) -> dict:
    """One sample's pyhf input in hnlpi0's schema (see module docstring).

    ``pot`` is the POT the counts are normalised to (``scale`` already applied).
    For an HNL sample pass ``weighted=True``, ``m`` [MeV] and ``U``, the |U|^2
    the scaled counts correspond to (``simU * sqrt(display scale)`` if
    ``scale`` includes a display factor, as hnlpi0's ``plotU``).
    ``systs`` (a :func:`~.funcs.systs_input` built with ``df``'s own POT and
    cuts) adds the systematic errors; see :func:`syst_errors`.
    """
    h = hist_with_err(df, var, bins, scale, weighted)
    out = {"bins": h["bins"], "counts": h["counts"], "POT": pot,
           "total_err": h["stat_err"], "stat_err": np.sqrt(h["counts"]), "mcstat_err": h["stat_err"]}
    if systs is not None:
        out.update(syst_errors(df, systs, var, bins, scale))
        out["total_err"] = np.sqrt(h["stat_err"] ** 2 + out["syst_err"] ** 2)
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
                   err_key: str = "total_err") -> dict:
    """Expected CLs upper limit on mu, ``{'median', 'minus1', 'plus1'}``.

    ``sig`` and ``bkg`` are pyhf-input dicts at the same POT. With ``mc_stat``
    the ``err_key`` errors (MC stat plus systematics by default; falling back
    to ``stat_err``) enter as per-bin modifiers; ``mc_stat=False`` is data
    statistics only. The signal is
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
                         "B syst err": np.sqrt(np.sum(np.square(bkg.get("syst_err", 0.0)))),
                         "S/sqrt(B)": s / np.sqrt(b),
                         "mu95 (-1s)": lim["minus1"], "mu95": lim["median"], "mu95 (+1s)": lim["plus1"],
                         "|U|^2 (-1s)": u2_limit(sig["U"], lim["minus1"]),
                         "|U|^2 limit": u2_limit(sig["U"], lim["median"]),
                         "|U|^2 (+1s)": u2_limit(sig["U"], lim["plus1"])})
    return pd.DataFrame(rows)
