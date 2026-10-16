"""Tune the track-cut thresholds with a Punzi figure of merit, and save them.

The thresholds of :data:`~cafpybara.analyses.hnl_mupi.preprocess.PFP_CUTS` are
applied to every PFP of a slice, so they cannot be changed on the collapsed
slice frames. Tuning therefore works on the PFP-level frames loaded with
``preprocess_mcbnb_pfp`` / ``preprocess_mchnl_pfp`` (one row per PFP, the slice
cuts already applied): :func:`make_sample` computes the threshold-independent
PFP quantities once (:func:`~preprocess.pfp_features`), and every scan point
then only re-runs the cheap :func:`~preprocess.pfp_counts`.

For each scan in :data:`SCANS` (in cut order) the cuts before it are held at
their current thresholds, the scanned threshold is varied, and the Punzi FOM
``eff_S / (a/2 + sqrt(B))`` is evaluated after the cut it belongs to (``eff_S``:
rate-weighted HNL-matched slices over those left by the slice cuts; ``B``: BNB
slices, all categories, at the target POT). The threshold is then set to the
loosest value whose FOM is within ``tol`` of the peak (so a flat plateau, or
a peak driven by a handful of BNB slices, does not tighten the cut for nothing)
and the next scan starts from it. Truth enters only through the HNL-matched
slice label; every cut stays reco-level. :func:`write_thresholds` saves the result.

Chosen values depend on the reference mass (``ref_mass``) and on the BNB
statistics at the chosen point (``B_n``, reported in every scan table).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...core.selection import _mask
from .analysis import signal_dict, track_cuts
from .preprocess import PFP_CUTS, PFP_LEVEL, pfp_counts, pfp_features

__all__ = ['Scan', 'SCANS', 'TuneSample', 'make_sample', 'tune', 'candidates', 'plot_param']


@dataclass
class Scan:
    """One threshold: its cut, scanned values, and whether larger values make the cut tighter."""
    param: str
    stage: str
    values: list
    tighter_up: bool
    feature: str | None = None        # PFP feature histogrammed in the distribution plot
    label: str = ""
    bins: np.ndarray = field(default=None, repr=False)


SCANS = [
    Scan("min_len", "trk_quality", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20], True, "len", "Track length [cm]", np.linspace(0, 60, 61)),
    Scan("max_dist_vtx", "trk_quality", [2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50, 80], False, "dist",
         "Track start to vertex [cm]", np.linspace(0, 60, 61)),
    Scan("mip_min_len", "mip", [5, 8, 10, 12, 15, 20, 30], True, "len", "Track length [cm]", np.linspace(0, 200, 81)),
    Scan("mip_chi2mu_max", "mip", [5, 10, 15, 20, 30, 40, 60], False, "chi2mu",
         r"$\chi^2_\mu$ (per point)", np.linspace(0, 100, 51)),
    Scan("mip_chi2p_min", "mip", [20, 40, 60, 85, 100, 120, 150], True, "chi2p",
         r"$\chi^2_p$ (per point)", np.linspace(0, 300, 61)),
    Scan("shw_min_energy", "no_shower", [0.01, 0.02, 0.035, 0.055, 0.08, 0.12, 0.2, 0.3, 0.5, 1.0], False, "shw_e",
         "Primary shower energy [GeV]", np.linspace(0, 0.5, 51)),
    Scan("open_angle_max", "open_angle", [10, 20, 30, 45, 60, 75, 90, 110, 130, 150, 160, 170, 175], False, None,
         "MIP-pair opening angle [deg]", np.linspace(0, 180, 61)),
    Scan("contain_margin", "contained", [0, 2, 5, 10, 15, 20, 30], True, "edge",
         "Track end distance inside TPC [cm]", np.linspace(-10, 60, 71)),
    Scan("michel_max_energy", "no_michel", [0.015, 0.03, 0.045, 0.06, 0.09, 0.12, 0.2, 0.3, 0.5], True, "vis",
         "Visible energy [GeV]", np.linspace(0, 0.3, 61)),
    Scan("michel_score", "no_michel", [0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0], True, "ts",
         "Track score", np.linspace(0, 1, 51)),
]


@dataclass
class TuneSample:
    """PFP features and the per-slice weight, signal flag and category of one sample."""
    feat: pd.DataFrame
    slc: pd.DataFrame      # index: slice; columns w (scaled weight), sig (HNL-matched), signal (category)


def make_sample(df_pfp: pd.DataFrame, scale: float = 1.0, hnl: bool = False) -> TuneSample:
    """Features and slice table of a PFP-level frame (slice cuts already applied).

    ``scale`` multiplies the slice weight: ``target_pot / pot`` (BNB; HNL, times
    ``weights_mc``).
    """
    feat = pfp_features(df_pfp)
    levels = [i for i in range(df_pfp.index.nlevels) if i != PFP_LEVEL]
    first = df_pfp.groupby(level=levels).head(1)
    first.index = first.index.droplevel(PFP_LEVEL)
    w = first["weights_mc"].to_numpy(float).ravel() * scale if hnl else np.full(len(first), float(scale))
    sig = first["signal"].to_numpy().ravel()
    slc = pd.DataFrame({"w": w, "signal": sig, "sig": sig == signal_dict["hnl"]}, index=first.index)
    return TuneSample(feat, slc)


def _passing(sample: TuneSample, thr: dict, upto: str | None, after: str | None = None):
    """Counts frame and mask of slices passing the track cuts up to and including ``upto``.

    With ``after`` set, ``upto`` is exclusive and the mask is the cuts before it.
    """
    cnt = pfp_counts(sample.feat, **{k: v for k, v in thr.items() if k in PFP_CUTS}).reindex(sample.slc.index)
    m = np.ones(len(cnt), bool)
    for spec in track_cuts(thr["open_angle_max"]):
        if after and spec.name == upto:
            break
        m &= _mask(cnt, spec).to_numpy()
        if spec.name == upto and not after:
            break
    return cnt, m


def _yields(sample: TuneSample, thr: dict, upto: str):
    _, m = _passing(sample, thr, upto)
    w, sig = sample.slc.w.to_numpy(), sample.slc.sig.to_numpy()
    return w[m].sum(), w[m & sig].sum(), int(m.sum())


def tune(bnb: TuneSample, hnl: dict, ref_mass, start: dict | None = None, scans=SCANS,
         a: float = 3.0, tol: float = 0.02, verbose: bool = True):
    """Run the sequential scans; returns ``(thresholds, {param: scan table})``.

    ``hnl`` is ``{mass: TuneSample}``, ``ref_mass`` the mass whose FOM sets the
    thresholds, ``a`` the Punzi significance and ``tol`` the plateau tolerance
    (module docstring). Scan tables hold, per value: ``B``, ``B_n`` (BNB
    slices), and per mass ``eff``, ``fom``, plus ``chosen``.
    """
    from .thresholds import read_thresholds
    thr = dict(start or read_thresholds())
    s0 = {m: s.slc.w[s.slc.sig].sum() for m, s in hnl.items()}
    tables = {}
    for sc in scans:
        rows = []
        for v in sc.values:
            t = {**thr, sc.param: v}
            b, _, bn = _yields(bnb, t, sc.stage)
            row = {"value": v, "B": b, "B_n": bn}
            for m, s in hnl.items():
                _, sg, _ = _yields(s, t, sc.stage)
                row[f"eff {m}"] = sg / s0[m]
                row[f"fom {m}"] = row[f"eff {m}"] / (a / 2 + np.sqrt(b))
            rows.append(row)
        tab = pd.DataFrame(rows)
        f = tab[f"fom {ref_mass}"]
        ok = tab[f >= (1 - tol) * f.max()]
        pick = ok.value.min() if sc.tighter_up else ok.value.max()
        tab["chosen"] = tab.value == pick
        thr[sc.param] = float(pick)
        tables[sc.param] = tab
        if verbose:
            edge = " [peak at the edge of the scanned range: extend it]" if f.idxmax() in (0, len(tab) - 1) else ""
            print(f"{sc.param:20s} -> {pick:g}   (FOM peak {f.max():.3g} at {tab.value[f.idxmax()]:g}){edge}")
    return thr, tables


def candidates(sample: TuneSample, thr: dict, sc: Scan, hnl: bool):
    """Rows of the distribution plot for ``sc``: DataFrame ``x``, ``signal``, ``weights_mc``.

    The candidate PFPs of the cut (all neutrino daughters passing the per-PFP
    requirements that precede the scanned threshold), in slices passing the
    cuts before it; ``hnl`` keeps only the HNL-matched slices. ``open_angle_max``
    uses one row per slice.
    """
    cnt, m = _passing(sample, thr, sc.stage, after=sc.stage)
    slc = sample.slc.assign(ok=m)
    if hnl:
        slc = slc[slc.sig]
    slc = slc[slc.ok]
    if sc.param == "open_angle_max":
        x = cnt.mip_open_angle.reindex(slc.index)
        return pd.DataFrame({"x": x.to_numpy(), "signal": slc.signal.to_numpy(), "weights_mc": slc.w.to_numpy()})
    f = sample.feat
    f = f[f.index.droplevel(PFP_LEVEL).isin(slc.index)]
    p = {**PFP_CUTS, **{k: v for k, v in thr.items() if k in PFP_CUTS}}
    good = f.nu_d & f.valid & (f.ts > p["trk_score"])
    sel = {
        "min_len": f.nu_d & f.valid & (f.ts > p["trk_score"]) & (f.dist < p["max_dist_vtx"]),
        "max_dist_vtx": f.nu_d & f.valid & (f.ts > p["trk_score"]) & (f.len > p["min_len"]),
        "mip_min_len": good & (f.len > p["min_len"]) & (f.dist < p["max_dist_vtx"]),
        "mip_chi2mu_max": good & (f.len > p["mip_min_len"]) & (f.dist < p["max_dist_vtx"]),
        "mip_chi2p_min": good & (f.len > p["mip_min_len"]) & (f.dist < p["max_dist_vtx"]),
        "shw_min_energy": f.nu_d & (f.ts > 0) & (f.ts < p["shw_score"]),
        "contain_margin": good & (f.len > p["min_len"]) & (f.dist < p["max_dist_vtx"]),
        "michel_max_energy": f.nu_d & (f.ts > 0) & (f.ts < p["michel_score"]) & (f.nchild == 0),
        "michel_score": f.nu_d & (f.ts > 0) & (f.nchild == 0) & (f.vis < p["michel_max_energy"]),
    }[sc.param]
    f = f[sel.to_numpy()]
    s = slc.reindex(f.index.droplevel(PFP_LEVEL))
    return pd.DataFrame({"x": f[sc.feature].to_numpy(), "signal": s.signal.to_numpy(), "weights_mc": s.w.to_numpy()})


def plot_param(sc: Scan, table: pd.DataFrame, thr: dict, bnb: TuneSample, hnl: dict, ref_mass,
               w_bnb: float, savefig: str = ""):
    """Distribution of the scanned variable (BNB stack, HNL line) with the Punzi FOM of the scan.

    Top: candidate PFPs (see :func:`candidates`), BNB at ``w_bnb`` per slice
    (already in the slice weights, so ``scale=1``) and the HNL at ``ref_mass``.
    Bottom: the cumulative per-candidate FOM of ``plot_mc_hnl`` plus, on the
    right axis, the slice-level FOM of the scan (what the threshold is set
    from), with the chosen value marked.
    """
    from .plotting import plot_mc_hnl
    mc = candidates(bnb, thr, sc, hnl=False)
    hs = candidates(hnl[ref_mass], thr, sc, hnl=True)
    fig, ax, ax_fom, _, _ = plot_mc_hnl(mc, hs, "x", sc.bins, scale_nu=1.0, scale_hnl=1.0,
                                        hnl_label=f"m{ref_mass} HNL", log_y=True, show_fom=True,
                                        fom_nsigma=3.0, xlabel=sc.label, title="")
    ax.set_title("")
    ax2 = ax_fom.twinx()
    ax2.plot(table.value, table[f"fom {ref_mass}"], "k.-", label="slice-level FOM (scan)")
    ax2.set_ylabel("slice-level\nPunzi FOM")
    ax2.set_ylim(bottom=0)
    ch = table.value[table.chosen].iloc[0]
    ax_fom.axvline(ch, color="red", lw=2, label=f"chosen {ch:g}")
    ax_fom.legend(fontsize="small", frameon=False, loc="upper left")
    ax2.legend(fontsize="small", frameon=False, loc="upper right")
    if savefig:
        fig.savefig(savefig, bbox_inches="tight")
    return fig
