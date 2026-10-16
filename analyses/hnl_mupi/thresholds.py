"""Saved thresholds of the track cuts: one ``thresholds.txt`` with a line per number.

Format, one ``key = value`` per line, ``#`` starts a comment::

    min_len = 3.0      # trk_quality: track length > this [cm]

Keys are the :data:`~cafpybara.analyses.hnl_mupi.preprocess.PFP_CUTS` keys plus
``open_angle_max`` (the ``open_angle`` cut interval). :mod:`tuning` writes the
file from the FOM scans; the loaders read it::

    t = ana.read_thresholds()
    bnb, pot, _ = ana.load_mc(config.BNB_FILE,
                              preprocess_fn=partial(ana.preprocess_mcbnb, thresholds=t),
                              cuts=ana.DEFAULT_CUTS + ana.track_cuts(t["open_angle_max"]))
"""
from __future__ import annotations

from pathlib import Path

from .analysis import OPEN_ANGLE_MAX
from .preprocess import PFP_CUTS

__all__ = ['THRESHOLD_FILE', 'DEFAULT_THRESHOLDS', 'THRESHOLD_CUT', 'read_thresholds', 'write_thresholds']

THRESHOLD_FILE = Path(__file__).with_name("thresholds.txt")

DEFAULT_THRESHOLDS = {**PFP_CUTS, "open_angle_max": OPEN_ANGLE_MAX}

# The cut each threshold belongs to, and what it is (written as the line's comment).
THRESHOLD_CUT = {
    "trk_score":         ("trk_quality", "track-like: trackScore >"),
    "min_len":           ("trk_quality", "good track length > [cm]"),
    "max_dist_vtx":      ("trk_quality", "good track start-to-vertex distance < [cm]"),
    "mip_min_len":       ("mip", "MIP track length > [cm]"),
    "mip_chi2mu_max":    ("mip", "MIP track chi2_mu (per point, plane 2) <"),
    "mip_chi2p_min":     ("mip", "MIP track chi2_p (per point, plane 2) >"),
    "shw_score":         ("no_shower", "primary shower: 0 < trackScore <"),
    "shw_min_energy":    ("no_shower", "primary shower energy > [GeV]"),
    "open_angle_max":    ("open_angle", "MIP-pair opening angle < [deg]"),
    "contain_margin":    ("contained", "track ends at least this far inside the TPC [cm]"),
    "michel_score":      ("no_michel", "Michel-like: 0 < trackScore <"),
    "michel_max_energy": ("no_michel", "Michel-like visible energy < [GeV]"),
}


def read_thresholds(path=None) -> dict:
    """Read a thresholds file (default :data:`THRESHOLD_FILE`); keys it lacks keep their defaults."""
    out = dict(DEFAULT_THRESHOLDS)
    for line in Path(path or THRESHOLD_FILE).read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key not in DEFAULT_THRESHOLDS:
            raise KeyError(f"unknown threshold '{key}'; known: {sorted(DEFAULT_THRESHOLDS)}")
        out[key] = float(val)
    return out


def write_thresholds(values: dict, path=None, header: str = "", notes: dict | None = None) -> Path:
    """Write ``values`` (a thresholds dict) in cut order, one line per number.

    ``header`` becomes ``#`` lines at the top; ``notes[key]`` is appended to that line's comment
    (e.g. the FOM at the chosen value). Overwrites an existing file (it lives in the repo).
    """
    path = Path(path or THRESHOLD_FILE)
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    order = sorted(THRESHOLD_CUT, key=lambda k: [c for c, _ in THRESHOLD_CUT.values()].index(THRESHOLD_CUT[k][0]))
    for k in order:
        cut, desc = THRESHOLD_CUT[k]
        note = f"; {notes[k]}" if notes and k in notes else ""
        lines.append(f"{k} = {values[k]:g}   # {cut}: {desc}{note}")
    path.write_text("\n".join(lines) + "\n")
    return path
