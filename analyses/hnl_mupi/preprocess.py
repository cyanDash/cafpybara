"""HNL mu-pi preprocessing: PFParticle -> slice collapse plus flash-timing fixes.

cafpyana's ``*_savepfp`` frames hold one row per PFParticle, indexed by
``(__ntuple, entry, rec.slc..index, rec.slc.reco.pfp..index)``. Every core
tool (``select``, ``merge_hdr``, ``plot_var``) expects one row per slice, so
:func:`collapse_pfp` reduces the frame to one row per
``(__ntuple, entry, rec.slc..index)`` with:

* the ``slc`` columns (identical on every PFP row of a slice),
* ``n_trk``: number of track-like PFPs in the slice,
* ``long1``/``long2``: the longest and second-longest track-like PFP, as
  column groups with the same sub-columns as ``pfp`` (``('long1', 'trk',
  'len', ...)``). NaN when the slice has fewer tracks. Ranked by length only;
  this is not a muon/pion assignment.

Track-like = ``trackScore > TRACK_SCORE`` with a fitted track (``trk.len > 0``
and valid start/end), the same definition as the selection notebook.

Timing follows hnlpi0: the generator-specific BCFM flash-time bugfix is
applied in place to ``slc.barycenterFM.flashTime``, then
:func:`~cafpybara.core.preprocess.fix_timing_calibration` adds
``flashTime_calib`` and the bunch phase ``flashTime_calib_mod``. The -999
sentinel is set to NaN first, so the bugfix offset can't turn it into a
plausible-looking value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...core import timing_calibration as tc
from ...core.preprocess import (
    preprocess_mc, fix_timing_calibration, _skip_if_applied, _mark_applied,
)

__all__ = [
    'TRACK_SCORE', 'SLICE_LEVELS', 'PFP_LEVEL', 'pfp_col', 'track_like', 'collapse_pfp',
    'fix_bfm_flashtime_mcbnb', 'fix_bfm_flashtime_mchnl',
    'preprocess_mcbnb', 'preprocess_mchnl',
]

TRACK_SCORE = 0.51            # same track-like threshold as makedf (make_hnldf)
SLICE_LEVELS = [0, 1, 2]      # (__ntuple, entry, rec.slc..index)
# The PFP level (rec.slc.reco.pfp..index) is dropped by position, not as the
# last level: detvar frames carry an extra trailing ``file_idx`` level
# (core/detvar/store.py).
PFP_LEVEL = 3
_FT = ('slc', 'barycenterFM', 'flashTime', '', '', '')


def pfp_col(path: str, depth: int = 6) -> tuple:
    """``'pfp.trk.len'`` -> ``('pfp', 'trk', 'len', '', '', '')``."""
    p = path.split('.')
    return tuple(p + [''] * (depth - len(p)))


def track_like(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of track-like PFP rows."""
    return ((df.pfp.trackScore > TRACK_SCORE) & (df.pfp.trk.len > 0)
            & (df.pfp.trk.start.x != -999) & (df.pfp.trk.end.x != -999))


def _slice_index(idx: pd.MultiIndex) -> pd.MultiIndex:
    """PFP-level index -> slice-level index (drops ``PFP_LEVEL``)."""
    return idx.droplevel(PFP_LEVEL)


def _rank_group(trk: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rename the ``pfp`` columns of a one-row-per-slice track frame to ``name``."""
    out = trk['pfp'].copy()
    out.columns = pd.MultiIndex.from_tuples([(name,) + c for c in out.columns])
    out.index = _slice_index(trk.index)
    return out


def _add_track_derived(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Angle to the beam [deg], start-to-slice-vertex distance [cm] and plane-2
    chi2(mu)/chi2(pi) for a track group."""
    g = df[name]
    df[(name, 'theta_z', '', '', '', '')] = np.degrees(
        np.arccos(np.clip(g.trk.dir.z.astype(float), -1, 1)))
    df[(name, 'dist_vtx', '', '', '', '')] = np.sqrt(sum(
        (g.trk.start[a] - df.slc.vertex[a]) ** 2 for a in 'xyz'))
    pid = g.trk.chi2pid.I2
    df[(name, 'chi2_mu_over_pi_pl2', '', '', '', '')] = pid.chi2_muon / pid.chi2_pion
    return df


def collapse_pfp(df: pd.DataFrame) -> pd.DataFrame:
    """One row per slice, with ``n_trk`` and the ``long1``/``long2`` track groups."""
    name = 'pfp_collapse'
    if _skip_if_applied(df, name):
        return df
    trk = df[track_like(df)]
    srt = trk.sort_values(pfp_col('pfp.trk.len'), ascending=False, kind='stable')
    slc_of = _slice_index(srt.index)
    first = ~slc_of.duplicated()
    rest = srt[~first]
    long1 = _rank_group(srt[first], 'long1')
    long2 = _rank_group(rest[~_slice_index(rest.index).duplicated()], 'long2')

    slc_levels = [i for i in range(df.index.nlevels) if i != PFP_LEVEL]
    slc = df.drop(columns='pfp', level=0).groupby(level=slc_levels).head(1)
    slc.index = _slice_index(slc.index)
    n_trk = trk.groupby(level=slc_levels).size().reindex(slc.index).fillna(0)
    out = pd.concat([slc, long1.reindex(slc.index), long2.reindex(slc.index)], axis=1)
    out[('n_trk', '', '', '', '', '')] = n_trk.to_numpy()
    for g in ('long1', 'long2'):
        out = _add_track_derived(out, g)
    return _mark_applied(out, name)


def _nan_flash_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    df[_FT] = df[_FT].astype(float).where(df[_FT] > -998)
    return df


def fix_bfm_flashtime_mcbnb(df: pd.DataFrame) -> pd.DataFrame:
    """BNB-generator BCFM flash-time offset (+0.135 us cable, +2 bunch periods), in place."""
    name = 'bfm_flashtime_mcbnb'
    if _skip_if_applied(df, name):
        return df
    df = tc.bugfix_mcbnb_bfm_flashtime(_nan_flash_sentinel(df))
    return _mark_applied(df, name)


def fix_bfm_flashtime_mchnl(df: pd.DataFrame) -> pd.DataFrame:
    """MeVPrtl-generator BCFM flash-time offset (-0.135 us cable), in place."""
    name = 'bfm_flashtime_mchnl'
    if _skip_if_applied(df, name):
        return df
    df = tc.bugfix_mchnl_bfm_flashtime(_nan_flash_sentinel(df))
    return _mark_applied(df, name)


def preprocess_mcbnb(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to slices, then BNB-generator flash-time fix and timing calibration."""
    df = collapse_pfp(preprocess_mc(df))
    df = fix_bfm_flashtime_mcbnb(df)
    return fix_timing_calibration(df, period=tc.mcbnb_period_calib, t0_offset=tc.mcbnb_offset_calib)


def preprocess_mchnl(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to slices, then MeVPrtl-generator flash-time fix and timing calibration."""
    df = collapse_pfp(preprocess_mc(df))
    df = fix_bfm_flashtime_mchnl(df)
    return fix_timing_calibration(df, period=tc.mchnl_period_calib, t0_offset=tc.mchnl_offset_calib)
