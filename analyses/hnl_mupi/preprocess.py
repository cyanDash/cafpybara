"""HNL mu-pi preprocessing: PFParticle -> slice collapse plus flash-timing fixes.

cafpyana's ``*_savepfp`` frames hold one row per PFParticle, indexed by
``(__ntuple, entry, rec.slc..index, rec.slc.reco.pfp..index)``. Every core
tool (``select``, ``merge_hdr``, ``plot_var``) expects one row per slice, so
:func:`collapse_pfp` reduces the frame to one row per
``(__ntuple, entry, rec.slc..index)`` with:

* the ``slc`` columns (identical on every PFP row of a slice),
* ``n_trk``: number of track-like PFPs in the slice,
* the per-slice PFP selection counts of :func:`pfp_selection` (``n_good``,
  ``n_mip``, ``n_shw``, ``mip_open_angle``, ``n_uncontained``, ``n_cathode``,
  ``n_michel``), cut on in ``analysis.TRACK_CUTS``,
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
    'TRACK_SCORE', 'SLICE_LEVELS', 'PFP_LEVEL', 'PFP_CUTS', 'in_high_yz', 'pfp_col', 'track_like',
    'pfp_features', 'pfp_counts', 'pfp_selection', 'collapse_pfp',
    'fix_bfm_flashtime_mcbnb', 'fix_bfm_flashtime_mchnl',
    'preprocess_mcbnb', 'preprocess_mchnl', 'preprocess_mcbnb_pfp', 'preprocess_mchnl_pfp',
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


# Thresholds of the PFP-level selection (all reco-level; energies in GeV, lengths
# in cm). Every PFP of the slice is tested, so these are applied here, before
# the collapse to one row per slice. Tune by passing overrides to
# :func:`pfp_selection` through a ``functools.partial`` preprocess function.
PFP_CUTS = dict(
    trk_score=0.5,          # track-like: trackScore > this
    min_len=3.0,            # good track: length > this
    max_dist_vtx=10.0,      # good track: start-to-reco-vertex distance < this
    mip_min_len=10.0,       # MIP track: length > this ...
    mip_chi2mu_max=20.0,    # ... chi2_mu (per point, collection plane) < this ...
    mip_chi2p_min=85.0,     # ... and chi2_p > this
    shw_score=0.5,          # primary shower: 0 < trackScore < this ...
    shw_min_energy=0.055,   # ... and energy > this (lower-energy ones may be Michels)
    contain_margin=5.0,     # contained: start and end this far inside the TPC
    michel_score=0.6,       # Michel-like: 0 < trackScore < this ...
    michel_max_energy=0.045,  # ... visible energy < this ...
                              # ... and no daughter PFPs
)
_TPC = dict(x=(-200.0, 200.0), y=(-200.0, 200.0), z=(0.0, 500.0))
_M_PION = 0.13957
# High y-z region (E-field distortions): z > 250 cm and y > 100 cm. The complement of
# ``makedf.util.InFV(det="SBND_nohighyz")`` inside its x/z box (checked in tuning tests).
HIGHYZ_Z_MIN, HIGHYZ_Y_MIN = 250.0, 100.0


def in_high_yz(pos) -> np.ndarray:
    """Boolean array: ``pos`` (columns x, y, z) lies in the high y-z region."""
    return ((pos.z > HIGHYZ_Z_MIN) & (pos.y > HIGHYZ_Y_MIN)).to_numpy()


def _edge_dist(pos):
    """Distance [cm] from ``pos`` to the nearest TPC face (negative outside)."""
    return np.minimum.reduce([np.minimum(pos[a] - lo, hi - pos[a]).to_numpy(float)
                              for a, (lo, hi) in _TPC.items()])


def pfp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Threshold-independent per-PFP quantities for :func:`pfp_counts` (reco only).

    Indexed like ``df`` (the PFP-level frame). Computed once, so a threshold
    scan only re-runs the cheap :func:`pfp_counts`. Columns: ``nu_d`` (daughter
    of the neutrino PFP), ``ts`` (trackScore), ``valid`` (fitted track), ``len``,
    ``dist`` (track start to reco vertex), ``chi2mu``/``chi2p`` (plane 2, per
    point, NaN if ``pid_ndof <= 0``), ``shw_e`` (shower energy, GeV), ``edge``
    (smallest distance of the track start/end to a TPC face), ``highyz`` (start or end
    in the high y-z region), ``cathode``
    (start and end on opposite sides of x = 0), ``vis`` (visible energy: shower
    energy if there is one, else range KE under the pion hypothesis),
    ``nchild`` (daughter PFPs) and ``dx``/``dy``/``dz`` (track direction).
    """
    p = df.pfp
    st, en = p.trk.start, p.trk.end
    pid = p.trk.chi2pid.I2
    ok = (pid.pid_ndof > 0).to_numpy()   # the CAF chi2pid values are already per point
    shw_e = p.shw.bestplane_energy.astype(float)
    ke = np.sqrt(p.trk.rangeP.p_pion.astype(float) ** 2 + _M_PION ** 2) - _M_PION
    lv = [df.index.get_level_values(i) for i in SLICE_LEVELS]
    par = pd.DataFrame({'a': lv[0], 'b': lv[1], 'c': lv[2], 'p': p.parent.to_numpy()})
    nchild = par.groupby(['a', 'b', 'c', 'p']).size().reindex(
        pd.MultiIndex.from_arrays([*lv, p.id.to_numpy()])).fillna(0).to_numpy()
    f = pd.DataFrame({
        'nu_d': ((p.parent_is_primary == 1) & (p.parent != -1)).to_numpy(),
        'ts': p.trackScore.astype(float).to_numpy(),
        'valid': ((p.trk.len > 0) & (st.x != -999) & (en.x != -999)).to_numpy(),
        'len': p.trk.len.astype(float).to_numpy(),
        'dist': np.sqrt(sum((st[a] - df.slc.vertex[a]) ** 2 for a in 'xyz')).to_numpy(float),
        'chi2mu': pid.chi2_muon.astype(float).where(ok).to_numpy(),
        'chi2p': pid.chi2_proton.astype(float).where(ok).to_numpy(),
        'shw_e': shw_e.to_numpy(),
        'edge': np.minimum(_edge_dist(st), _edge_dist(en)),
        'cathode': ((st.x * en.x) < 0).to_numpy(),
        'highyz': in_high_yz(st) | in_high_yz(en),
        'vis': shw_e.where(shw_e > 0, ke).to_numpy(),
        'nchild': nchild,
        'dx': p.trk.dir.x.to_numpy(float), 'dy': p.trk.dir.y.to_numpy(float),
        'dz': p.trk.dir.z.to_numpy(float),
    }, index=df.index)
    return f


def pfp_counts(feat: pd.DataFrame, **over) -> pd.DataFrame:
    """Per-slice counts from the cuts on every neutrino-daughter PFP, from :func:`pfp_features`.

    Returns a frame indexed like the slice (``PFP_LEVEL`` dropped) with:

    * ``n_good``: daughters of the neutrino with trackScore > ``trk_score``,
      length > ``min_len`` and start within ``max_dist_vtx`` of the reco vertex;
    * ``n_mip``: good tracks with length > ``mip_min_len``, chi2_mu <
      ``mip_chi2mu_max`` and chi2_p > ``mip_chi2p_min`` (plane 2, per point);
    * ``n_shw``: primary showers, 0 < trackScore < ``shw_score`` with energy >
      ``shw_min_energy``;
    * ``mip_open_angle`` [deg]: between the two longest MIP tracks (NaN if < 2);
    * ``n_uncontained``: good tracks with an end less than ``contain_margin``
      inside the TPC or in the high y-z region;
    * ``n_cathode``: good tracks crossing the cathode (x = 0);
    * ``n_michel``: daughters with 0 < trackScore < ``michel_score``, visible
      energy < ``michel_max_energy`` and no daughter PFPs.
    """
    c = {**PFP_CUTS, **over}
    f = feat
    nu_d, ts = f.nu_d.to_numpy(), f.ts.to_numpy()
    good = nu_d & f.valid.to_numpy() & (ts > c['trk_score']) & (f.len.to_numpy() > c['min_len']) \
        & (f.dist.to_numpy() < c['max_dist_vtx'])
    mip = good & (f.len.to_numpy() > c['mip_min_len']) & (f.chi2mu.to_numpy() < c['mip_chi2mu_max']) \
        & (f.chi2p.to_numpy() > c['mip_chi2p_min'])
    shw = nu_d & (ts > 0) & (ts < c['shw_score']) & (f.shw_e.to_numpy() > c['shw_min_energy'])
    uncont = good & (~(f.edge.to_numpy() > c['contain_margin']) | f.highyz.to_numpy())
    cathode = good & f.cathode.to_numpy()
    michel = nu_d & (ts > 0) & (ts < c['michel_score']) & (f.vis.to_numpy() < c['michel_max_energy']) \
        & (f.nchild.to_numpy() == 0)

    flags = pd.DataFrame({'n_good': good, 'n_mip': mip, 'n_shw': shw, 'n_uncontained': uncont,
                          'n_cathode': cathode, 'n_michel': michel}, index=f.index)
    slc_levels = [i for i in range(f.index.nlevels) if i != PFP_LEVEL]
    out = flags.groupby(level=slc_levels).sum()

    # opening angle between the two longest MIP tracks
    m = f.loc[mip, ['len', 'dx', 'dy', 'dz']].sort_values('len', ascending=False, kind='stable')
    m.index = m.index.droplevel(PFP_LEVEL)
    rank = m.groupby(level=list(range(m.index.nlevels))).cumcount()
    first, second = m[rank == 0], m[rank == 1]
    cosang = (first[['dx', 'dy', 'dz']] * second[['dx', 'dy', 'dz']].reindex(first.index)).sum(axis=1, min_count=3)
    out['mip_open_angle'] = np.degrees(np.arccos(np.clip(cosang, -1, 1))).reindex(out.index)
    return out


def pfp_selection(df: pd.DataFrame, **over) -> pd.DataFrame:
    """:func:`pfp_counts` of :func:`pfp_features` (see those)."""
    return pfp_counts(pfp_features(df), **over)


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


def _pfp_overrides(thresholds):
    """The :data:`PFP_CUTS` entries of a thresholds dict (other keys, e.g. ``open_angle_max``, ignored)."""
    return {k: v for k, v in (thresholds or {}).items() if k in PFP_CUTS}


def collapse_pfp(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """One row per slice, with ``n_trk``, the PFP-selection counts and the ``long1``/``long2`` track groups.

    ``thresholds`` overrides :data:`PFP_CUTS` (see ``thresholds.read_thresholds``).
    """
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

    sel = pfp_selection(df, **_pfp_overrides(thresholds))
    slc_levels = [i for i in range(df.index.nlevels) if i != PFP_LEVEL]
    slc = df.drop(columns='pfp', level=0).groupby(level=slc_levels).head(1)
    slc.index = _slice_index(slc.index)
    n_trk = trk.groupby(level=slc_levels).size().reindex(slc.index).fillna(0)
    out = pd.concat([slc, long1.reindex(slc.index), long2.reindex(slc.index)], axis=1)
    out[('n_trk', '', '', '', '', '')] = n_trk.to_numpy()
    for k in sel.columns:
        out[(k, '', '', '', '', '')] = sel[k].reindex(out.index).to_numpy()
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


def preprocess_mcbnb(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Collapse to slices, then BNB-generator flash-time fix and timing calibration.

    ``thresholds`` as in :func:`collapse_pfp`; pass it with ``functools.partial``.
    """
    df = collapse_pfp(preprocess_mc(df), thresholds)
    df = fix_bfm_flashtime_mcbnb(df)
    return fix_timing_calibration(df, period=tc.mcbnb_period_calib, t0_offset=tc.mcbnb_offset_calib)


def preprocess_mcbnb_pfp(df: pd.DataFrame) -> pd.DataFrame:
    """As :func:`preprocess_mcbnb` but keeping one row per PFP (no collapse), for threshold tuning."""
    df = fix_bfm_flashtime_mcbnb(preprocess_mc(df))
    return fix_timing_calibration(df, period=tc.mcbnb_period_calib, t0_offset=tc.mcbnb_offset_calib)


def preprocess_mchnl(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Collapse to slices, then MeVPrtl-generator flash-time fix and timing calibration.

    ``thresholds`` as in :func:`collapse_pfp`; pass it with ``functools.partial``.
    """
    df = collapse_pfp(preprocess_mc(df), thresholds)
    df = fix_bfm_flashtime_mchnl(df)
    return fix_timing_calibration(df, period=tc.mchnl_period_calib, t0_offset=tc.mchnl_offset_calib)


def preprocess_mchnl_pfp(df: pd.DataFrame) -> pd.DataFrame:
    """As :func:`preprocess_mchnl` but keeping one row per PFP (no collapse), for threshold tuning."""
    df = fix_bfm_flashtime_mchnl(preprocess_mc(df))
    return fix_timing_calibration(df, period=tc.mchnl_period_calib, t0_offset=tc.mchnl_offset_calib)