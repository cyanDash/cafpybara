"""HNL mu-pi loaders: ``load_mc`` (BNB nu + cosmics) and ``load_mchnl`` (MeVPrtl HNL).

Both pre-fill :func:`cafpybara.core.io.load_mc` with ``rec_key='rec'``, this
analysis's preprocessing (PFP -> slice collapse, timing) and truth
categorisation.

``max_events`` keeps only the first N events (``(__ntuple, entry)`` pairs) of
each split before preprocessing, for quick layout checks. For a one-split file
the returned POT is the header POT up to the last kept event; otherwise it is
scaled by the kept fraction of header events, which is only approximate: use
it for layout, not for numbers.

``load_mchnl`` pre-fills :func:`cafpybara.core.io.load_mevprtl` (as hnlpi0
does) and returns ``(df, pot, info)`` with ``info = {'simU', 'hnlM'}``. Core
adds ``weights_mc`` = flux x ray x decay weight of the slice's HNL, falling back
to the event's ``mevprtl`` truth weight for slices without one (cosmic slices
in cosmic-overlay samples). ``sum(weights_mc) * pot_target / pot`` is the
expected number of slices at ``pot_target`` and ``info['simU']``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.io import (load_mc as _core_load_mc, load_data as _core_load_data,
                        load_mevprtl as _core_load_mevprtl, load_dfs, get_n_split)
from .analysis import define_signal, define_signal_hnl, signal_dict
from .preprocess import preprocess_mcbnb, preprocess_mchnl

__all__ = ['load_mc', 'load_mchnl', 'load_data']

REC_KEY = "rec"
MEVPRTL_KEY = "mevprtl"


def _first_events(preprocess_fn, max_events, last_kept):
    def fn(df):
        events = df.index.droplevel(list(range(2, df.index.nlevels))).unique()[:max_events]
        keep = df.index.droplevel(list(range(2, df.index.nlevels))).isin(events)
        last_kept.append(events.max())
        return preprocess_fn(df[keep])
    return fn


def _kept_fraction(file, max_events, last_kept):
    """POT fraction of the first ``max_events`` events (see module docstring)."""
    hdr = load_dfs(file, ['hdr'], n_max_concat=get_n_split(file))['hdr']
    if get_n_split(file) == 1 and hdr.pot.sum() > 0:
        # The rec table only holds events with a preselected slice, so count
        # the header POT of every event up to the last kept one.
        ev = hdr.index.droplevel(list(range(2, hdr.index.nlevels))) if hdr.index.nlevels > 2 else hdr.index
        upto = np.array([e <= last_kept[0] for e in ev])
        return hdr.pot[upto].sum() / hdr.pot.sum()
    return min(1.0, max_events * get_n_split(file) / len(hdr))


def _load(file, preprocess_fn, define_signal_fn, cuts, max_events, **kwargs):
    if max_events is None:
        return _core_load_mc(file, rec_key=REC_KEY, preprocess_fn=preprocess_fn,
                             define_signal_fn=define_signal_fn, cuts=cuts, **kwargs)
    last_kept = []
    df, pot, ngen = _core_load_mc(file, rec_key=REC_KEY,
                                  preprocess_fn=_first_events(preprocess_fn, max_events, last_kept),
                                  define_signal_fn=define_signal_fn, cuts=cuts, **kwargs)
    frac = _kept_fraction(file, max_events, last_kept)
    return df, pot * frac, ngen * frac


def load_mc(file: str, cuts=None, max_events: int | None = None,
            preprocess_fn=preprocess_mcbnb, define_signal_fn=define_signal, **kwargs):
    """Load a BNB nu + cosmics ``*_savepfp`` frame. See :func:`cafpybara.core.io.load_mc`."""
    return _load(file, preprocess_fn, define_signal_fn, cuts, max_events, **kwargs)


def load_mchnl(file: str, cuts=None, max_events: int | None = None,
               preprocess_fn=preprocess_mchnl, define_signal_fn=define_signal_hnl, **kwargs):
    """Load a MeVPrtl HNL ``*_savepfp`` frame. See :func:`cafpybara.core.io.load_mevprtl`.

    Returns ``(df, pot, info)``: ``pot`` is the MeVPrtl generator POT
    (``histpotdf``), to be used with ``weights_mc``; ``info`` is
    ``{'simU', 'hnlM'}``.
    """
    if max_events is None:
        return _core_load_mevprtl(file, rec_key=REC_KEY, preprocess_fn=preprocess_fn,
                                  define_signal_fn=define_signal_fn, mevprtl_key=MEVPRTL_KEY,
                                  cuts=cuts, **kwargs)
    last_kept = []
    df, pot, info = _core_load_mevprtl(file, rec_key=REC_KEY,
                                       preprocess_fn=_first_events(preprocess_fn, max_events, last_kept),
                                       define_signal_fn=define_signal_fn, mevprtl_key=MEVPRTL_KEY,
                                       cuts=cuts, **kwargs)
    return df, pot * _kept_fraction(file, max_events, last_kept), info


def load_data(file: str, onbeam: bool = True, cuts=None, preprocess_fn=None, **kwargs):
    """Load a data frame. TODO: no data or off-beam frames for this topology yet."""
    return _core_load_data(file, rec_key=REC_KEY, preprocess_fn=preprocess_fn, onbeam=onbeam,
                           cuts=cuts, offbeam_signal_value=signal_dict["offbeam"], **kwargs)
