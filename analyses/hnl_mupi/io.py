"""HNL mu-pi loaders: ``load_mc`` (BNB nu + cosmics) and ``load_mchnl`` (MeVPrtl HNL).

Both pre-fill :func:`cafpybara.core.io.load_mc` with ``rec_key='rec'``, this
analysis's preprocessing (PFP -> slice collapse, timing) and truth
categorisation.

``max_events`` keeps only the first N events (``(__ntuple, entry)`` pairs) of
each split before preprocessing, for quick layout checks. The returned POT is
then scaled by the kept fraction of header events, which is only approximate:
use it for layout, not for numbers.
"""
from __future__ import annotations

from ...core.io import load_mc as _core_load_mc, load_data as _core_load_data
from .analysis import define_signal, define_signal_hnl, signal_dict
from .preprocess import preprocess_mcbnb, preprocess_mchnl

__all__ = ['load_mc', 'load_mchnl', 'load_data']

REC_KEY = "rec"


def _first_events(preprocess_fn, max_events):
    def fn(df):
        events = df.index.droplevel(list(range(2, df.index.nlevels))).unique()[:max_events]
        keep = df.index.droplevel(list(range(2, df.index.nlevels))).isin(events)
        return preprocess_fn(df[keep])
    return fn


def _load(file, preprocess_fn, define_signal_fn, cuts, max_events, **kwargs):
    if max_events is None:
        return _core_load_mc(file, rec_key=REC_KEY, preprocess_fn=preprocess_fn,
                             define_signal_fn=define_signal_fn, cuts=cuts, **kwargs)
    from ...core.io import load_dfs, get_n_split
    df, pot, ngen = _core_load_mc(file, rec_key=REC_KEY,
                                  preprocess_fn=_first_events(preprocess_fn, max_events),
                                  define_signal_fn=define_signal_fn, cuts=cuts, **kwargs)
    hdr = load_dfs(file, ['hdr'], n_max_concat=get_n_split(file))['hdr']
    frac = min(1.0, max_events * get_n_split(file) / len(hdr))
    return df, pot * frac, ngen * frac


def load_mc(file: str, cuts=None, max_events: int | None = None,
            preprocess_fn=preprocess_mcbnb, define_signal_fn=define_signal, **kwargs):
    """Load a BNB nu + cosmics ``*_savepfp`` frame. See :func:`cafpybara.core.io.load_mc`."""
    return _load(file, preprocess_fn, define_signal_fn, cuts, max_events, **kwargs)


def load_mchnl(file: str, cuts=None, max_events: int | None = None,
               preprocess_fn=preprocess_mchnl, define_signal_fn=define_signal_hnl, **kwargs):
    """Load a MeVPrtl HNL ``*_savepfp`` frame. See :func:`cafpybara.core.io.load_mc`."""
    return _load(file, preprocess_fn, define_signal_fn, cuts, max_events, **kwargs)


def load_data(file: str, onbeam: bool = True, cuts=None, preprocess_fn=None, **kwargs):
    """Load a data frame. TODO: no data or off-beam frames for this topology yet."""
    return _core_load_data(file, rec_key=REC_KEY, preprocess_fn=preprocess_fn, onbeam=onbeam,
                           cuts=cuts, offbeam_signal_value=signal_dict["offbeam"], **kwargs)
