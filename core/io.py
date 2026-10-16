"""File input/output utilities for loading HDF5 data files.

``load_mc``/``load_data`` here take no topology default for ``rec_key``,
``preprocess_fn``, or ``define_signal_fn``/``offbeam_signal_value`` (nor does
``load_mevprtl``) -- every
analysis supplies its own via a thin wrapper in
``cafpybara/analyses/<topology>/io.py`` (e.g. nueCC's wrapper defaults
``rec_key='nuecc'``; HNL/pi0's defaults ``rec_key='rec'``).
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

__all__ = ['xrootd_url', 'ensure_token', 'open_store', 'get_n_split', 'print_keys', 'load_dfs', 'load_mc', 'load_data',
           'correct_cosmic_weight_mevprtl_df', 'load_mevprtl']

_UNSET = object()

def xrootd_url(path):
    """``root://`` URL of a dCache file given by a ``/pnfs/sbnd/...`` path or a symlink into it
    (e.g. ``~/persistent/...``; resolved without needing ``/pnfs`` to be mounted)."""
    import os
    real = os.path.realpath(path)
    if not real.startswith('/pnfs/sbnd/'):
        raise ValueError(f"{path} does not resolve to /pnfs/sbnd/... ({real})")
    return 'root://fndcadoor.fnal.gov:1094//pnfs/fnal.gov/usr/sbnd/' + real[len('/pnfs/sbnd/'):]


def ensure_token(min_hours: float = 1.0, path=None):
    """Make sure ``BEARER_TOKEN_FILE`` holds an SBND token valid for ``min_hours``; else ask for one.

    Runs ``htgettoken -a htvaultprod.fnal.gov -i sbnd`` and prints its output as it
    comes, including the verification link to open in a browser; blocks until
    it is approved. The token stays in the token file, nothing else is stored.
    """
    import base64, json, os, subprocess, time
    path = path or os.environ.get('BEARER_TOKEN_FILE') or f'/tmp/bt_u{os.getuid()}'
    os.environ['BEARER_TOKEN_FILE'] = path

    def hours_left():
        try:
            body = open(path).read().strip().split('.')[1]
            return (json.loads(base64.urlsafe_b64decode(body + '=' * (-len(body) % 4)))['exp'] - time.time()) / 3600
        except Exception:
            return -1.0

    if hours_left() >= min_hours:
        print(f"token ok: {hours_left():.1f} h left ({path})")
        return path
    print(f"no valid token at {path}: requesting one; open the link below")
    import shlex, shutil, sys
    cmd = ['htgettoken', '-a', 'htvaultprod.fnal.gov', '-i', 'sbnd', '-o', path]
    if not sys.stdin.isatty() and shutil.which('script'):
        # htgettoken skips the web login ("running in the background") without a terminal
        cmd = ['script', '-qfc', shlex.join(cmd), '/dev/null']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True)
    for line in proc.stdout:
        print(line, end='', flush=True)
    if proc.wait() != 0 or hours_left() < min_hours:
        raise RuntimeError("htgettoken failed")
    print(f"token ok: {hours_left():.1f} h left")
    return path


def open_store(file, mode='r'):
    """``pd.HDFStore`` for a local path, or for an xrootd URL (``root://...``).

    A URL is streamed with fsspec into memory (``H5FD_CORE`` image, nothing is
    written to disk), so frames in dCache can be read where ``/pnfs`` is not
    mounted (e.g. JupyterHub). Needs an SBND bearer token (``BEARER_TOKEN_FILE``)
    and the whole file fits in memory.
    """
    if not str(file).startswith('root://'):
        return pd.HDFStore(file, mode=mode)
    import fsspec
    with fsspec.open(file, 'rb') as f:
        image = f.read()
    return pd.HDFStore(file.rsplit('/', 1)[-1], mode='r', driver='H5FD_CORE',
                       driver_core_image=image, driver_core_backing_store=0)


def get_n_split(file):
    """Get the number of splits in an HDF5 file.

    Parameters
    ----------
    file : str
        Path to HDF5 file.

    Returns
    -------
    int
        Number of splits in the file.
    """
    with open_store(file) as store:
        return store['split'].n_split.iloc[0]

def print_keys(file):
    """Print all keys available in an HDF5 file.

    Parameters
    ----------
    file : str
        Path to HDF5 file.
    """
    with open_store(file) as store:
        keys = store.keys()
        print("Keys:", keys)

def load_dfs(file, keys2load, n_max_concat=10, start_split=0):
    """Load DataFrames from split HDF5 file.

    Parameters
    ----------
    file : str
        Path to HDF5 file.
    keys2load : list
        List of key names to load from the file.
    n_max_concat : int, optional
        Maximum number of splits to concatenate (default: 10).
    start_split : int, optional
        Starting split index to load from (default: 0).

    Returns
    -------
    dict
        Dictionary mapping key names to concatenated DataFrames.
    """
    out_df_dict = {}
    with open_store(file) as store:
        this_n_keys = store['split'].n_split.iloc[0] - start_split
        n_concat = min(n_max_concat, this_n_keys)
        for key in keys2load:
            dfs = [store[f'{key}_{i}'] for i in range(start_split, start_split + n_concat)]
            out_df_dict[key] = pd.concat(dfs, ignore_index=False)
    return out_df_dict


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

def load_mc(
    file: str,
    rec_key: str,
    preprocess_fn,
    define_signal_fn,
    keys: list | None = None,
    cuts=None,
    max_splits: int | None = None,
    chunk_splits: int = 1,
    add_pi0_fn=None,
    excl_mc_df=None,
    remove_signal_overlap_fn=None,
) -> tuple:
    """Load, preprocess, and optionally select an MC HDF5 file in chunks.

    Splits are loaded in batches of chunk_splits to balance memory and I/O
    overhead.  POT and generated-event counts are accumulated across all
    splits.  Header columns (run/subrun/event) are merged into the output
    DataFrame.

    Parameters
    ----------
    file : str
        Path to the HDF5 file.
    rec_key : str
        Key of the main slc-level table within ``keys``/the HDF5 file (e.g.
        ``'nuecc'`` or ``'rec'``). Required -- no topology default here.
    preprocess_fn : callable or None
        Called as ``preprocess_fn(df)`` on each chunk's ``rec_key`` table
        before selection. Pass ``None`` to skip preprocessing entirely.
        Required (explicitly, even if None) -- no topology default here.
    define_signal_fn : callable
        Called as ``define_signal_fn(df)`` on each merged chunk to stamp
        signal categories (e.g.
        ``functools.partial(nuecc.analysis.define_signal, prefix=('slc', 'truth'))``
        or ``hnlpi0.selection.define_signal_hnl``).
        Required -- no topology default here.
    keys : list of str, optional
        Table keys to load.  Defaults to
        ``['hdr', rec_key, 'histpotdf', 'histgenevtdf']``.
    cuts : list of CutSpec, optional
        If supplied, passed to :func:`~cafpybara.core.selection.select`.
        When None the full preprocessed DataFrame is returned.
    max_splits : int, optional
        Cap on the number of splits to load.  Defaults to all splits.
    chunk_splits : int, default 1
        Number of splits to load per iteration.  Increase to reduce I/O
        overhead at the cost of higher peak memory per chunk.
    add_pi0_fn : callable, optional
        If given, called as ``add_pi0_fn(df)`` on each chunk after preprocessing
        to compute pi0 kinematics.
    excl_mc_df : pd.DataFrame, optional
        Exclusive DataFrame with signal categorisation already applied (i.e.
        has a top-level ``signal`` column).  When provided,
        ``remove_signal_overlap_fn`` is called on the final concatenated
        result to strip events that are already covered by the exclusive
        sample and would otherwise be double-counted.
    remove_signal_overlap_fn : callable, optional
        Called as ``remove_signal_overlap_fn(result, excl_mc_df)`` when
        ``excl_mc_df`` is provided. Required if ``excl_mc_df`` is not None.

    Returns
    -------
    df : pd.DataFrame
        Concatenated, preprocessed (and optionally selected) MC DataFrame
        with header columns merged in and signal categories defined.
    pot : float
        Accumulated POT.
    ngen : float
        Accumulated generated-event count.
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    from .selection import select
    from .utils import merge_hdr

    if keys is None:
        keys = ['hdr', rec_key, 'histpotdf', 'histgenevtdf']

    n_total  = get_n_split(file)
    n_splits = min(max_splits, n_total) if max_splits is not None else n_total
    starts   = range(0, n_splits, chunk_splits)
    iterator = _tqdm(starts) if _tqdm is not None else starts

    pot    = 0.0
    ngen   = 0.0
    chunks = []

    for i in iterator:
        n_load = min(chunk_splits, n_splits - i)
        dfs = load_dfs(file, keys2load=keys, n_max_concat=n_load, start_split=i)

        if 'histpotdf' in dfs:    pot  += dfs['histpotdf'].TotalPOT.sum()
        elif 'hdr' in dfs:        pot  += dfs['hdr'].pot.sum()
        if 'histgenevtdf' in dfs: ngen += dfs['histgenevtdf'].TotalGenEvents.sum()
        elif 'hdr' in dfs:        ngen += dfs['hdr'][dfs['hdr'].first_in_subrun == 1].ngenevt.sum()

        df    = preprocess_fn(dfs[rec_key]) if preprocess_fn is not None else dfs[rec_key]
        sel   = select(df, cuts=cuts) if cuts is not None else df
        chunk = merge_hdr(dfs['hdr'], sel)
        del dfs
        chunk = define_signal_fn(chunk)
        if add_pi0_fn is not None:
            chunk = add_pi0_fn(chunk)
        chunks.append(chunk)
        del chunk
        gc.collect()

    result = pd.concat(chunks, ignore_index=False).copy()
    if excl_mc_df is not None:
        if remove_signal_overlap_fn is None:
            raise ValueError("load_mc: excl_mc_df was given but remove_signal_overlap_fn is None")
        result = remove_signal_overlap_fn(result, excl_mc_df)
    return result, pot, ngen


def load_data(
    file: str,
    rec_key: str,
    preprocess_fn,
    keys: list | None = None,
    onbeam: bool = True,
    cuts=None,
    offbeam_signal_value=_UNSET,
) -> tuple:
    """Load, preprocess, and optionally select a data HDF5 file.

    Parameters
    ----------
    file : str
        Path to the HDF5 file.
    rec_key : str
        Key of the main slc-level table within ``keys``/the HDF5 file (e.g.
        ``'nuecc'`` or ``'rec'``). Required -- no topology default here.
    preprocess_fn : callable or None
        Called as ``preprocess_fn(df)`` on the merged table before selection.
        Pass ``None`` to skip preprocessing entirely. Required (explicitly,
        even if None) -- no topology default here.
    keys : list of str, optional
        Table keys to load.  Defaults to ``['hdr', rec_key, 'histpotdf']``.
    onbeam : bool, default True
        True for on-beam (BNB) data; False for off-beam.  Controls which
        gate counter is returned and whether the offbeam signal category is
        stamped on the output DataFrame.
    cuts : list of CutSpec, optional
        If supplied, passed to :func:`~cafpybara.core.selection.select`.
        When None the full preprocessed DataFrame is returned.
    offbeam_signal_value : int, required if onbeam=False
        Signal value stamped on off-beam rows when ``onbeam=False`` (e.g.
        ``nuecc.analysis.signal_dict['offbeam']`` or
        ``hnlpi0.analysis.signal_dict_hnl['offbeam']``). No topology default.

    Returns
    -------
    df : pd.DataFrame
        Preprocessed (and optionally selected) data DataFrame with header
        columns merged in.  Off-beam DataFrames also have ``signal`` set to
        ``offbeam_signal_value``.
    pot : float
        Accumulated on-beam POT (0.0 for off-beam files).
    ngates : float
        BNB gate count (on-beam) or off-beam gate count.
    """
    from pyanalib.pandas_helpers import multicol_add
    from .selection import select
    from .utils import merge_hdr

    if keys is None:
        keys = ['hdr', rec_key, 'histpotdf']
    if not onbeam and offbeam_signal_value is _UNSET:
        raise ValueError("load_data: offbeam_signal_value is required when onbeam=False")

    dfs = load_dfs(file, keys2load=keys)
    df  = merge_hdr(dfs['hdr'], dfs[rec_key])
    df  = preprocess_fn(df) if preprocess_fn is not None else df

    pot    = 0.0
    ngates = 0.0
    if onbeam:
        pot    = dfs['histpotdf'].TotalPOT.sum() if 'histpotdf' in dfs else dfs['hdr'].pot.sum()
        ngates = dfs['hdr'].nbnbinfo.sum()
    else:
        ngates = dfs['hdr'].noffbeambnb.sum()
        signal = pd.Series(
            np.ones(len(df), dtype=np.int16) * offbeam_signal_value,
            name="signal", index=df.index,
        )
        df = multicol_add(df, signal)

    sel = select(df, cuts=cuts) if cuts is not None else df
    return sel, pot, ngates


# ---------------------------------------------------------------------------
# MeVPrtl (e.g. HNL) generator samples
# ---------------------------------------------------------------------------

def correct_cosmic_weight_mevprtl_df(indf_rec, indf_truth, indf_hdr):
    """Correct the cosmic-background event weight in an HNL MeVPrtl reco DataFrame
    using the matching truth-level total weight.

    Reco and truth don't share a common natural row index -- their sub-object index
    levels differ (``rec.slc..index`` vs. ``rec.mc.prtl..index``) -- so they're joined
    via a composite key built from ``evt`` (looked up per-row from the header table)
    instead of the natural index.

    When present, ``file_idx`` is included in that composite key alongside
    ``(__ntuple, entry, evt)``. This matters for any file produced by
    ``concat_hdf.py``: it resets ``__ntuple``/``entry`` numbering independently per
    source shard and adds a ``file_idx`` level specifically to disambiguate the
    result, so omitting ``file_idx`` from the join key causes rows from different
    shards to collide on the same ``(__ntuple, entry, evt)`` triple -- ``reindex``
    then raises ``ValueError: cannot handle a non-unique multi-index`` . Falls back to the
    ``(__ntuple, entry, evt)``-only key when ``file_idx`` isn't present on either
    side, matching the original behaviour for a raw, non-concatenated single-shard
    file.

    Parameters
    ----------
    indf_rec : pd.DataFrame
        HNL reco (slice-level) DataFrame.
    indf_truth : pd.DataFrame
        HNL MeVPrtl truth DataFrame.
    indf_hdr : pd.DataFrame
        Header DataFrame, used to look up ``evt`` for both ``indf_rec`` and
        ``indf_truth``.
    """


    fluxw_column = ('slc', 'prtl', 'flux_weight', '', '', '')
    rayw_column = ('slc', 'prtl', 'ray_weight', '', '', '')
    decayw_column = ('slc', 'prtl', 'decay_weight', '', '', '')
    totalw_column = ('weights_mc', '', '', '', '', '')
    indf_rec[totalw_column] = indf_rec[rayw_column] * indf_rec[decayw_column] * indf_rec[fluxw_column]

    truth_fluxw_column = ('flux_weight', '')
    truth_rayw_column = ('ray_weight', '')
    truth_decayw_column = ('decay_weight', '')
    truth_totalw_column = ('weights_mc_truth', '')
    indf_truth[truth_totalw_column] = indf_truth[truth_rayw_column] * indf_truth[truth_decayw_column] * indf_truth[truth_fluxw_column]

    evt_col = ('evt', '', '', '', '', '')
    indf_rec[evt_col] = indf_hdr['evt'].reindex(indf_rec.index)

    totalw_truth_col = ('weights_mc_truth', '', '', '', '', '')

    truth_total_weight = indf_truth[truth_totalw_column]
    truth_evt = indf_hdr['evt'].reindex(indf_truth.index).values

    join_levels = ['__ntuple', 'entry']
    if 'file_idx' in indf_truth.index.names and 'file_idx' in indf_rec.index.names:
        join_levels = ['file_idx'] + join_levels
    join_names = join_levels + ['evt']

    truth_key = pd.MultiIndex.from_arrays(
        [indf_truth.index.get_level_values(lvl) for lvl in join_levels] + [truth_evt],
        names=join_names,
    )
    truth_lookup = pd.Series(truth_total_weight.values, index=truth_key)

    reco_key = pd.MultiIndex.from_arrays(
        [indf_rec.index.get_level_values(lvl) for lvl in join_levels] + [indf_rec[evt_col].values],
        names=join_names,
    )

    indf_rec[totalw_truth_col] = reco_key.map(truth_lookup)


    mask_cosmic = np.isnan(indf_rec[('slc', 'prtl', 'E', '', '', '')])
    indf_rec.loc[mask_cosmic, totalw_column] = indf_rec.loc[mask_cosmic, truth_totalw_column]

    indf_rec = indf_rec.drop(columns=[evt_col])

    return indf_rec


def load_mevprtl(
    file: str,
    rec_key: str,
    preprocess_fn,
    define_signal_fn,
    keys: list | None = None,
    mevprtl_key: str = 'mevprtl',
    cuts=None,
) -> tuple:
    """Load, preprocess, and optionally select a MeVPrtl-generator MC HDF5 file.

    Applies the MeVPrtl cosmic-weight correction
    (:func:`correct_cosmic_weight_mevprtl_df`), stamps signal categories, and
    merges header columns in. Also extracts the simulated mixing angle and
    particle mass from the truth table.

    Parameters
    ----------
    file : str
        Path to the HDF5 file.
    rec_key : str
        Key of the main slc-level table. Required -- no topology default here.
    preprocess_fn : callable or None
        Called as ``preprocess_fn(df)`` on the raw ``rec_key`` table before the
        cosmic-weight correction. Pass ``None`` to skip. Required (explicitly,
        even if None) -- no topology default here.
    define_signal_fn : callable
        Called as ``define_signal_fn(df)`` to stamp signal categories.
        Required -- no topology default here.
    keys : list of str, optional
        Table keys to load, not including ``mevprtl_key`` (added automatically
        if missing). Defaults to ``['hdr', rec_key, 'histpotdf']``.
    mevprtl_key : str, default 'mevprtl'
        Key of the MeVPrtl truth table.
    cuts : list of CutSpec, optional
        If supplied, passed to :func:`~cafpybara.core.selection.select` as the
        last step, after ``simU``/``hnlM`` are extracted from the full
        (unselected) DataFrame -- selection can otherwise remove every
        truth-matched row and make that extraction raise. When None the full
        preprocessed DataFrame is returned.

    Returns
    -------
    df : pd.DataFrame
        Preprocessed (and optionally selected) DataFrame with header columns
        merged in and signal categories defined.
    pot : float
        MC POT (unscaled).
    info : dict
        ``{'simU': ..., 'hnlM': ...}`` -- simulated mixing angle squared and
        mass in MeV (converted from the truth table's GeV).
    """
    from .selection import select
    from .utils import merge_hdr

    if keys is None:
        keys = ['hdr', rec_key, 'histpotdf']
    load_keys = keys if mevprtl_key in keys else keys + [mevprtl_key]

    dfs = load_dfs(file, keys2load=load_keys, n_max_concat=get_n_split(file))
    rec_df = preprocess_fn(dfs[rec_key]) if preprocess_fn is not None else dfs[rec_key]
    df  = correct_cosmic_weight_mevprtl_df(rec_df, dfs[mevprtl_key], dfs['hdr'])
    df  = define_signal_fn(df)
    df  = merge_hdr(dfs['hdr'], df)

    pot  = dfs['histpotdf'].TotalPOT.sum() if 'histpotdf' in dfs else dfs['hdr'].pot.sum()
    simU = df[('slc', 'prtl', 'C2', '', '', '')].dropna().unique()[0]
    hnlM = df[('slc', 'prtl', 'M', '', '', '')].dropna().unique()[0] * 1000

    sel = select(df, cuts=cuts) if cuts is not None else df
    return sel, pot, {'simU': simU, 'hnlM': hnlM}
