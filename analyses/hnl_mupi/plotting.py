"""HNL mu-pi plot_var()/plot_mc_hnl() convenience wrappers.

``plot_var`` pre-fills :func:`cafpybara.core.plotting.plot_var` with this
analysis's :data:`background_categories`.

``plot_mc_hnl`` is this analysis's own version of the core function, one
figure per HNL mass: the BNB stack and its MC-stat band come from
``core.plot_var``, and the HNL is drawn as an unfilled step line. (Core's
``plot_mc_hnl`` passes ``hist_filled=False``/``error_legend=False`` to
``plot_var``, which ignores both, so its HNL is drawn as a translucent fill
over the stack with a second "MC stat." legend entry.)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ...core import plotting as _core_plotting
from ...core.physics import pdg_categories as _pdg_categories, mode_categories as _mode_categories
from ...core.utils import get_hist1d
from .analysis import background_categories, hnl_categories, signal_dict

__all__ = ['annotate_sbnd', 'plot_var', 'plot_mc_hnl', 'valid_rows', 'auto_bins']

annotate_sbnd = _core_plotting.annotate_sbnd


def valid_rows(df, var):
    """Rows of ``df`` where ``var`` is finite and not a cafpyana -999/-9999 sentinel."""
    v = df[var].astype(float)
    return df[np.isfinite(v) & (v != -999) & (v != -9999)]


def auto_bins(frames, var, nb=50, pct=(0.5, 99.5)):
    """``nb`` equal bins spanning the ``pct`` percentiles of ``var`` over all ``frames``.

    The selection notebook's rule for variables without fixed bins. Returns
    None when no frame has a valid entry.
    """
    v = np.concatenate([valid_rows(df, var)[var].to_numpy(float) for df in frames])
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, pct)
    return np.linspace(lo, hi, nb + 1) if hi > lo else np.linspace(lo - 1, hi + 1, nb + 1)


def _inject_defaults(kwargs):
    kwargs.setdefault('categories', None)
    if kwargs['categories'] is None and not kwargs.get('pdg') and not kwargs.get('mode'):
        kwargs['categories'] = background_categories
    kwargs.setdefault('pdg_categories', _pdg_categories)
    kwargs.setdefault('mode_categories', _mode_categories)
    kwargs.setdefault('signal_dict', signal_dict)
    return kwargs


def plot_var(df, var, bins, ax=None, config=None, **kwargs):
    """Stacked histogram, defaulting to :data:`background_categories`.

    See :func:`cafpybara.core.plotting.plot_var` for the full parameter list.
    """
    kwargs = _inject_defaults(kwargs)
    return _core_plotting.plot_var(df, var, bins, ax=ax, config=config, **kwargs)


def plot_mc_hnl(mc_df, hnl_df, var, bins, scale_nu=1.0, scale_hnl=1.0, hnl_label="HNL",
                log_y=False, show_fom=False, fom_nsigma=1.0, figsize=(7, 5), savefig="", **kwargs):
    """BNB stack (with MC-stat band) + HNL step line, for one HNL mass.

    ``mc_df``/``hnl_df`` are slice frames; pass ``hnl_df`` already restricted to
    the HNL-matched slices. Rows where ``var`` is non-finite or a -999/-9999
    sentinel are dropped from both. ``scale_nu`` scales the BNB stack (POT
    weight), ``scale_hnl`` the HNL line; ``scale_hnl="area"`` scales it to the
    stack's in-range total. The HNL histogram is weighted by ``weights_mc``
    when the frame has it (the MeVPrtl rate weights from ``load_mchnl``), so
    ``scale_hnl = target_pot / hnl_pot`` draws the expected rate at the
    simulated mixing; multiply by a display factor to enlarge it. Other
    keywords go to :func:`plot_var` for the stack (``xlabel``, ``counts``,
    ``overflow``, ``cut_val``, ``legend_kwargs``, ...).

    ``show_fom=True`` adds a Punzi figure-of-merit strip,
    ``eff_S / (a/2 + sqrt(B))`` with ``a = fom_nsigma``, for "keep > x" and
    "keep < x" at each bin edge. ``eff_S`` is the (rate-weighted) HNL
    efficiency, so the curve does not depend on ``scale_hnl``.

    Returns ``(fig, ax, mc_dict, hnl_counts)``, or
    ``(fig, ax, ax_fom, mc_dict, hnl_counts)`` with ``show_fom``.
    """
    if show_fom:
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 2))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax_fom = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_fom = None

    kwargs = _inject_defaults(kwargs)
    overflow = kwargs.get('overflow', True)
    mc_df, hnl_df = valid_rows(mc_df, var), valid_rows(hnl_df, var)
    n_before = len(ax.collections)
    _, mc_steps, mc_err, mc_dict = _core_plotting.plot_var(mc_df, var, bins, ax=ax, scale=scale_nu, **kwargs)
    # core.plot_var fills each category at alpha 0.4; redraw them opaque with thin
    # black edges (the selection-notebook look). The fills are the first new
    # collections, one per category, in category order; the stat band follows.
    cats = kwargs['categories'] or {}
    for coll, entry in zip(ax.collections[n_before:], cats.values()):
        coll.set_facecolor(entry['color'])
        coll.set_edgecolor('black')
        coll.set_linewidth(0.4)

    w_hnl = hnl_df['weights_mc'].to_numpy(float).ravel() if 'weights_mc' in hnl_df.columns else None
    h_raw = get_hist1d(data=hnl_df[var], bins=bins, weights=w_hnl, overflow=overflow)
    area = isinstance(scale_hnl, str) and scale_hnl == "area"
    if area:
        scale_hnl = mc_steps[-1][1:].sum() / h_raw.sum() if h_raw.sum() else 1.0
    h = h_raw * scale_hnl
    # Area-normalised: the raw slice count (the scaled sum is meaningless).
    # Otherwise the drawn sum, like the stack's category counts.
    n_label = f"N={len(hnl_df):,}" if area else f"{h.sum():,.1f}"
    label = hnl_label + (f" ({n_label})" if kwargs.get('counts') else "")
    ax.stairs(h, bins, color=hnl_categories["hnl"]["color"], lw=2, label=label, zorder=1e3)

    # Legend outside the axes, to the right, so it never covers the histograms.
    legend_kwargs = {'ncol': 1, 'loc': 'upper left', 'bbox_to_anchor': (1.01, 1.0), 'frameon': False,
                     **(kwargs.get('legend_kwargs') or {})}
    # Categories listed top-down in the same order as the stack, then stat band and HNL.
    handles, labels = ax.get_legend_handles_labels()
    order = list(range(len(cats)))[::-1] + list(range(len(cats), len(handles)))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], **legend_kwargs).set_zorder(1e3 + 1)
    if log_y:
        ax.set_yscale('log')

    if show_fom:
        b = mc_steps[-1][1:]
        n_s = h_raw.sum() if h_raw.sum() > 0 else 1.0
        a = fom_nsigma
        edges = np.asarray(bins)[:-1]
        with np.errstate(invalid='ignore', divide='ignore'):
            fom_gt = (np.cumsum(h_raw[::-1])[::-1] / n_s) / (a / 2 + np.sqrt(np.cumsum(b[::-1])[::-1]))
            fom_lt = (np.cumsum(h_raw) / n_s) / (a / 2 + np.sqrt(np.cumsum(b)))
        hi = np.asarray(bins)[1:]
        best_gt, best_lt = edges[np.nanargmax(fom_gt)], hi[np.nanargmax(fom_lt)]
        ax_fom.plot(edges, fom_gt, color='steelblue', label=f'keep > x (best {best_gt:.3g})')
        ax_fom.plot(hi, fom_lt, color='darkorange', label=f'keep < x (best {best_lt:.3g})')
        ax_fom.axvline(best_gt, color='steelblue', ls='--', lw=1, alpha=0.7)
        ax_fom.axvline(best_lt, color='darkorange', ls='--', lw=1, alpha=0.7)
        ax_fom.set_ylabel(f'Punzi FOM\n(a={a:g})')
        ax_fom.set_xlabel(ax.get_xlabel())
        ax_fom.legend(fontsize='small', frameon=False)
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
        for cut in kwargs.get('cut_val') or []:
            ax_fom.axvline(cut, color='black', ls='--', lw=2, alpha=0.5)

    mc_dict['counts'] = mc_steps[-1][1:]
    mc_dict['total_err'] = mc_err
    if savefig:
        fig.savefig(savefig, bbox_inches='tight')
    if show_fom:
        return fig, ax, ax_fom, mc_dict, h
    return fig, ax, mc_dict, h
