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
from .analysis import background_categories, hnl_categories, signal_categories, signal_dict

__all__ = ['annotate_sbnd', 'plot_var', 'plot_mc_hnl', 'plot_mc_hnl_data', 'valid_rows', 'auto_bins',
           'stack_categories', 'plot_cut_progression']

# BNB categories plus the off-beam data that completes the cosmic prediction.
stack_categories = {**background_categories, 'offbeam': signal_categories['offbeam']}

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


def plot_mc_hnl_data(mc_df, hnl_df, data_df, var, bins, scale_nu=1.0, scale_hnl=1.0, hnl_label="HNL",
                     log_y=False, figsize=(7, 6), ratio_range=(0.0, 2.0), savefig="", **kwargs):
    """Prediction stack (BNB + off-beam) + data points + HNL step line, with a Data/Pred panel.

    ``mc_df`` is the BNB slice frame concatenated with the off-beam data frame (``signal`` =
    ``signal_dict['offbeam']``, ``weights_mc`` = the off-beam scale to the MC POT); ``scale_nu``
    (data POT / BNB MC POT) scales the whole stack. ``hnl_df`` is the HNL-matched slices, drawn
    weighted by ``weights_mc`` times ``scale_hnl`` as in :func:`plot_mc_hnl`; ``data_df`` is the
    on-beam data. Rows where ``var`` is non-finite or a -999/-9999 sentinel are dropped from all
    three. Other keywords go to :func:`plot_var` for the stack. The ratio uses the stack's
    statistical error band; the annotation is the integrated Data/Pred with its statistical error.

    Returns ``(fig, ax, ax_ratio, mc_dict, hnl_counts, data_counts)``.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)

    kwargs.setdefault('categories', stack_categories)
    kwargs = _inject_defaults(kwargs)
    overflow = kwargs.get('overflow', True)
    mc_df, hnl_df, data_df = valid_rows(mc_df, var), valid_rows(hnl_df, var), valid_rows(data_df, var)
    n_before = len(ax.collections)
    _, mc_steps, mc_err, mc_dict = _core_plotting.plot_var(mc_df, var, bins, ax=ax, scale=scale_nu, **kwargs)
    cats = kwargs['categories']
    for coll, entry in zip(ax.collections[n_before:], cats.values()):
        coll.set_facecolor(entry['color'])
        coll.set_edgecolor('black')
        coll.set_linewidth(0.4)
    mc_tot = mc_steps[-1][1:]

    w_hnl = hnl_df['weights_mc'].to_numpy(float).ravel() if 'weights_mc' in hnl_df.columns else None
    h = get_hist1d(data=hnl_df[var], bins=bins, weights=w_hnl, overflow=overflow) * scale_hnl
    label = hnl_label + (f" ({h.sum():,.1f})" if kwargs.get('counts') else "")
    # Draw order, bottom to top: prediction stack, HNL line, data.
    ax.stairs(h, bins, color=hnl_categories["hnl"]["color"], lw=2, label=label, zorder=2e3)

    d = get_hist1d(data=data_df[var], bins=bins, overflow=overflow)
    centers, d_err = 0.5 * (np.asarray(bins)[1:] + np.asarray(bins)[:-1]), np.sqrt(d)
    ax.errorbar(centers, d, yerr=d_err, fmt='o', color='black', ms=5, mec='white', mew=0.6, elinewidth=1.2,
                zorder=3e3,
                label=f"Data ({int(d.sum()):,})" if kwargs.get('counts') else "Data")

    legend_kwargs = {'ncol': 1, 'loc': 'upper left', 'bbox_to_anchor': (1.01, 1.0), 'frameon': False,
                     **(kwargs.get('legend_kwargs') or {})}
    handles, labels = ax.get_legend_handles_labels()
    n_cat = len(cats)
    data_i = labels.index(next(l for l in labels if l.startswith("Data")))
    order = [data_i] + list(range(n_cat))[::-1] + [i for i in range(n_cat, len(handles)) if i != data_i]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], **legend_kwargs).set_zorder(4e3)
    if log_y:
        ax.set_yscale('log')

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio, ratio_err, band = d / mc_tot, d_err / mc_tot, mc_err / mc_tot
    ax_r.stairs(1 + band, bins, baseline=1 - band, fill=True, color='gray', alpha=0.4, lw=0)
    ax_r.errorbar(centers, ratio, yerr=ratio_err, fmt='s', ms=3, color='black', zorder=1e3)
    ax_r.axhline(1, color='red', ls='--', lw=1, zorder=0)
    ax_r.set_ylim(*ratio_range)
    ax_r.set_ylabel("Data/Pred")
    ax_r.set_xlabel(ax.get_xlabel())
    ax.set_xlabel('')
    ax.tick_params(labelbottom=False)
    tot_d, tot_mc = d.sum(), mc_tot.sum()
    if tot_mc > 0:
        ax.annotate(rf"$\Sigma$ Data/Pred = {tot_d / tot_mc:.2f} $\pm$ {np.sqrt(tot_d) / tot_mc:.2f} (stat.)",
                    xy=(0.02, 0.98), xycoords='axes fraction', va='top', fontsize=9)
    if savefig:
        fig.savefig(savefig, bbox_inches='tight')
    mc_dict['counts'] = mc_tot
    mc_dict['total_err'] = mc_err
    return fig, ax, ax_r, mc_dict, h, d


def plot_cut_progression(flow, metric, ylabel, labels=None, log_y=False, note=None, savefig=None, ax=None,
                         xlabels=None):
    """One line per signal sample of ``flow[f"{metric} {k}"]`` against the cut stage.

    ``flow`` is the frame of :func:`~cafpybara.analyses.hnl_mupi.selection.eff_purity_flow`;
    ``labels`` maps a signal key (``"m300"``) to its legend label (default: the key).
    The x axis is the cumulative cut sequence, its first point the preselection. Rows with
    ``branch = True`` (contained / uncontained split) are drawn as open diamonds forking from
    the last cut, on a shaded band.
    ``xlabels`` overrides the tick labels (default ``flow["cut"]``), e.g. labels without
    per-mass threshold values when ``flow`` joins several masses.
    ``note`` (e.g. the POT) is written in the upper-left corner. No title (slide plots).
    Returns ``(fig, ax)``.
    """
    keys = [c[len(metric) + 1:] for c in flow.columns if c.startswith(metric + " ")]
    labels = labels or {}
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    x = np.arange(len(flow))
    branch = flow["branch"].to_numpy(bool) if "branch" in flow.columns else np.zeros(len(flow), bool)
    main = np.flatnonzero(~branch)
    for i, k in enumerate(keys):
        y = flow[f"{metric} {k}"].to_numpy(float)
        col = _PROGRESSION_COLORS[i % len(_PROGRESSION_COLORS)]
        ax.plot(x[main], y[main], marker="o", ms=5, lw=1.8, color=col, label=labels.get(k, k))
        for j in np.flatnonzero(branch):        # split branches fork from the last main point
            ax.plot([x[main[-1]], x[j]], [y[main[-1]], y[j]], ls="--", lw=1.2, color=col)
            ax.plot(x[j], y[j], marker="D", ms=6, color=col, mfc="white", mew=1.5)
    if branch.any():
        ax.axvspan(x[branch].min() - 0.5, x[branch].max() + 0.5, color="0.92", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(flow["cut"] if xlabels is None else xlabels, rotation=40, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_xlim(-0.5, len(flow) - 0.5)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_title("")
    fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig, bbox_inches="tight")
    return fig, ax


# Okabe-Ito, one per HNL mass.
_PROGRESSION_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
