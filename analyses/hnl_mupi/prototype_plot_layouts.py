"""PROTOTYPE -- throwaway. Answers one question: what should the hnl_mupi plots look like?

Plan: plot-layout variants of the same three variables (slc nu_score,
slc BCFM score, long1 track length; after DEFAULT_CUTS), on a few hundred
events per sample, written to a new output directory together with one
viewer page (``index.html``, switch variants with the bottom bar, the
arrow keys, or ``?variant=A|B|C|D``).

  A  Notebook style -- the selection notebook's ``stack_plot``: m300 | m365
     side by side, shared y, HNL line area-normalised to the BNB stack.
     Saves ``<level>_<var>.pdf`` (the selection_study naming).
  B  core ``plot_mc_hnl``: one figure per mass, MC-stat band, SBND Internal
     label; HNL drawn as a translucent fill. Saves ``<level>_<var>_m<mass>_core.pdf``.
  C  ``hnl_mupi.plot_mc_hnl``: B's legend and stat band, HNL as an unfilled
     step line (A's style). Saves ``<level>_<var>_m<mass>.pdf``.
  D  C with the Punzi-FOM strip (``show_fom=True``). Saves ``..._m<mass>_fom.pdf``.

Run (after sourcing cafpyana's setup.sh)::

    cafpybara/env/bin/python -I cafpybara/analyses/hnl_mupi/prototype_plot_layouts.py \
        ~/scratch/hnl_mupi_layout_proto_<tag> [max_events]

BNB numbers on a subsample are layout-only: the POT weight is approximate.
"""
import base64
import io
import sys
from pathlib import Path

sys.path[:0] = ['/exp/sbnd/app/users/sdas1/cafpyana', '/exp/sbnd/app/users/sdas1']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import cafpybara.analyses.hnl_mupi as ana
from cafpybara.analyses.hnl_mupi import config

OUT = Path(sys.argv[1]).expanduser()
MAX_EVENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
OUT.mkdir(parents=True, exist_ok=False)          # write-once dCache: always a fresh dir
MASSES = (300, 365)

VARS = [  # (save name, column, xlabel, bins, logy)
    ("slc_nu_score",   ("slc", "nu_score", "", "", "", ""),               r"Pandora $\nu$ score",  np.linspace(0.5, 1, 51),  True),
    ("slc_bcfm_score", ("slc", "barycenterFM", "score", "", "", ""),       "BCFM score",            np.linspace(0, 2, 81),    True),
    ("long1_len",      ("long1", "trk", "len", "", "", ""),               "Longest track: Track length [cm]", np.linspace(0, 400, 81), True),
]


# ---------------------------------------------------------------- load
def load():
    bnb, pot, _ = ana.load_mc(config.BNB_FILE, max_events=MAX_EVENTS * 20)
    hnl = {m: ana.load_mchnl(config.HNL_FILES[m], max_events=MAX_EVENTS)[0] for m in MASSES}
    return bnb, pot, hnl


bnb0, POT, hnl0 = load()
W = config.TARGET_POT / POT
frames0 = {"bnb": bnb0, **{f"m{m}": hnl0[m] for m in MASSES}}
CUTFLOW = ana.cutflow(frames0, ana.DEFAULT_CUTS, weights={"bnb": W}).round(3)
bnb = ana.select(bnb0, ana.DEFAULT_CUTS)
hnl = {m: ana.select(h, ana.DEFAULT_CUTS) for m, h in hnl0.items()}
hnl_sig = {m: h[h.signal == ana.signal_dict["hnl"]] for m, h in hnl.items()}

STATE = {
    "max_events (HNL; BNB x20)": MAX_EVENTS,
    "BNB slices pre/post cuts": f"{len(bnb0)} / {len(bnb)}",
    **{f"m{m} slices pre/post cuts (HNL-matched post)": f"{len(hnl0[m])} / {len(hnl[m])} ({len(hnl_sig[m])})" for m in MASSES},
    "BNB POT (scaled to subsample, approx.)": f"{POT:.3e}",
    "BNB weight to 1e20 POT": f"{W:.3f}",
    "fixes on BNB frame": ", ".join(ana.applied_fixes(bnb0)),
}
print("\n".join(f"{k}: {v}" for k, v in STATE.items()))
print(CUTFLOW.to_string())


def finite(df, col):
    v = df[col].astype(float)
    return df[np.isfinite(v) & (v != -999) & (v != -9999)]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return name + ".pdf", base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- variant A
def variant_a(name, col, xlabel, bins, logy):
    b = finite(bnb, col)
    cats = list(ana.background_categories.items())
    data = [b[b.signal == e["value"]][col].to_numpy(float) for _, e in cats]
    total = np.histogram(b[col].astype(float), bins=bins)[0] * W
    fig, axes = plt.subplots(1, len(MASSES), figsize=(14, 5), sharey=True)
    for ax, m in zip(axes, MASSES):
        ax.hist(data, bins=bins, weights=[np.full(len(d), W) for d in data], stacked=True,
                color=[e["color"] for _, e in cats], label=[e["label"] for _, e in cats],
                histtype="stepfilled", edgecolor="black", linewidth=0.4)
        v = finite(hnl_sig[m], col)[col].to_numpy(float)
        h = np.histogram(v, bins=bins)[0]
        if h.sum() and total.sum():
            ax.stairs(h * total.sum() / h.sum(), bins, color="black", lw=2,
                      label=f"HNL {m} MeV (area-norm., N={v.size})")
        ax.set_xlabel(xlabel)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8, ncol=2, loc="best")
    axes[0].set_ylabel(f"Slices / bin (BNB @ {config.TARGET_POT:.0e} POT)")
    fig.tight_layout()
    return [save(fig, name)]


# ---------------------------------------------------------------- variants B, C, D
def _per_mass(name, col, xlabel, bins, logy, plot_fn, suffix="", **extra):
    out = []
    b = finite(bnb, col)
    for m in MASSES:
        s = finite(hnl_sig[m], col)
        n_b = np.histogram(b[col].astype(float), bins=bins)[0].sum() * W
        n_s = np.histogram(s[col].astype(float), bins=bins)[0].sum()
        res = plot_fn(m, b, s, col, bins, scale_nu=W, scale_hnl=n_b / n_s if n_s else 1.0,
                      log_y=logy, overflow=False, xlabel=xlabel,
                      ylabel=f"Slices (BNB @ {config.TARGET_POT:.0e} POT)", title=" ",
                      counts=True, legend_kwargs={"fontsize": 8}, **extra)
        out.append(save(res[0], f"{name}_m{m}{suffix}"))
    return out


def _core_fn(m, b, s, col, bins, **kw):
    from cafpybara.core.plotting import plot_mc_hnl as core_plot_mc_hnl
    cats = {"hnl": {**ana.hnl_categories["hnl"], "label": f"HNL {m} MeV (area-norm.)"}}
    return core_plot_mc_hnl(b, s, col, bins, hnl_categories=cats,
                            categories=ana.background_categories, signal_dict=ana.signal_dict, **kw)


def _ana_fn(m, b, s, col, bins, **kw):
    return ana.plot_mc_hnl(b, s, col, bins, hnl_label=f"HNL {m} MeV (area-norm.)", **kw)


def variant_b(*v):
    return _per_mass(*v, _core_fn, suffix="_core")


def variant_c(*v):
    return _per_mass(*v, _ana_fn)


def variant_d(*v):
    return _per_mass(*v, _ana_fn, suffix="_fom", show_fom=True)


VARIANTS = {
    "A": ("Notebook style: masses side by side", variant_a),
    "B": ("core plot_mc_hnl per mass (HNL filled)", variant_b),
    "C": ("hnl_mupi plot_mc_hnl: B + HNL as stairs", variant_c),
    "D": ("C + Punzi FOM strip", variant_d),
}
RESULTS = {k: [img for v in VARS for img in fn(*v)] for k, (_, fn) in VARIANTS.items()}
for k, imgs in RESULTS.items():
    print(k, [f for f, _ in imgs])


# ---------------------------------------------------------------- viewer
def table(d):
    return "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in d.items())


sections = []
for k, (label, _) in VARIANTS.items():
    figs = "".join(f'<figure><img src="data:image/png;base64,{b64}" alt="{f}"><figcaption>{f}</figcaption></figure>'
                   for f, b64 in RESULTS[k])
    sections.append(f'<section data-variant="{k}"><h2>{k}: {label}</h2>{figs}</section>')

html = f"""<title>HNL Plot Layouts</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* layout: one reading column; variants swap in place; floating pill switcher */
:root{{--bg:#f7f8f6;--fg:#1b1f1c;--muted:#5d665f;--line:#d9ddd8;--bar:#1b1f1c;--barfg:#f7f8f6;--card:#fff;--cap:#4a524c}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161615;--fg:#ececea;--muted:#a3a39e;--line:#333;--bar:#ececea;--barfg:#161615;--card:#fff;--cap:#4a524c;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#161615;--fg:#ececea;--muted:#a3a39e;--line:#333;--bar:#ececea;--barfg:#161615;--card:#fff;--cap:#4a524c;color-scheme:dark}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;padding-block:16px 96px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
h1,h2{{text-wrap:balance}} h2{{font-size:1.1rem}} code{{font-family:ui-monospace,monospace}}
main{{max-width:1200px;margin:0 auto}} h1{{font-size:1.4rem;margin:.2em 0}} p{{color:var(--muted);max-width:75ch}}
details{{margin:1em 0}} table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;white-space:nowrap}}
section{{display:none}} section.on{{display:block}}
figure{{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px}}
figure img{{max-width:100%;height:auto;display:block}} figcaption{{color:var(--cap);font:12px ui-monospace,monospace}}
#bar{{position:fixed;left:50%;bottom:calc(16px + env(safe-area-inset-bottom,0px));transform:translateX(-50%);background:var(--bar);color:var(--barfg);
border-radius:999px;padding:6px 10px;display:flex;gap:10px;align-items:center;box-shadow:0 4px 18px rgba(0,0,0,.3);font-size:14px}}
#bar button{{background:none;border:0;color:inherit;font-size:18px;cursor:pointer;padding:4px 10px}} #bar button:focus-visible{{outline:2px solid var(--barfg);border-radius:999px}}
</style><main>
<h1>HNL mu-pi plot layouts (prototype)</h1>
<p>Throwaway prototype. Same three variables after DEFAULT_CUTS, {MAX_EVENTS} events per HNL mass and {MAX_EVENTS*20} BNB events.
BNB normalisation on a subsample is approximate, so judge layout, not numbers. Switch with the bar, the arrow keys, or <code>?variant=</code>.</p>
<details><summary>State: sample sizes and cut flow</summary><table>{table(STATE)}</table><br>{CUTFLOW.to_html()}</details>
{''.join(sections)}
</main>
<div id="bar"><button id="prev" aria-label="previous">&larr;</button><span id="lab"></span><button id="next" aria-label="next">&rarr;</button></div>
<script>
const names={{{",".join(f'"{k}":"{v[0]}"' for k, v in VARIANTS.items())}}};const keys=Object.keys(names);
function show(k){{if(!keys.includes(k))k=keys[0];document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.dataset.variant===k));
document.getElementById('lab').textContent=k+' ('+names[k]+')';const u=new URL(location);u.searchParams.set('variant',k);try{{history.replaceState(null,'',u)}}catch(e){{}}}}
let cur=new URLSearchParams(location.search).get('variant')||'A';
function step(d){{cur=keys[(keys.indexOf(cur)+d+keys.length)%keys.length];show(cur)}}
document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);
addEventListener('keydown',e=>{{const t=e.target;if(t.closest&&t.closest('input,textarea,[contenteditable]'))return;
if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1)}});show(cur);
</script>"""
(OUT / "index.html").write_text(html)
print("viewer:", OUT / "index.html")
