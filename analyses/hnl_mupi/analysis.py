"""HNL mu-pi signal categories, truth categorisation and cut sequence.

Truth is used only to colour plots and tune thresholds; every cut in
:data:`DEFAULT_CUTS` uses reco columns only.
"""

import numpy as np
import pandas as pd

from ...core.classes import CutSpec

__all__ = [
    'signal_categories', 'signal_dict', 'background_categories', 'hnl_categories',
    'FLASH_TIME_WINDOW', 'DEFAULT_CUTS', 'in_fv', 'define_signal', 'define_signal_hnl',
    'HNL_DISPLAY_SCALE', 'hnl_label',
]


# ---------------------------------------------------------------------------
# Signal/background category definitions
# ---------------------------------------------------------------------------

# Dict order is the stacking order, bottom to top. Colours are Okabe-Ito,
# the same as the selection notebook.
signal_categories = {
    "cosmic":    {"value": 8,  "label": r"Cosmic (no $\nu$ match)",     "color": "#8C8C8C"},
    "nonFV":     {"value": 7,  "label": r"$\nu$ vertex outside FV",      "color": "#F2E640"},
    "NC":        {"value": 6,  "label": "NC",                            "color": "#E69900"},
    "CCQE":      {"value": 1,  "label": "CC QE",                         "color": "#0073B3"},
    "CC2p2h":    {"value": 2,  "label": "CC 2p2h",                       "color": "#59B3E6"},
    "CCRES":     {"value": 3,  "label": "CC RES",                        "color": "#009980"},
    "CCDIS":     {"value": 4,  "label": "CC DIS",                        "color": "#CC4D00"},
    "CCother":   {"value": 5,  "label": "CC COH / other",                "color": "#CC99B3"},
    "offbeam":   {"value": 9,  "label": "Offbeam",                       "color": "#D0D2D4"},
    "hnl":       {"value": 0,  "label": "HNL",                           "color": "#000000"},
    "hnlcosmic": {"value": 10, "label": "HNL cosmic",                    "color": "#131E29"},
}
signal_dict = {k: v["value"] for k, v in signal_categories.items()}

background_categories = {k: v for k, v in signal_categories.items()
                         if k not in ("hnl", "hnlcosmic", "offbeam")}
hnl_categories = {"hnl": signal_categories["hnl"]}

# Factor the HNL rate is multiplied by on plots, per mass in MeV. The rate goes
# as |U_mu4|^4 (production and decay), so the legend's |U|^2 is the simulated
# value times sqrt(scale).
HNL_DISPLAY_SCALE = {300: 1, 365: 1}


def hnl_label(mass, simU, scale=1.0):
    """Legend label for an HNL line drawn at ``scale`` times the simulated rate."""
    from ...core.utils import sci_notation
    return (f"{mass} MeV HNL " + r"$\mu\pi$" + "\n"
            + r"|U$_{\mu 4}$|$^2$ = " + str(sci_notation(simU * np.sqrt(scale), 2, 2)))


# ---------------------------------------------------------------------------
# Truth categorisation
# ---------------------------------------------------------------------------

def in_fv(x, y, z):
    """Same box as makedf.util.InFV(det='SBND') used by the cafpyana preselection."""
    return (x > -190) & (x < 190) & (y > -190) & (y < 190) & (z > 10) & (z < 450)


def define_signal(indf: pd.DataFrame) -> pd.DataFrame:
    """BNB nu + cosmics: stamp ``signal`` from the truth-matched neutrino."""
    s = indf.slc
    matched = (s.tmatch.idx >= 0).to_numpy()
    pos = s.truth.position
    infv = in_fv(pos.x, pos.y, pos.z).to_numpy()
    iscc = (s.truth.iscc == 1).to_numpy()
    mode = s.truth.genie_mode.to_numpy()
    nu = matched & infv
    cc = nu & iscc

    signal = np.full(len(indf), signal_dict["cosmic"], dtype=np.int16)
    signal[matched & ~infv] = signal_dict["nonFV"]
    signal[nu & ~iscc] = signal_dict["NC"]
    signal[cc] = signal_dict["CCother"]
    signal[cc & (mode == 0)] = signal_dict["CCQE"]
    signal[cc & (mode == 10)] = signal_dict["CC2p2h"]
    signal[cc & (mode == 1)] = signal_dict["CCRES"]
    signal[cc & (mode == 2)] = signal_dict["CCDIS"]
    indf["signal"] = signal
    return indf


def define_signal_hnl(indf: pd.DataFrame) -> pd.DataFrame:
    """MeVPrtl HNL: ``hnl`` if the slice matches the HNL decay, else ``hnlcosmic``."""
    matched = (indf.slc.tmatch.idx >= 0).to_numpy()
    indf["signal"] = np.where(matched, signal_dict["hnl"], signal_dict["hnlcosmic"]).astype(np.int16)
    return indf


# ---------------------------------------------------------------------------
# Cut sequence (applied after the cafpyana preselection)
# ---------------------------------------------------------------------------

# Beam window on the calibrated BCFM flash time [ns], the hnlpi0 window with the
# upper edge raised from 2250 to 3500 ns. HNLs from kaons decaying at rest are
# slow (m365: E = 0.37 GeV, beta ~ 0.17) and arrive up to ~3.6 us; 2250 ns would
# drop 2.1% (weighted) of the m365 signal, 3500 ns drops 0.003%.
FLASH_TIME_WINDOW = (-250.0, 3500.0)

DEFAULT_CUTS = [
    CutSpec("bcfm_score", variable=("slc", "barycenterFM", "score"), min=0.05,
            label="BCFM score > 0.05"),
    CutSpec("flash_time", variable=("slc", "barycenterFM", "flashTime_calib"),
            min=FLASH_TIME_WINDOW[0], max=FLASH_TIME_WINDOW[1],
            label=f"BCFM flash time in ({FLASH_TIME_WINDOW[0]:g}, {FLASH_TIME_WINDOW[1]:g}) ns"),
    CutSpec("crlongtrkdiry", variable=("slc", "nuid", "crlongtrkdiry"), min=-0.5,
            label="CR long-track dir y > -0.5"),
]
