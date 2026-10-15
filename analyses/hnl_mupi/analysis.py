"""HNL mu-pi signal categories, truth categorisation and cut sequence.

Truth is used only to colour plots and tune thresholds; every cut in
:data:`DEFAULT_CUTS` uses reco columns only.
"""

import numpy as np
import pandas as pd

from ...core.classes import CutSpec

__all__ = [
    'signal_categories', 'signal_dict', 'background_categories', 'hnl_categories',
    'DEFAULT_CUTS', 'in_fv', 'define_signal', 'define_signal_hnl',
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

DEFAULT_CUTS = [
    CutSpec("bcfm_score", variable=("slc", "barycenterFM", "score"), min=0.05,
            label="BCFM score > 0.05"),
    CutSpec("crlongtrkdiry", variable=("slc", "nuid", "crlongtrkdiry"), min=-0.5,
            label="CR long-track dir y > -0.5"),
]
