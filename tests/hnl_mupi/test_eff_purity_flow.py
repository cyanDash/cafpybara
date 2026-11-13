"""eff_purity_flow on toy slice frames with known answers."""
import numpy as np
import pandas as pd
import pytest

from cafpybara.core.classes import CutSpec
from cafpybara.analyses.hnl_mupi.analysis import signal_dict
from cafpybara.analyses.hnl_mupi.selection import eff_purity_flow

CUTS = [CutSpec("x_gt_1", accessor=lambda df: df.x, min=1.0, label="x > 1"),
        CutSpec("x_gt_2", accessor=lambda df: df.x, min=2.0, label="x > 2")]


def toy(x, signal, w=None):
    df = pd.DataFrame({"x": np.asarray(x, float), "signal": signal})
    if w is not None:
        df["weights_mc"] = np.asarray(w, float)
    return df


def test_efficiency_and_purity_are_cumulative_and_weighted():
    hnl = toy([0.5, 1.5, 2.5, 2.5], [signal_dict["hnl"]] * 3 + [signal_dict["hnlcosmic"]], w=[1, 1, 2, 100])
    bnb = toy([0.5, 1.5, 2.5, 3.5], signal_dict["cosmic"])                       # no weights_mc: ones
    off = toy([2.5], signal_dict["offbeam"], w=[3.0])
    f = eff_purity_flow({"m": hnl}, {"bnb": bnb, "off": off}, CUTS, sig_scale={"m": 0.5}, bkg_scale=2.0)
    assert list(f.index) == ["preselection", "x_gt_1", "x_gt_2"]
    # hnlcosmic (w=100) never counts; S = 0.5 * sum(w)
    np.testing.assert_allclose(f["S m"], [2.0, 1.5, 1.0])
    np.testing.assert_allclose(f["eff m"], [1.0, 0.75, 0.5])
    # B = 2 * (bnb ones + off w=3)
    np.testing.assert_allclose(f["B"], [14.0, 12.0, 10.0])
    np.testing.assert_allclose(f["purity m"], f["S m"] / (f["S m"] + f["B"]))
    np.testing.assert_allclose(f["B bnb"] + f["B off"], f["B"])
    np.testing.assert_allclose(f["B off"], [6.0, 6.0, 6.0])                     # off-beam is in B
    np.testing.assert_allclose(f["ssb_rel m"], (f["S m"] / np.sqrt(f["B"])) / (2.0 / np.sqrt(14.0)))
    assert f.loc["x_gt_2", "cut"] == "x > 2"


def test_track_cuts_containment_last():
    from cafpybara.analyses.hnl_mupi.analysis import track_cuts
    default = [c.name for c in track_cuts(30.0)]
    last = [c.name for c in track_cuts(30.0, containment_last=True)]
    assert last[-1] == "contained" and sorted(last) == sorted(default)
    assert [n for n in last if n != "contained"] == [n for n in default if n != "contained"]


def test_eff_purity_flow_split_branches():
    cuts = [CutSpec("x_gt_1", accessor=lambda df: df.x, min=1.0, label="x > 1"),
            CutSpec("contained", accessor=lambda df: df.n_uncontained, max=0.5, label="contained")]
    hnl = toy([0.5, 2, 2, 2], signal_dict["hnl"], w=[1, 1, 2, 3]).assign(n_uncontained=[0, 0, 1, 1])
    bnb = toy([0.5, 2, 2, 2, 2, 2], signal_dict["cosmic"]).assign(n_uncontained=[0, 0, 0, 1, 1, 1])
    f = eff_purity_flow({"m": hnl}, {"bnb": bnb}, cuts, sig_scale={"m": 1.0}, split=True)
    assert list(f.index) == ["preselection", "x_gt_1", "contained", "uncontained"]
    assert list(f.branch) == [False, False, True, True]
    np.testing.assert_allclose(f["S m"], [7, 6, 1, 5])
    np.testing.assert_allclose(f["B"], [6, 5, 2, 3])
    with pytest.raises(ValueError):
        eff_purity_flow({"m": hnl}, {"bnb": bnb}, cuts[:1], sig_scale={"m": 1.0}, split=True)


def test_split_table_categories_and_combination():
    from cafpybara.analyses.hnl_mupi.selection import split_table
    cuts = [CutSpec("x_gt_1", accessor=lambda df: df.x, min=1.0, label="x > 1")]
    hnl = toy([0.5, 2, 2, 2], signal_dict["hnl"], w=[1, 1, 2, 3]).assign(n_uncontained=[0, 0, 1, 1])
    bnb = toy([0.5, 2, 2, 2, 2, 2], signal_dict["cosmic"]).assign(n_uncontained=[0, 0, 0, 1, 1, 1])
    t = split_table({"m": hnl}, {"bnb": bnb}, cuts, sig_scale={"m": 1.0}, bkg_scale=1.0)
    np.testing.assert_allclose(t.loc[("m", "contained"), ["S", "B", "B_n"]].astype(float), [1, 2, 2])
    np.testing.assert_allclose(t.loc[("m", "uncontained"), ["S", "B", "B_n"]].astype(float), [5, 3, 3])
    z = np.sqrt(1 / 2 + 25 / 3)
    np.testing.assert_allclose(t.loc[("m", "combined"), "S/sqrt(B)"], z)
    np.testing.assert_allclose(t.loc[("m", "combined"), "rel. presel."], z / (7 / np.sqrt(6)))
