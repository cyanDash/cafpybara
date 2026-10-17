"""thresholds.txt round trip, in_high_yz against cafpyana's InFV, open_store/xrootd_url, pyhf-input keys."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

import cafpybara.analyses.hnl_mupi as ana
from cafpybara.core.io import open_store, xrootd_url


def test_every_threshold_is_documented_and_defaulted():
    assert set(ana.THRESHOLD_CUT) == set(ana.DEFAULT_THRESHOLDS)


def test_thresholds_round_trip_and_unknown_key(tmp_path):
    t = dict(ana.DEFAULT_THRESHOLDS, min_len=8.0, open_angle_max=120.0)
    p = ana.write_thresholds(t, tmp_path / "t.txt", header="test", notes={"min_len": "peak"})
    assert ana.read_thresholds(p) == t
    bad = tmp_path / "bad.txt"
    bad.write_text("not_a_threshold = 1\n")
    with pytest.raises(KeyError):
        ana.read_thresholds(bad)


def test_missing_keys_keep_their_defaults(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("min_len = 7   # only one\n")
    t = ana.read_thresholds(p)
    assert t["min_len"] == 7.0 and t["max_dist_vtx"] == ana.DEFAULT_THRESHOLDS["max_dist_vtx"]


def test_in_high_yz_is_the_complement_of_cafpyanas_nohighyz_fv():
    path = os.environ.get("CAFPYANA_PATH")
    if not path:
        pytest.skip("CAFPYANA_PATH not set")
    sys.path.insert(0, path)
    util = pytest.importorskip("makedf.util")
    rng = np.random.default_rng(1)
    p = pd.DataFrame({"x": rng.uniform(-250, 250, 50000), "y": rng.uniform(-250, 250, 50000),
                      "z": rng.uniform(-20, 520, 50000)})
    fv = util.InFV(p, inzback=0, det="SBND_nohighyz").to_numpy()
    box = ((p.x.abs() < 190) & (p.z > 10) & (p.z < 450) & (p.y.abs() < 190)).to_numpy()   # the FV box in x, y, z
    np.testing.assert_array_equal(fv[box], ~ana.in_high_yz(p)[box])


def test_open_store_reads_a_local_file(tmp_path):
    f = tmp_path / "x.df"
    pd.DataFrame({"a": [1, 2]}).to_hdf(f, key="k_0")
    with open_store(str(f)) as s:
        assert list(s["k_0"].a) == [1, 2]


def test_xrootd_url_follows_symlinks_into_pnfs_and_rejects_other_paths(tmp_path):
    link = tmp_path / "persistent"
    link.symlink_to("/pnfs/sbnd/persistent/users/someone")
    assert xrootd_url(str(link / "f.df")) == \
        "root://fndcadoor.fnal.gov:1094//pnfs/fnal.gov/usr/sbnd/persistent/users/someone/f.df"
    with pytest.raises(ValueError):
        xrootd_url(str(tmp_path / "f.df"))


def test_pyhf_inputs_have_hnlpi0_keys_and_the_limit_needs_its_error_key():
    df = pd.DataFrame({"weights_mc": [1.0, 1.0]})
    d = ana.pyhf_input_dict(df, 1e20, weighted=True, m=300, U=1e-7)
    assert {"bins", "counts", "POT", "total_err", "stat_err", "mcstat_err", "m", "U"} <= set(d)
    bad = {k: v for k, v in d.items() if k != "total_err"}
    with pytest.raises(KeyError):
        ana.upper_limit_mu(d, bad)
