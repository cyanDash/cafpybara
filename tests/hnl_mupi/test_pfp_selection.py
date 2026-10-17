"""pfp_features/pfp_counts on toy frames: one case per track cut, with known answers."""
import numpy as np
import pytest

from frames import frame, neutrino_pfp
import cafpybara.analyses.hnl_mupi.preprocess as pp


def counts(*daughters, **over):
    """Counts of one slice: ``daughters`` (PFP dicts) plus the neutrino PFP, whose id is their count."""
    nu = len(daughters)
    pfps = [{"pfp.parent": nu, **d} for d in daughters] + [neutrino_pfp(nu)]
    return pp.pfp_counts(pp.pfp_features(frame({0: pfps})), **over).iloc[0]


def test_good_tracks_need_daughter_score_length_and_vertex_distance():
    c = counts({},                                   # good
               {"pfp.trk.len": 2.0},                 # too short
               {"pfp.trk.start.x": 80.0},            # 30 cm from the vertex
               {"pfp.trackScore": 0.3},              # shower-like
               {"pfp.parent_is_primary": 0})         # not a neutrino daughter
    assert c.n_good == 1


def test_neutrino_pfp_is_never_a_daughter():
    assert counts().n_good == 0


def test_mip_needs_length_and_chi2_cuts_and_valid_pid():
    c = counts({},                                           # MIP
               {"pfp.trk.len": 8.0},                         # good but not > 10 cm
               {"pfp.trk.chi2pid.I2.chi2_muon": 25.0},       # chi2_mu >= 20
               {"pfp.trk.chi2pid.I2.chi2_proton": 50.0},     # chi2_p <= 85
               {"pfp.trk.chi2pid.I2.pid_ndof": 0})           # no valid PID
    assert (c.n_good, c.n_mip) == (5, 1)


def test_primary_shower_veto_counts_only_daughter_showers_above_threshold():
    c = counts({"pfp.trackScore": 0.3, "pfp.shw.bestplane_energy": 0.10},                    # veto
               {"pfp.trackScore": 0.3, "pfp.shw.bestplane_energy": 0.03},                    # below 55 MeV
               {"pfp.trackScore": 0.3, "pfp.shw.bestplane_energy": 0.10, "pfp.parent_is_primary": 0})
    assert c.n_shw == 1


def test_michel_like_needs_low_energy_and_no_daughters():
    c = counts({"pfp.id": 0, "pfp.trackScore": 0.3, "pfp.shw.bestplane_energy": 0.03},       # Michel-like
               {"pfp.id": 1, "pfp.trackScore": 0.3, "pfp.shw.bestplane_energy": 0.10},       # too energetic
               {"pfp.id": 2, "pfp.trackScore": 0.3, "pfp.shw.bestplane_energy": 0.03},       # has a child (next row)
               {"pfp.id": 3, "pfp.parent": 2, "pfp.parent_is_primary": 0, "pfp.trackScore": 0.9})
    assert c.n_michel == 1


def test_opening_angle_uses_the_two_longest_mip_tracks():
    c = counts({"pfp.trk.len": 50.0},                                                               # dir +z
               {"pfp.trk.len": 40.0, "pfp.trk.dir.x": 1.0, "pfp.trk.dir.z": 0.0},                    # dir +x
               {"pfp.trk.len": 30.0, "pfp.trk.dir.y": 1.0, "pfp.trk.dir.z": 0.0})                    # shortest, ignored
    assert c.mip_open_angle == pytest.approx(90.0)


def test_opening_angle_is_nan_with_fewer_than_two_mips():
    assert np.isnan(counts({}).mip_open_angle)


def test_track_leaving_the_tpc_or_in_the_high_yz_region_is_uncontained():
    assert counts({}).n_uncontained == 0
    assert counts({"pfp.trk.end.x": 198.0}).n_uncontained == 1                       # 2 cm inside, margin 5 cm
    assert counts({"pfp.trk.end.x": 198.0}, contain_margin=1.0).n_uncontained == 0
    assert counts({"pfp.trk.end.y": 150.0, "pfp.trk.end.z": 300.0}).n_uncontained == 1   # high y-z


def test_cathode_crossing_track():
    assert counts({"pfp.trk.end.x": -10.0}).n_cathode == 1
    assert counts({}).n_cathode == 0


def test_thresholds_override_the_defaults():
    assert counts({}).n_good == 1
    assert counts({}, min_len=60.0).n_good == 0


def test_counts_cover_every_slice():
    f = pp.pfp_features(frame({0: [{}, neutrino_pfp(1)], 5: [{"pfp.trackScore": 0.3}, neutrino_pfp(1)]}))
    c = pp.pfp_counts(f)
    assert list(c.index.get_level_values(2)) == [0, 5]
    assert list(c.n_good) == [1, 0]
