"""Toy PFP-level frames with the cafpyana column layout, for fast selection tests."""
import numpy as np
import pandas as pd

DEPTH = 6


def col(path: str) -> tuple:
    p = path.split(".")
    return tuple(p + [""] * (DEPTH - len(p)))


# One PFP row; every field is overridable. Defaults: a contained, good, MIP-like track
# that is a daughter of the neutrino, starting at the vertex and heading along +z.
DEFAULTS = {
    "pfp.parent_is_primary": 1, "pfp.parent": 3, "pfp.trackScore": 0.9,
    "pfp.trk.len": 50.0,
    "pfp.trk.start.x": 50.0, "pfp.trk.start.y": 0.0, "pfp.trk.start.z": 100.0,
    "pfp.trk.end.x": 50.0, "pfp.trk.end.y": 0.0, "pfp.trk.end.z": 150.0,
    "pfp.trk.dir.x": 0.0, "pfp.trk.dir.y": 0.0, "pfp.trk.dir.z": 1.0,
    "pfp.trk.chi2pid.I2.pid_ndof": 50, "pfp.trk.chi2pid.I2.chi2_muon": 5.0,
    "pfp.trk.chi2pid.I2.chi2_proton": 120.0,
    "pfp.trk.rangeP.p_pion": 0.5, "pfp.shw.bestplane_energy": -5.0,
}
VERTEX = (50.0, 0.0, 100.0)


def neutrino_pfp(pid=3):
    """The neutrino PFP itself: parent -1, trackScore -5 (never a daughter)."""
    return {"pfp.id": pid, "pfp.parent": -1, "pfp.trackScore": -5.0}


def frame(slices: dict) -> pd.DataFrame:
    """``slices``: {slice_number: [PFP dict, ...]} -> PFP-level frame.

    Index levels ``(ntuple, entry, slc, pfp)``; PFP ``id`` defaults to its position
    (0, 1, ...), so ``neutrino_pfp(pid)`` should use an id past the daughters.
    """
    rows, index = [], []
    for s, pfps in slices.items():
        for i, over in enumerate(pfps):
            r = {**DEFAULTS, "pfp.id": i, **over}
            r.update({"slc.vertex.x": VERTEX[0], "slc.vertex.y": VERTEX[1], "slc.vertex.z": VERTEX[2]})
            rows.append(r)
            index.append((0, 0, s, i))
    df = pd.DataFrame(rows)
    df.columns = pd.MultiIndex.from_tuples([col(c) for c in df.columns])
    df.index = pd.MultiIndex.from_tuples(index, names=["__ntuple", "entry", "rec.slc..index", "rec.slc.reco.pfp..index"])
    return df
