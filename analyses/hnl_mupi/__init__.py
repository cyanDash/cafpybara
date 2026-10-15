"""HNL -> mu pi analysis -- built on cafpybara.core with this analysis's own defaults.

Usage::

    import cafpybara.analyses.hnl_mupi as ana

    bnb, pot, ngen = ana.load_mc(config.BNB_FILE)                     # BNB nu + cosmics
    hnl, _, _      = ana.load_mchnl(config.HNL_FILES[300])            # MeVPrtl HNL
    ana.cutflow({"bnb": bnb, "m300": hnl}, ana.DEFAULT_CUTS, weights={"bnb": w})

The input frames are cafpyana ``*_savepfp`` frames (one row per PFParticle);
:func:`preprocess.collapse_pfp` turns them into one row per slice, with the
longest and second-longest track-like PFP as the ``long1``/``long2`` column
groups. Every loader here applies that collapse.
"""

from ... import core

from ...core.utils import *
from ...core.io import *
from ...core.plotting import *
from ...core.physics import *
from ...core.syst import *
from ...core.selection import *
from ...core.classes import *
from ...core.funcs import *
from ...core.preprocess import *
from ...core.detvar import *

from . import config
from . import detvar
from .analysis import *
from .preprocess import *
from .selection import *
from .io import *
from .plotting import *
from .funcs import *
