"""HNL mu-pi -- per-analysis wiring for core/detvar/process_detvars.py.

The raw ``detvar_<name>_<idx>.df`` frames come from cafpyana's
``hnl_mcnu_detvar`` config (``rec`` savepfp + ``hdr`` + ``nulite``), so the
same :func:`~.preprocess.preprocess_mcbnb` PFP->slice collapse as the nominal
BNB frame applies.
"""

from . import preprocess

SLC_KEY = "rec"

DEFAULT_PREPROCESS_FN = preprocess.preprocess_mcbnb

# Preprocess-only store for now; the real cut list is applied later via
# get_total_cov(..., cuts=...).
SELECTION_FN_MAP = {}
