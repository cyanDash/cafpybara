"""TEMPLATE -- your analysis's own detector-variation wiring for
core/detvar/process_detvars.py.
"""

# TODO: table key for the slice-level df in raw detvar_*.df files (see
# hnlpi0/nuecc detvar.py).
SLC_KEY = "TODO_SET_ME"


def DEFAULT_PREPROCESS_FN(df):
    """TODO: replace with your real MC preprocess function."""
    raise NotImplementedError(
        "TEMPLATE: set DEFAULT_PREPROCESS_FN in detvar.py to your real MC "
        "preprocess function (e.g. preprocess.preprocess_mc)."
    )


# TODO: selection-type name -> (cuts, stage). Leave empty if the
# always-available "preprocess" (no-selection) store is all you need for now.
SELECTION_FN_MAP = {}
