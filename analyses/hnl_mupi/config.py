"""HNL mu-pi sample paths.

The frames are cafpyana ``*_savepfp`` outputs (``hnl_mcmevprtl_savepfp`` for
the HNL masses, ``hnl_mcnu_savepfp`` for BNB nu + CORSIKA cosmics), already
cut by the cafpyana preselection (not a clear cosmic, nu_score > 0.5, vertex
in the FV box).

No cosmic overlay on the HNL samples yet; regenerate after the migration and
only these paths change.
"""

DF_DIR = "/nashome/s/sdas1/persistent/hnl_mupi/dataframes"
HNL_FILES = {
    300: DF_DIR + "/hnl_m300/hnl_mupi_m300_savepfp.df",
    365: DF_DIR + "/hnl_m365/hnl_mupi_m365_savepfp.df",
}
BNB_FILE = DF_DIR + "/bnb/hnl_mupi_bnb_savepfp.df"

# BNB plots are scaled to this POT.
TARGET_POT = 1.0e20

# TODO: no detvar samples for this topology yet.
DETVAR_DIR = "TODO_SET_ME/replace_with_your_detvar_directory"
DETVAR_DICT_FILES = [DETVAR_DIR + "/detvars.h5"]

# No in-time-cosmic sample for this topology yet.
INTIME_FILE = None
