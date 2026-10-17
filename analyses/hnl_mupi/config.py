"""HNL mu-pi sample paths.

The frames are cafpyana ``*_savepfp`` outputs (``hnl_mcmevprtl_savepfp`` for
the HNL masses, ``hnl_mcnu_savepfp`` for BNB nu + CORSIKA cosmics), already
cut by the cafpyana preselection (not a clear cosmic, nu_score > 0.5, vertex
in the FV box).

The HNL samples carry the CORSIKA cosmic overlay (sbndcode v10_22_00,
prodoverlay_corsika_MeVPrtl_hnl_mupi, ~2000 files x 10 events per mass);
hdr.subrun is unique per input file (merge_dfs.py --fix-subrun).
"""

DF_DIR = "/nashome/s/sdas1/persistent/hnl_mupi/dataframes"
HNL_FILES = {
    300: DF_DIR + "/hnl_m300_overlay_full/hnl_mupi_m300_overlay_savepfp.df",
    365: DF_DIR + "/hnl_m365_overlay_full/hnl_mupi_m365_overlay_savepfp.df",
}
BNB_FILE = DF_DIR + "/bnb/hnl_mupi_bnb_savepfp.df"

# BNB plots are scaled to this POT.
TARGET_POT = 1.0e20

# PDS light-yield detvars (CV, 0.94xLY, 1.19xLY, PMTSPEDecrease), run_number
# 400-500 and 501-600 subsets (~10k files each per sample, ~1.45e19 POT per
# group), from cafpyana hnl_mcnu_detvar. Each subset is built by
# process_detvars --analysis hnl_mupi -s preprocess, then the two stores are
# combined by merge_detvar_stores.py (file_idx 0 = run 400-500, 1 = 501-600).
DETVAR_DIR = DF_DIR + "/detvar"
DETVAR_DICT_FILES = [DETVAR_DIR + "/detvars.h5"]

# No in-time-cosmic sample for this topology yet.
INTIME_FILE = None
