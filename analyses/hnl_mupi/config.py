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
# 60k files: the 20k snapshot plus 40k more (disjoint), merged with merge_dfs.py; 4.33e19 POT.
BNB_FILE = DF_DIR + "/bnb/hnl_mupi_bnb60k_savepfp.df"

# Data, hnl_data_savepfp, plain merge (no --fix-subrun). Run 1 (Feb-Jul 2025, SBND2026A gen2 respin BNB+light and
# the MCP2025C Fall25 in-time off-beam+light) is the default: the per-period timing constants in
# core/timing_calibration.py cover Run 1 only. Run 3 (fixed-dev, ~5e18 POT, Oct 2026) has no calibration yet:
# preprocess_databnb drops every slice.
DATA_BNB_FILE = DF_DIR + "/data_run1/hnl_mupi_run1_bnb_savepfp.df"
DATA_OFFBEAM_FILE = DF_DIR + "/data_run1/hnl_mupi_run1_offbeam_savepfp.df"
DATA_RUN3_BNB_FILE = DF_DIR + "/data_run3/hnl_mupi_data_bnb_savepfp.df"
DATA_RUN3_OFFBEAM_FILE = DF_DIR + "/data_run3/hnl_mupi_data_offbeam_savepfp.df"
# Fraction of off-beam gates that are not cosmic-only (beam-on cosmic fraction correction), as in hnlpi0/nuecc.
OFFBEAM_F = 0.0725

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
