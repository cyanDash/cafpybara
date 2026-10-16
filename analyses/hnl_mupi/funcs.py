"""HNL mu-pi SystematicsInput factory.

Do NOT define your own `get_total_cov` here. `core.funcs.get_total_cov` is
the single implementation for every analysis -- this file's only job
is `systs_input()`, a factory that fully resolves every default into
a concrete value before anything reaches `core`.

Available blocks: ``'detv'`` (PDS light-yield detvars in
``config.DETVAR_DIR``, plus the shower-only ``calo_*`` ones, which are zero
for track variables) and ``'norm'`` (POT and target-count normalisation).
``'rate'`` needs ``_wgt`` multisim frames and ``'cosmic'`` an in-time-cosmic
sample; neither exists yet. The pyhf inputs take their errors from this
through ``sensitivity.pyhf_input_dict(..., systs=systs_input(...))``.
"""

from ...core.classes import SystematicsInput
from . import config
from .analysis import DEFAULT_CUTS, define_signal

__all__ = ['DEFAULT_UNCERTAINTY_KEYS', 'systs_input']

DEFAULT_UNCERTAINTY_KEYS = frozenset({'detv', 'norm'})
_MISSING_SAMPLES = {
    'rate': "no _wgt (GENIE/flux/G4 multisim) frames",
    'cosmic': "no in-time-cosmic sample",
}


def systs_input(mcbnb_pot, *, cuts=None, uncertainty_keys=None, **kwargs):
    """Build a fully-resolved SystematicsInput for this analysis.

    Pass the result straight to :func:`~cafpybara.core.funcs.get_total_cov`
    via ``**systs_input(...).to_kwargs()``, to
    :func:`~.sensitivity.pyhf_input_dict` as ``systs=``, or to
    :class:`~cafpybara.core.classes.PlottingConfig`'s ``systs=`` argument.

    ``mcbnb_pot`` is the POT of the frame the covariance is computed for (the
    MeVPrtl generator POT for an HNL frame). ``cuts`` defaults to
    ``DEFAULT_CUTS`` and must be the cuts the frame was selected with (the
    detvar guard in ``get_total_cov`` checks this). ``uncertainty_keys``
    defaults to ``{'detv', 'norm'}``.
    """
    if cuts is None:
        cuts = DEFAULT_CUTS

    detvar_dict = kwargs.pop('detvar_dict', None)
    detvar_files = kwargs.pop('detvar_files', None)
    if detvar_dict is None and detvar_files is None:
        detvar_files = config.DETVAR_DICT_FILES

    kwargs.setdefault('define_signal_fn', define_signal)

    if uncertainty_keys is None:
        uncertainty_keys = set(DEFAULT_UNCERTAINTY_KEYS)
    for key, why in _MISSING_SAMPLES.items():
        if key in set(uncertainty_keys):
            raise ValueError(f"hnl_mupi: {key!r} was requested, but there is {why} yet.")

    return SystematicsInput(
        mcbnb_pot=mcbnb_pot, cuts=cuts, detvar_dict=detvar_dict, detvar_files=detvar_files,
        uncertainty_keys=uncertainty_keys, **kwargs,
    )
