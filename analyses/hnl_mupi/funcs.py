"""TEMPLATE -- your analysis's own SystematicsInput factory.

Do NOT define your own `get_total_cov` here. `core.funcs.get_total_cov` is
the single implementation for every analysis -- this file's only job
is `systs_input()`, a factory that fully resolves every default into
a concrete value before anything reaches `core`.
"""

from ...core.classes import SystematicsInput
from . import config
from .analysis import DEFAULT_CUTS, define_signal

__all__ = ['systs_input']


def systs_input(mcbnb_pot, *, cuts=None, uncertainty_keys=None, **kwargs):
    """Build a fully-resolved SystematicsInput for this analysis.

    Pass the result straight to :func:`~cafpybara.core.funcs.get_total_cov`
    via ``**systs_input(...).to_kwargs()``, or to
    :class:`~cafpybara.core.classes.PlottingConfig`'s ``systs=`` argument.

    TODO: keep `cuts=None -> DEFAULT_CUTS` only if one obvious default cut
    list exists (nueCC's case); else drop the default and make `cuts`
    required (hnlpi0's case).
    """
    if cuts is None:
        cuts = DEFAULT_CUTS

    detvar_dict = kwargs.pop('detvar_dict', None)
    detvar_files = kwargs.pop('detvar_files', None)
    if detvar_dict is None and detvar_files is None:
        # TODO: set your detvar_dict/detvar_files default.
        detvar_files = config.DETVAR_DICT_FILES

    kwargs.setdefault('define_signal_fn', define_signal)

    # TODO: if you have an in-time-cosmic sample, default
    # intime_file/intime_key/offbeam_value/intime_preprocess_fn here and
    # add 'cosmic' to the default uncertainty_keys below. Otherwise keep
    # this raise.
    if uncertainty_keys is not None and 'cosmic' in set(uncertainty_keys):
        raise ValueError(
            "TEMPLATE: 'cosmic' was requested, but no in-time-cosmic sample "
            "has been configured for this analysis yet. Either configure "
            "one (see the TODO above) or drop 'cosmic' from uncertainty_keys."
        )
    if uncertainty_keys is None:
        uncertainty_keys = {'rate', 'detv', 'norm'}

    return SystematicsInput(
        mcbnb_pot=mcbnb_pot, cuts=cuts, detvar_dict=detvar_dict, detvar_files=detvar_files,
        uncertainty_keys=uncertainty_keys, **kwargs,
    )
