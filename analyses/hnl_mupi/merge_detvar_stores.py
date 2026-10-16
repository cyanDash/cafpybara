"""Merge detvar stores built from disjoint file batches into one store.

usage: python -m cafpybara.analyses.hnl_mupi.merge_detvar_stores OUT.h5 IN1.h5 IN2.h5 [...]

Every input must have the same groups and cv_keys (same detvar.py wiring) and
be preprocess-only (no cuts_signature). Batch b gets file_idx=b on every cv/dv
frame so (__ntuple, entry, file_idx) stays unique; cv_iloc of batch b is
shifted by the CV rows of batches < b; meta pot is summed.
"""
import sys
import warnings

import numpy as np
import pandas as pd

_HDF_KW = dict(format='fixed')


def _set_file_idx(df, b):
    idx = df.index.to_frame(index=False)
    idx['file_idx'] = b
    df = df.copy()
    df.index = pd.MultiIndex.from_frame(idx)
    return df


def merge_detvar_stores(out, ins):
    metas = [pd.read_hdf(f, 'meta') for f in ins]
    m0 = metas[0]
    for f, m in zip(ins[1:], metas[1:]):
        assert m.index.sort_values().equals(m0.index.sort_values()), f"groups differ in {f}"
        assert (m.loc[m0.index, ['cv_key', 'n_dv']] == m0[['cv_key', 'n_dv']]).all().all(), \
            f"cv_key/n_dv differ in {f}"
    assert all(m['cuts_signature'].isna().all() for m in metas), "only preprocess-only stores"

    warnings.filterwarnings('ignore', '.*not a valid Python identifier.*')
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    stores = [pd.HDFStore(f, 'r') for f in ins]
    try:
        with pd.HDFStore(out, 'w') as o:
            offsets = {}  # cv_key -> CV row offset of each batch
            for ck in sorted(set(m0['cv_key'])):
                parts = [_set_file_idx(s[f'cv/{ck}'], b) for b, s in enumerate(stores)]
                offsets[ck] = np.cumsum([0] + [len(p) for p in parts[:-1]])
                cv = pd.concat(parts)
                assert cv.index.is_unique
                o.put(f'cv/{ck}', cv, **_HDF_KW)
                print(f"cv/{ck}: {[len(p) for p in parts]} -> {len(cv)}")
            for g, row in m0.iterrows():
                ck = row['cv_key']
                il = np.concatenate([s[f'cv_iloc/{g}'].to_numpy() + offsets[ck][b]
                                     for b, s in enumerate(stores)])
                o.put(f'cv_iloc/{g}', pd.Series(il.astype(np.int64)), **_HDF_KW)
                for i in range(int(row['n_dv'])):
                    dv = pd.concat([_set_file_idx(s[f'dv/{g}/v{i}'], b) for b, s in enumerate(stores)])
                    assert dv.index.is_unique, (g, i)
                    o.put(f'dv/{g}/v{i}', dv, **_HDF_KW)
                print(f"{g}: cv_iloc {len(il)}, dv rows {len(dv)}")
            meta = m0.copy()
            meta['pot'] = sum(m.loc[m0.index, 'pot'] for m in metas)
            o.put('meta', meta, **_HDF_KW)
            print(meta[['cv_key', 'n_dv', 'pot']])
    finally:
        for s in stores:
            s.close()
    print("Wrote", out)


if __name__ == '__main__':
    merge_detvar_stores(sys.argv[1], sys.argv[2:])
