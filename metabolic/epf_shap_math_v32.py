"""Vectorised antithetic Shapley paths, extracted unchanged from v30_explain.

The extraction removes historical experiment imports from the v32 package.
The numerical calculation and return dtypes are unchanged.
"""
from __future__ import annotations

import numpy as np


def shapley_fold(f, raw: np.ndarray, rows: np.ndarray, perms: np.ndarray,
                 background: np.ndarray, chunk_rows: int) -> dict:
    n, m, g = perms.shape
    ranks = np.argsort(perms, axis=2)
    t = np.arange(g + 1)
    phi = sum_sq = None
    fx_all, fz_all = [], []
    for start in range(0, n, chunk_rows):
        sl = slice(start, min(start + chunk_rows, n))
        r = rows[sl]
        x = raw[r]
        z = raw[background[sl]]
        mask = ranks[sl][:, :, None, :] < t[None, None, :, None]
        hybrid = np.where(mask, x[:, None, None, :], z[:, :, None, :])
        c = len(r)
        values = f(hybrid.reshape(-1, g))
        s, _, k = values.shape
        values = values.reshape(s, c, m, g + 1, k)
        steps = np.diff(values, axis=3)
        contrib = np.zeros((s, c, m, g, k))
        idx = perms[sl]
        np.put_along_axis(contrib, idx[None, :, :, :, None].repeat(s, 0).repeat(k, 4), steps, axis=3)
        pair = (contrib[:, :, : m // 2] + contrib[:, :, m // 2:]) / 2
        block_phi = contrib.mean(axis=2)
        block_var = pair.var(axis=2, ddof=1) / (m // 2)
        phi = block_phi if phi is None else np.concatenate([phi, block_phi], axis=1)
        sum_sq = block_var if sum_sq is None else np.concatenate([sum_sq, block_var], axis=1)
        fx_all.append(values[:, :, 0, g, :])
        fz_all.append(values[:, :, :, 0, :].mean(axis=2))
    fx = np.concatenate(fx_all, axis=1)
    fz = np.concatenate(fz_all, axis=1)
    additivity = np.abs(phi.sum(axis=2) - (fx - fz)).max()
    return {'phi':phi.astype(np.float32),'mc_var':sum_sq.astype(np.float32),'fx':fx.astype(np.float32),
            'f_background_mean':fz.astype(np.float32),'additivity_max_abs_error':float(additivity)}
