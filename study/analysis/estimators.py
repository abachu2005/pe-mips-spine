"""Fixed-effect WLS and CR1/t inference for the single spine-only model."""
import math

import numpy as np
from scipy import stats

def codes(values):
    _, inverse = np.unique(np.asarray(values).astype(str), return_inverse=True)
    return inverse


def absorb(matrix, groups, weight, tol=1e-10, max_iter=20000):
    out = np.array(matrix, dtype=float, copy=True)
    if out.ndim == 1:
        out = out[:, None]
    sums = [np.bincount(g, weights=weight) for g in groups]
    for _ in range(max_iter):
        delta = 0.0
        for g, s in zip(groups, sums):
            for j in range(out.shape[1]):
                means = np.bincount(g, weights=weight * out[:, j]) / s
                out[:, j] -= means[g]
                delta = max(delta, float(np.max(np.abs(means))))
        if delta < tol * max(1.0, float(np.max(np.abs(out)))):
            return out
    raise RuntimeError("fixed-effect absorption did not converge")


def fit_fe(y, columns, groups, weight, cluster, treated_cluster_mask):
    resid = absorb(np.column_stack([y] + columns), groups, weight)
    yr, x = resid[:, 0], resid[:, 1:]
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("fixed_effect_design_rank_deficient")
    xtwx = x.T @ (weight[:, None] * x)
    beta = np.linalg.solve(xtwx, x.T @ (weight * yr))
    e = yr - x @ beta
    cl = codes(cluster)
    g = cl.max() + 1
    scores = np.column_stack([np.bincount(cl, weights=weight * x[:, j] * e, minlength=g) for j in range(x.shape[1])])
    bread = np.linalg.inv(xtwx)
    c = g / (g - 1)
    vcov = c * bread @ (scores.T @ scores) @ bread
    treated_clusters = len(set(np.asarray(cluster)[treated_cluster_mask]))
    return {"beta": beta, "vcov": vcov, "se": np.sqrt(np.diag(vcov)), "df": max(treated_clusters - 1, 1),
            "treated_clusters": treated_clusters, "clusters": int(g),
            "_y": yr, "_x": x, "_cluster": cl, "_w": weight, "_c": c}


def summarize(model, index=0):
    beta, se, df = float(model["beta"][index]), float(model["se"][index]), model["df"]
    crit = stats.t.ppf(0.975, df)
    return {"estimate": beta, "se": se, "df": df, "ci": [beta - crit * se, beta + crit * se],
            "p": float(2 * stats.t.sf(abs(beta / se), df)) if se > 0 else None,
            "treated_clusters": model["treated_clusters"], "clusters": model["clusters"]}


def wald(model, indices):
    b = model["beta"][indices]
    v = model["vcov"][np.ix_(indices, indices)]
    stat = float(b @ np.linalg.solve(v, b))
    q = len(indices)
    f = stat / q
    return {"F": f, "df1": q, "df2": model["df"], "p": float(stats.f.sf(f, q, model["df"]))}

