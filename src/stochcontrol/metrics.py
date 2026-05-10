import numpy as np


def summarize_costs(costs):
    mean = costs.mean()
    std = costs.std(ddof=1)
    p95 = np.quantile(costs, 0.95)
    p99 = np.quantile(costs, 0.99)

    # distribution diagnostics
    centered = costs - mean
    if std == 0.0:
        skew = 0.0
        kurt = 0.0
    else:
        skew = (centered**3).mean() / std**3
        kurt = (centered**4).mean() / std**4 - 3  # excess kurtosis

    return {
        "mean": float(mean),
        "std": float(std),
        "p95": float(p95),
        "p99": float(p99),
        "skew": float(skew),
        "excess_kurtosis": float(kurt),
    }


def add_scores(summary, lam_var=None, k_std=None):
    mean = summary["mean"]
    std = summary["std"]
    var = std**2

    if lam_var is not None:
        summary["score_mean_plus_lam_var"] = mean + lam_var * var
    if k_std is not None:
        summary["score_mean_plus_k_std"] = mean + k_std * std
    return summary


def mean_ci(costs, alpha=0.05):
    mean = costs.mean()
    std = costs.std(ddof=1)
    N = len(costs)
    # Normal approximation; alpha retained for future extension.
    _ = alpha
    z = 1.96  # ~95% CI
    half = z * std / np.sqrt(N)
    return float(mean - half), float(mean + half)
