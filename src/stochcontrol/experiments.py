from .metrics import add_scores, mean_ci, summarize_costs
from .policies import immediate_policy, pov_policy, solve_optimal_execution_dp, twap_policy
from .simulator import simulate_costs


def compare_policies(
    T,
    Q0,
    sigma,
    eta,
    risk_lambda,
    S0,
    gamma,
    n_paths,
    seed,
    pov_alpha,
    lam_var=None,
):
    dp_policy, _ = solve_optimal_execution_dp(
        T=T, Q0=Q0, sigma=sigma, eta=eta, risk_lambda=risk_lambda
    )
    twap = twap_policy(T=T, Q0=Q0)
    immediate = immediate_policy(T=T, Q0=Q0)
    pov = pov_policy(T=T, Q0=Q0, alpha=pov_alpha)

    dp_costs = simulate_costs(
        u_star=dp_policy,
        T=T,
        Q0=Q0,
        S0=S0,
        sigma=sigma,
        eta=eta,
        gamma=gamma,
        n_paths=n_paths,
        seed=seed,
    )
    twap_costs = simulate_costs(
        u_star=twap,
        T=T,
        Q0=Q0,
        S0=S0,
        sigma=sigma,
        eta=eta,
        gamma=gamma,
        n_paths=n_paths,
        seed=seed,
    )
    immediate_costs = simulate_costs(
        u_star=immediate,
        T=T,
        Q0=Q0,
        S0=S0,
        sigma=sigma,
        eta=eta,
        gamma=gamma,
        n_paths=n_paths,
        seed=seed,
    )
    pov_costs = simulate_costs(
        u_star=pov,
        T=T,
        Q0=Q0,
        S0=S0,
        sigma=sigma,
        eta=eta,
        gamma=gamma,
        n_paths=n_paths,
        seed=seed,
    )

    dp_sum = summarize_costs(dp_costs)
    tw_sum = summarize_costs(twap_costs)
    im_sum = summarize_costs(immediate_costs)
    pov_sum = summarize_costs(pov_costs)

    dp_sum["mean_CI95"] = mean_ci(dp_costs)
    tw_sum["mean_CI95"] = mean_ci(twap_costs)
    im_sum["mean_CI95"] = mean_ci(immediate_costs)
    pov_sum["mean_CI95"] = mean_ci(pov_costs)

    if lam_var is not None:
        dp_sum = add_scores(dp_sum, lam_var=lam_var)
        tw_sum = add_scores(tw_sum, lam_var=lam_var)
        im_sum = add_scores(im_sum, lam_var=lam_var)
        pov_sum = add_scores(pov_sum, lam_var=lam_var)

    return dp_sum, tw_sum, im_sum, pov_sum


def sweep_dp_risk_lambda(T, Q0, sigma, eta, S0, gamma, n_paths, seed, lams):
    rows = []

    for lam in lams:
        pol, _ = solve_optimal_execution_dp(
            T=T, Q0=Q0, sigma=sigma, eta=eta, risk_lambda=lam
        )
        costs = simulate_costs(pol, T, Q0, S0, sigma, eta, gamma, n_paths, seed)
        s = summarize_costs(costs)
        rows.append(
            {
                "risk_lambda": float(lam),
                "mean": float(s["mean"]),
                "std": float(s["std"]),
                "p95": float(s["p95"]),
                "p99": float(s["p99"]),
            }
        )

    return rows


def sweep_dp_risk_lambda_multi_seed(
    T, Q0, sigma, eta, S0, gamma, n_paths, lams, seeds, k_std
):
    rows = []

    for lam in lams:
        means = []
        stds = []
        p95s = []
        p99s = []

        pol, _ = solve_optimal_execution_dp(
            T=T, Q0=Q0, sigma=sigma, eta=eta, risk_lambda=lam
        )

        for seed in seeds:
            costs = simulate_costs(pol, T, Q0, S0, sigma, eta, gamma, n_paths, seed)
            s = summarize_costs(costs)
            means.append(float(s["mean"]))
            stds.append(float(s["std"]))
            p95s.append(float(s["p95"]))
            p99s.append(float(s["p99"]))

        agg_mean = sum(means) / len(means)
        agg_std = sum(stds) / len(stds)
        agg_p95 = sum(p95s) / len(p95s)
        agg_p99 = sum(p99s) / len(p99s)
        score = agg_mean + k_std * agg_std

        rows.append(
            {
                "risk_lambda": float(lam),
                "mean": float(agg_mean),
                "std": float(agg_std),
                "p95": float(agg_p95),
                "p99": float(agg_p99),
                "score_mean_plus_k_std": float(score),
                "n_seeds": int(len(seeds)),
            }
        )

    best_row = min(rows, key=lambda r: r["score_mean_plus_k_std"])
    return rows, best_row
