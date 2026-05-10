from datetime import datetime
from pathlib import Path

from config import RunConfig
from experiments import compare_policies, sweep_dp_risk_lambda_multi_seed
from results_io import write_json, write_lambda_sweep_csv

if __name__ == "__main__":
    cfg = RunConfig()

    dp_sum, tw_sum, im_sum, pov_sum = compare_policies(
        T=cfg.T,
        Q0=cfg.Q0,
        sigma=cfg.sigma,
        eta=cfg.eta,
        risk_lambda=cfg.risk_lambda,
        S0=cfg.S0,
        gamma=cfg.gamma,
        n_paths=cfg.n_paths,
        seed=cfg.seed,
        pov_alpha=cfg.pov_alpha,
        lam_var=cfg.lam_var,
    )

    print("DP:", dp_sum)
    print("TWAP:", tw_sum)
    print("IMMEDIATE:", im_sum)
    print("POV:", pov_sum)

    rows, best_row = sweep_dp_risk_lambda_multi_seed(
        T=cfg.T,
        Q0=cfg.Q0,
        sigma=cfg.sigma,
        eta=cfg.eta,
        S0=cfg.S0,
        gamma=cfg.gamma,
        n_paths=cfg.n_paths,
        lams=cfg.lams,
        seeds=cfg.seeds,
        k_std=cfg.k_std,
    )

    print(rows)
    print("Best lambda by mean + k*std:", best_row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    comparison_path = out_dir / f"comparison_{timestamp}.json"
    sweep_path = out_dir / f"lambda_sweep_{timestamp}.csv"

    write_json(
        comparison_path,
        {
            "config": cfg.to_dict(),
            "dp": dp_sum,
            "twap": tw_sum,
            "immediate": im_sum,
            "pov": pov_sum,
            "best_lambda_by_mean_plus_k_std": best_row,
        },
    )
    write_lambda_sweep_csv(sweep_path, rows)

    print(f"Wrote comparison JSON to: {comparison_path}")
    print(f"Wrote lambda sweep CSV to: {sweep_path}")
