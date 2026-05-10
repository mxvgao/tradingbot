import numpy as np


class PriceGenerator:
    """Generate price histories for simulation."""

    @staticmethod
    def gbm_path(S0: float, sigma: float, T: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        path = np.empty(T + 1, dtype=float)
        path[0] = S0
        for t in range(T):
            dW = rng.standard_normal()
            path[t + 1] = path[t] * np.exp(-0.5 * sigma ** 2 + sigma * dW)
        return path

    @staticmethod
    def random_walk(S0: float, sigma: float, T: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        path = np.empty(T + 1, dtype=float)
        path[0] = S0
        for t in range(T):
            dW = rng.standard_normal()
            path[t + 1] = path[t] + sigma * dW
        return path


def simulate_costs(
    u_star,
    T,
    Q0,
    S0,
    sigma,
    eta,
    gamma=0.0,
    n_paths=5000,
    seed=None,
    use_gbm=True,
):
    """Monte Carlo simulation of execution cost for an execution schedule."""
    costs = np.zeros(n_paths)
    for k in range(n_paths):
        path = PriceGenerator.gbm_path(S0, sigma, T, seed=(None if seed is None else seed + k)) if use_gbm else PriceGenerator.random_walk(S0, sigma, T, seed=(None if seed is None else seed + k))

        S = path[0]
        Q = Q0
        proceeds = 0.0

        for t in range(T):
            if Q == 0:
                break

            u = int(u_star[t, Q])
            p_exec = S - eta * u
            proceeds += u * p_exec

            Q -= u
            if t + 1 < len(path):
                S = path[t + 1] - gamma * u

        costs[k] = Q0 * S0 - proceeds

    return costs
