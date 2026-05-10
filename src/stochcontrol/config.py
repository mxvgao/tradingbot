from dataclasses import asdict, dataclass, field


@dataclass
class RunConfig:
    T: int = 50
    Q0: int = 100
    sigma: float = 1.0
    eta: float = 0.01
    risk_lambda: float = 0.1
    S0: float = 100.0
    gamma: float = 0.001
    n_paths: int = 20000
    seed: int = 12
    lam_var: float = 0.002
    k_std: float = 0.5
    pov_alpha: float = 0.1
    seeds: list[int] = field(default_factory=lambda: [11, 12, 13, 14, 15])
    lams: list[float] = field(
        default_factory=lambda: [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
    )

    def to_dict(self):
        return asdict(self)
