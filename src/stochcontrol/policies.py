import numpy as np

def solve_optimal_execution_dp(T, Q0, sigma, eta, risk_lambda):
    # Value function: V[t, Q]
    V = np.full((T + 1, Q0 + 1), np.inf)
    V[T, 0] = 0.0 # terminal condition: must end with zero inventory

    # Optimal control
    u_star = np.zeros((T, Q0 + 1), dtype=int)

    risk_coeff = risk_lambda * (sigma ** 2)

    # Backward induction
    for t in range(T - 1, -1, -1):
        V_next = V[t + 1]

        for Q in range(Q0 + 1):
            best_cost = float("inf")
            best_u = 0
            
            u_star[t, Q] = Q

            for u in range(Q + 1):
                cost = eta * u**2  + V_next[Q - u]
                if best_cost >= cost:
                    best_cost = cost
                    best_u = u
            
            u_star[t, Q] = best_u
            V[t, Q] = best_cost + risk_coeff * (Q ** 2)

    return u_star, V

def twap_policy(T, Q0):
    u = np.zeros((T, Q0 + 1), dtype=int)
    for t in range(T):
        for Q in range(Q0 + 1):
            # Ceil-based slicing avoids zero trades early and guarantees liquidation.
            remaining_steps = T - t
            u[t, Q] = (Q + remaining_steps - 1) // remaining_steps
    return u


def immediate_policy(T, Q0):
    u = np.zeros((T, Q0 + 1), dtype=int)
    for Q in range(Q0 + 1):
        u[0, Q] = Q
    return u


def pov_policy(T, Q0, alpha):
    u = np.zeros((T, Q0 + 1), dtype=int)
    for t in range(T):
        for Q in range(Q0 + 1):
            if t == T - 1:
                u[t, Q] = Q
            else:
                # Sell a fixed fraction of remaining inventory each step.
                trade = int(np.ceil(alpha * Q))
                u[t, Q] = min(Q, trade)
    return u
