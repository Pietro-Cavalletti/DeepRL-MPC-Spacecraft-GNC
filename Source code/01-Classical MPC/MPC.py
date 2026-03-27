# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Model Predictive control using CVXPY and OSQP solver. 
#              Includes linearized HCW dynamics, terminal cost calculation via 
#              Discrete Algebraic Riccati Equation (DARE) with iterative 
#              fallback, and docking cone constraints with slack variables.
# =============================================================================

import numpy as np
import cvxpy as cp
from scipy.linalg import expm, solve_discrete_are
from Utils import normalize, eci2lvlh

# --- SYSTEM DISCRETIZATION ---

def discretize_system(A, B, dt):
    """
    Discretizes a continuous linear system (A, B) with time step dt.
    Using the matrix exponential approach for state-space discretization.
    """
    n = A.shape[0]
    m = B.shape[1]

    # Augmented matrix for simultaneous discretization of A and B
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)

    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd

# --- MPC CONTROLLER ---

def mpc_control(x_chaser, x_target, Th, dt, Q, R, mu, h, r, u_max, u_last, v_max):
    """
    MPC Controller for the chaser satellite rendezvous.

    Inputs:
        x_chaser: Current ECI state of the chaser [x, y, z, vx, vy, vz]
        x_target: Current ECI state of the target [x, y, z, vx, vy, vz]
        Th:       Prediction horizon time [s]
        dt:       Time step for each prediction [s]
        Q, R:     Weight matrices for state error and control effort
        mu, h, r: Gravitational parameter, specific angular momentum, and orbital radius
        u_max:    Maximum thrust constraint
        v_max:    Velocity constraints for the approach
    """

    # Prediction horizon setup
    N = int(Th / dt)
    x = cp.Variable((6, N+1))  # State trajectory: [x, y, z, vx, vy, vz]
    u = cp.Variable((3, N))    # Control trajectory: [ax, ay, az]
    scaling_factor = 100       # Internal scaling for numerical stability

    # --- COORDINATE TRANSFORMATION ---
    cost = 0
    Rot = eci2lvlh(x_target)
    x_rel_eci = x_chaser - x_target
    # Convert relative state from ECI to LVLH frame
    x_rel_lvlh = np.hstack((Rot @ x_rel_eci[:3], Rot @ x_rel_eci[3:]))  

    # Initial state constraint
    constraints = [x[:, 0] == x_rel_lvlh]

    # --- CONTINUOUS SYSTEM MATRICES (Relative Motion) ---
    k1 = 2 * h * (np.dot(x_target[1:3], x_target[4:6])) / (r**4)
    k2 = 2 * h / r**2
    
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [2*mu/r**3 + h**2/r**4, -k1, 0, 0, k2, 0],
        [k1, -mu/r**3 + h**2/r**4, 0, -k2, 0, 0],
        [0, 0, -mu/r**3, 0, 0, 0]
    ])

    # Control matrix B (normalized by scaling factor)
    B = 1/scaling_factor * np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Discretize dynamics
    Ad, Bd = discretize_system(A, B, dt)

    # Solve Discrete Algebraic Riccati Equation for terminal weight P
    P = solve_discrete_are(Ad, Bd, Q, R)

    # --- DOCKING CONE AND GEOMETRY ---
    docking_axis = np.array([0, -1, 0]) # Defined axis in LVLH
    x_axis = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])
    tan_alpha = 0.125 * (1 - 0.1)       # Cone aperture slope
    cone_slack_weight = 1e6

    # Slack variables for docking cone constraints to ensure feasibility
    eps_z = cp.Variable(N, nonneg=True)
    eps_y = cp.Variable(N, nonneg=True)

    # --- OPTIMIZATION LOOP OVER HORIZON ---
    for k in range(N):
        # Linear dynamics constraint
        constraints += [x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k]]

        # Running cost: state error and control effort
        cost += cp.quad_form(x[:, k], Q)
        cost += cp.quad_form(u[:, k], R)

        # Projections on docking axes
        y_k = docking_axis @ x[0:3, k]
        z_perp = z_axis @ x[0:3, k]
        x_perp = x_axis @ x[0:3, k]
        vz_perp = z_axis @ x[3:, k]
        vx_perp = x_axis @ x[3:, k]

        # Safety constraints (Docking Cone and Velocity Limits)
        constraints += [
            10 * z_perp <= 10 * tan_alpha * y_k + eps_z[k],
            10 * z_perp >= -10 * tan_alpha * y_k - eps_z[k],
            10 * x_perp <= 10 * tan_alpha * y_k + eps_y[k],
            10 * x_perp >= -10 * tan_alpha * y_k - eps_y[k],
            cp.abs(docking_axis @ x[3:6, k]) <= 5 * v_max,
            cp.abs(vz_perp) <= 2 * v_max,
            cp.abs(vx_perp) <= 2 * v_max
        ]

        # Penalty for using slack variables and additional state weights
        cost += cone_slack_weight * (eps_z[k] + eps_y[k])
        cost += np.linalg.norm(Q[:3,:3]) * y_k**2
        cost += np.linalg.norm(Q[:3,:3]) * 3 * (z_perp**2 + x_perp**2) 

    # --- TERMINAL CONDITIONS AND CONSTRAINTS ---
    cost += cp.quad_form(x[:, N], P) * N

    constraints += [cp.abs(docking_axis @ x[3:6, N]) * 100 <= 100 * 5 * v_max]
    constraints += [cp.abs(u) <= u_max * scaling_factor]

    # --- SOLVER EXECUTION ---
    problem = cp.Problem(cp.Minimize(cost), constraints)
    u1 = None  

    try:
        # Initial solve attempt with OSQP
        result = problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            max_iter=70_000,
            eps_abs=1e-1,
            eps_rel=1e-3
        )

        if problem.status in ["optimal", "optimal_inaccurate"] and u.value is not None:
            u1 = u.value[:, 0]
        else:
            raise ValueError(f"OSQP found no solution: {problem.status}")

    except Exception:
        # Retry with relaxed tolerances if initial attempt fails
        result = problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            max_iter=1_000_000,
            eps_abs=1e0,
            eps_rel=1e0
        )

        if problem.status in ["optimal", "optimal_inaccurate"] and u.value is not None:
            u1 = u.value[:, 0]
        else:
            print("🚫 No solution available: return null control.")
            return None

    # Return the first control action transformed back to ECI frame
    return (Rot.T @ u1 / scaling_factor)