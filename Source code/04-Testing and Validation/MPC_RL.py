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
from Utils_RL_QR import normalize, eci2lvlh

def discretize_system(A, B, dt):
    """
    Discretizes a continuous-time system (A, B) using the matrix exponential method.
    """
    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)

    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd

def mpc_control(x_chaser, x_target, Th, dt, Q, R, mu, h, r, u_max, u_last, v_max):
    """
    MPC Controller for the chaser satellite.

    Args:
        x_chaser: Current ECI state of chaser [x, y, z, vx, vy, vz]
        x_target: Current ECI state of target [x, y, z, vx, vy, vz]
        Th: Prediction time horizon [s]
        dt: Time step for each prediction [s]
        Q: State error weight matrix
        R: Control effort weight matrix
        mu: Gravitational parameter
        h: Specific angular momentum
        r: Target orbital distance
    """

    # Prediction horizon steps
    N = int(Th / dt)
    x = cp.Variable((6, N + 1))  
    u = cp.Variable((3, N))
    scaling_factor = 100 # Numerical scaling for solver stability

    cost = 0
    
    # 1. Coordinate Transformation (ECI -> LVLH)
    Rot = eci2lvlh(x_target)
    x_rel_eci = x_chaser - x_target
    x_rel_lvlh = np.hstack((Rot @ x_rel_eci[:3], Rot @ x_rel_eci[3:]))  

    # Initial state constraint
    constraints = [x[:, 0] == x_rel_lvlh]

    # 2. Continuous-time Linearized Dynamics (Hill-like)
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

    B = 1 / scaling_factor * np.array([
        [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ])

    # Discretization for prediction steps
    Ad, Bd = discretize_system(A, B, dt)

    # 3. Terminal Cost P (DARE calculation with Iterative Fallback)
    try:
        P = solve_discrete_are(Ad, Bd, Q, R)
    except Exception:
        # Fallback: Discrete Riccati iteration if DARE solver fails
        P, eps_reg = Q.copy(), 1e-8
        for _ in range(100):
            M = R + Bd.T @ P @ Bd
            # Check matrix conditioning
            if not np.isfinite(np.linalg.cond(M)) or np.linalg.cond(M) > 1e12:
                M += eps_reg * np.eye(M.shape[0])
            try:
                K = np.linalg.solve(M, Bd.T @ P @ Ad)
            except Exception as e:
                print(f"❌ Riccati fallback failed: {e}")
                P = Q.copy()
                break
            P_next = Q + Ad.T @ (P - P @ Bd @ K) @ Ad
            if np.allclose(P, P_next, atol=1e-6): 
                break
            P = P_next
    
    # Ensure symmetry
    P = 0.5 * (P + P.T)

    # 4. Reference Frame and Cone Geometry
    docking_axis = np.array([0, -1, 0]) # Assuming -Y as approach axis in LVLH
    x_axis = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])
    tan_alpha = 0.125 * (1 - 0.1) # Cone aperture
    
    cone_slack_weight = 1e6
    eps_z = cp.Variable(N, nonneg=True) # Slack for Z-axis cone constraint
    eps_y = cp.Variable(N, nonneg=True) # Slack for X-axis cone constraint

    # 5. Optimization Loop
    for k in range(N):
        # Dynamics consistency
        constraints += [x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k]]

        # Quadratic weights
        cost += cp.quad_form(x[:, k], Q)
        cost += cp.quad_form(u[:, k], R)

        # Projections for constraints
        y_k = docking_axis @ x[0:3, k]
        z_perp = z_axis @ x[0:3, k]
        x_perp = x_axis @ x[0:3, k]
        vz_perp = z_axis @ x[3:, k]
        vx_perp = x_axis @ x[3:, k]

        # Geometry and Velocity constraints with Slacks
        constraints += [
            10*z_perp <= 10*tan_alpha * y_k + eps_z[k],
            10*z_perp >= -10*tan_alpha * y_k - eps_z[k],
            10*x_perp <= 10*tan_alpha * y_k + eps_y[k],
            10*x_perp >= -10*tan_alpha * y_k - eps_y[k],
            cp.abs(docking_axis @ x[3:6, k]) <= 5 * v_max,
            cp.abs(vz_perp) <= 2 * v_max,
            cp.abs(vx_perp) <= 2 * v_max
        ]

        # Cost function penalties (Slacks + Performance)
        cost += cone_slack_weight * (eps_z[k] + eps_y[k])
        cost += np.linalg.norm(Q[:3,:3]) * y_k**2
        cost += np.linalg.norm(Q[:3,:3]) * 3 * (z_perp**2 + x_perp**2) 

    # Terminal cost and final constraints
    cost += cp.quad_form(x[:, N], P) * N
    constraints += [cp.abs(docking_axis @ x[3:6, N]) * 100 <= 100 * 5 * v_max]
    constraints += [cp.abs(u) <= u_max * scaling_factor]

    # 6. Solve Problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    u1 = None  

    try:
        # First attempt: nominal precision
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
            raise ValueError(f"OSQP baseline failed: {problem.status}")

    except Exception:
        # Second attempt: high tolerance/robust fallback
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
            print("🚫 OSQP found no solution: returning null control.")
            return None

    # Return control acceleration converted back to ECI frame
    return (Rot.T @ u1 / scaling_factor)