# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Model Predictive Control (MPC) for satellite docking.
#              Uses linearized Hill-Clohessy-Wiltshire (HCW) equations with
#              linear constraints for the docking cone and velocity limits.
# =============================================================================

import numpy as np
import cvxpy as cp
from scipy.linalg import expm, solve_discrete_are
from Utils_RL import normalize, eci2lvlh

def discretize_system(A, B, dt):
    """
    Discretizes a continuous state-space system (A, B) using matrix exponential.
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
    
    Returns:
        u_eci: Control acceleration vector in ECI frame [km/s^2]
    """
    # Prediction horizon based on Time Horizon (Th) and time step (dt)
    N = int(Th / dt)
    x = cp.Variable((6, N + 1))  
    u = cp.Variable((3, N))
    
    # Scaling factor to improve numerical stability in the solver
    scaling_factor = 100

    # 1. State Transformation (ECI -> LVLH)
    Rot = eci2lvlh(x_target)
    x_rel_eci = x_chaser - x_target
    # Current relative state in LVLH frame
    x_rel_lvlh = np.hstack((Rot @ x_rel_eci[:3], Rot @ x_rel_eci[3:]))  

    # Initial condition constraint
    constraints = [x[:, 0] == x_rel_lvlh]

    # 2. Linearized Dynamics (HCW-like with varying orbital parameters)
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

    B = (1/scaling_factor) * np.array([
        [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ])

    # Discretize for the current dt
    Ad, Bd = discretize_system(A, B, dt)

    # 3. Terminal Cost (Discrete Algebraic Riccati Equation)
    try:
        P = solve_discrete_are(Ad, Bd, Q, R)
    except Exception:
        P = Q.copy() # Fallback to state weight

    # 4. Docking Cone and Corridor Constraints
    # Axis definition: Y is the approach axis (radial/prograde depending on LVLH convention)
    docking_axis = np.array([0, -1, 0])
    x_axis = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])
    
    tan_alpha = 0.125 * (1 - 0.1) # Cone slope
    cone_slack_weight = 1e6
    
    # Soft constraints (slack variables) to avoid infeasibility
    eps_z = cp.Variable(N, nonneg=True)
    eps_y = cp.Variable(N, nonneg=True)

    cost = 0
    for k in range(N):
        # Dynamics constraint
        constraints += [x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k]]

        # Running cost
        cost += cp.quad_form(x[:, k], Q)
        cost += cp.quad_form(u[:, k], R)

        # Geometry extraction
        pos_y = docking_axis @ x[0:3, k]
        pos_z = z_axis @ x[0:3, k]
        pos_x = x_axis @ x[0:3, k]
        
        vel_y = docking_axis @ x[3:6, k]
        vel_z = z_axis @ x[3:6, k]
        vel_x = x_axis @ x[3:6, k]

        # Corridor Constraints (Keep chaser inside the approach cone)
        constraints += [
            10 * pos_z <=  10 * tan_alpha * pos_y + eps_z[k],
            10 * pos_z >= -10 * tan_alpha * pos_y - eps_z[k],
            10 * pos_x <=  10 * tan_alpha * pos_y + eps_y[k],
            10 * pos_x >= -10 * tan_alpha * pos_y - eps_y[k],
            cp.abs(vel_y) <= 5 * v_max,
            cp.abs(vel_z) <= 2 * v_max,
            cp.abs(vel_x) <= 2 * v_max
        ]

        # Penalize slack usage and favor approach along the centerline
        cost += cone_slack_weight * (eps_z[k] + eps_y[k])
        cost += np.linalg.norm(Q[:3,:3]) * (pos_y**2 + 3 * (pos_z**2 + pos_x**2))

    # 5. Final Constraints and Solver Call
    cost += cp.quad_form(x[:, N], P) * N
    constraints += [cp.abs(u) <= u_max * scaling_factor]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    
    try:
        # High precision attempt
        problem.solve(solver=cp.OSQP, warm_start=True, max_iter=70000, eps_abs=1e-1, eps_rel=1e-3)
        
        if problem.status not in ["optimal", "optimal_inaccurate"] or u.value is None:
            # Low precision fallback
            problem.solve(solver=cp.OSQP, warm_start=True, max_iter=1000000, eps_abs=1.0, eps_rel=1.0)

        if u.value is not None:
            u_lvlh = u.value[:, 0] / scaling_factor
            return Rot.T @ u_lvlh # Back to ECI frame
        
    except Exception as e:
        print(f"MPC Solver Error: {e}")
    
    return None