# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Utility functions for orbital mechanics, coordinate 
#              transformations, numerical integration (RK8), and 
#              stochastic chaser initialization for docking simulations.
# =============================================================================

import numpy as np
from scipy.spatial.transform import Rotation as R_rot

def kep2car(a, e, i, Omega, omega, theta, mu):
    """
    Converts Keplerian elements to ECI Cartesian state.
    """
    cos, sin, sqrt = np.cos, np.sin, np.sqrt
    r_mag = a * (1 - e**2) / (1 + e * cos(theta))
    h = sqrt(mu * a * (1 - e**2))
    
    # Position and velocity in the Perifocal frame
    r_pf = np.array([r_mag * cos(theta), r_mag * sin(theta), 0])
    v_pf = np.array([-sin(theta), e + cos(theta), 0]) * (mu / h)
    
    # Rotation matrix from Perifocal to ECI
    Rot = np.array([
        [cos(Omega)*cos(omega)-sin(Omega)*sin(omega)*cos(i), -cos(Omega)*sin(omega)-sin(Omega)*cos(omega)*cos(i), sin(Omega)*sin(i)],
        [sin(Omega)*cos(omega)+cos(Omega)*sin(omega)*cos(i), -sin(Omega)*sin(omega)+cos(Omega)*cos(omega)*cos(i), -cos(Omega)*sin(i)],
        [sin(omega)*sin(i),                                  cos(omega)*sin(i),                                  cos(i)]
    ])
    
    return np.concatenate((Rot @ r_pf, Rot @ v_pf))

def rk8_step(state, dt, RHS, mu, r_sun, J2, Drag, control):
    """
    Performs a single integration step using the Runge-Kutta-Fehlberg 7(8) method.
    """
    # Butcher Tableau coefficients (RKF78)
    a = [0, 2/27, 1/9, 1/6, 5/12, 0.5, 5/6, 1/6, 2/3, 1/3, 1, 0, 1]
    b = [
        [],
        [2/27],
        [1/36, 1/12],
        [1/24, 0, 1/8],
        [5/12, 0, -25/16, 25/16],
        [1/20, 0, 0, 1/4, 1/5],
        [-25/108, 0, 0, 125/108, -65/27, 125/54],
        [31/300, 0, 0, 0, 61/225, -2/9, 13/900],
        [2, 0, 0, -53/6, 704/45, -107/9, 67/90, 3],
        [-91/108, 0, 0, 23/108, -976/135, 311/54, -19/60, 17/6, -1/12],
        [2383/4100, 0, 0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41],
        [3/205, 0, 0, 0, 0, -6/41, -3/205, -3/41, 3/41, 6/41, 0],
        [-1777/4100, 0, 0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 12/41, 0, 1]
    ]
    c_high = [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 41/840, 0, 0]

    k = []
    for i in range(13):
        s = state.copy()
        for j in range(i):
            s += dt * b[i][j] * k[j]
        k.append(RHS(s, mu, r_sun, J2, Drag, control))

    # 8th order accurate combination
    y_next = state + dt * sum(c * ki for c, ki in zip(c_high, k))
    return y_next

def normalize(v):
    """Returns the unit vector of v."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def random_chaser_state(state_target, theta_max_deg=9, r_min=100, r_max=300):
    """
    Generates a random initial chaser state within a defined approach cone
    relative to the target in the LVLH frame.
    """
    Rot = eci2lvlh(state_target)
    d0 = np.random.uniform(r_min, r_max) / 1000  # Convert to km
    
    # Spherical coordinates within the cone
    theta = np.random.uniform(0, np.radians(theta_max_deg))
    phi = np.random.uniform(0, 2 * np.pi)
    
    # Position in LVLH (radial, along-track, cross-track)
    r0_lvlh = d0 * np.array([np.sin(theta)*np.cos(phi), -np.cos(theta), np.sin(theta)*np.sin(phi)])
    # Velocity perturbation (Gaussian noise)
    v0_lvlh = np.random.normal(0, 0.1, 3) / 1000 
    
    x_rel_lvlh = np.hstack((r0_lvlh, v0_lvlh))
    # Transform relative state back to ECI
    x_rel_eci = np.hstack((Rot.T @ x_rel_lvlh[:3], Rot.T @ x_rel_lvlh[3:]))
    
    return state_target + x_rel_eci

def control_pert(u_ctrl_nom, std_theta_deg=3.0):
    """
    Applies execution errors to the control thrust, including intensity 
    scaling and pointing errors.
    """
    u_ctrl_nom = np.array(u_ctrl_nom, dtype=float)
    u_dist = np.zeros(3)

    for i in range(3):
        ui = u_ctrl_nom[i]
        if ui == 0:
            continue
            
        # Intensity error (Gaussian scaling)
        scale_mag = np.random.uniform(0.96, 1.01)
        ui_pert = ui * scale_mag

        # Pointing error via Rodrigues' rotation formula
        theta = np.radians(np.random.normal(loc=0.0, scale=std_theta_deg))
        axis = np.zeros(3)
        axis[(i+1)%3] = 1.0  # Use a fixed orthogonal axis
        
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_mat = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

        ei = np.zeros(3)
        ei[i] = 1.0
        u_pert_i = R_mat @ ei * ui_pert
        u_dist += u_pert_i

    return u_dist

def sun_pos(day_of_year):
    """
    Calculates the approximate ECI position vector of the Sun.
    """
    # Approximate orbital angle
    theta = 2 * np.pi * (day_of_year / 365.25)
    # Ecliptic obliquity
    epsilon = np.radians(23.44)
    # ECI unit vector coordinates
    x = np.cos(theta)
    y = np.cos(epsilon) * np.sin(theta)
    z = np.sin(epsilon) * np.sin(theta)
    return np.array([x, y, z]) * 1.495978707e8  # 1 AU in km

def eci2lvlh(x_target):
    """
    Returns the rotation matrix from ECI to Hill (LVLH) frame.
    Convention:
      - i: Radial inward (satellite to Earth center)
      - j: Prograde (along velocity vector)
      - k: Normal (opposite to orbital angular momentum)
    """
    r, v = x_target[:3], x_target[3:]
    j = normalize(v)
    k = -normalize(np.cross(r, v))
    i = normalize(np.cross(j, k))

    return np.column_stack([i, j, k])