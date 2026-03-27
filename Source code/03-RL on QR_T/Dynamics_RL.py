# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Implements high-fidelity orbital propagation including J2-J5 
#              geopotential, atmospheric drag (tabular density), Solar 
#              Radiation Pressure (SRP), and 8th-order Runge-Kutta integration.
# =============================================================================

import numpy as np
from Utils_RL_QR import rk8_step

# --- Physical Constants ---
cd = 2.2             # Drag coefficient (typical for a cube satellite)
l = 1.0              # Side length of the cube in meters
rho_sc = 300         # Spacecraft density [kg/m^3]
A_m = 1/(l*rho_sc)   # Area-to-mass ratio [m^2/kg]
rho0 = 1.225         # Sea level density [kg/m^3]
H = 8.5000           # Scale height [km] for exponential atmosphere model
Re = 6378.137        # Earth equatorial radius [km]
G = 6.67430e-20      # Gravitational constant [km^3/kg/s^2]
mu = 398600.4418     # Earth gravitational parameter [km^3/s^2]
S = 1361             # Solar constant [W/m²]
c = 299792458        # Speed of light [m/s]
AU = 1.495978707e8   # Astronomical Unit [km]
Cr = 1.5             # Coefficient of reflectivity

# --- Zonal Harmonic Coefficients (EGM96/JGM-3) ---
J2 = 1.08262668e-3
J3 = -2.5324105e-6
J4 = -1.6198976e-6
J5 = -2.272960828e-7

# --- Atmospheric Density Data (MSISE-90 Tabular) ---
alt_km = np.array([0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
rho = np.array([1.225, 0.4135, 0.04008, 1.03e-3, 1.19e-4, 5.6e-7, 2.07e-9, 5.9e-10,
                2.2e-11, 3.9e-12, 1.0e-12, 3.0e-13, 9.0e-14, 2.5e-14, 7.0e-15, 2.0e-15])

def compute_srp_force(r, r_sun):
    """Calculates acceleration due to Solar Radiation Pressure."""
    r = np.array(r[:3])
    r_rel = r - r_sun                         # Vector from Sun to spacecraft
    r_mag = np.linalg.norm(r_rel)
    r_hat = r_rel / r_mag                     # Unit vector of radiation direction

    P_srp = S / c * (AU / r_mag)**2           # Radiation pressure at current distance
    F_srp = -P_srp * l**2 * Cr * r_hat        # SRP force vector
    a_srp = F_srp / (l**3 * rho_sc)           # Resulting acceleration in m/s^2

    return a_srp / 1000                       # Convert to km/s^2

def drag(r_km, v_kmps):
    """Calculates acceleration due to atmospheric drag."""
    r, v = np.array(r_km), np.array(v_kmps)
    h = max(np.linalg.norm(r) - Re, 0.0)      # Calculate altitude above Earth's radius
    
    if h > 1000: return np.zeros(3)           # Negligible drag above 1000 km
    
    # Linear interpolation in log-space for density
    log_rho = np.interp(h, alt_km, np.log10(rho))
    d = 10**log_rho
    
    v_mps = v * 1000                          # Convert velocity to m/s
    # Drag equation: a = -0.5 * rho * v^2 * cd * A/m
    a_mps2 = -0.5 * d * np.linalg.norm(v_mps) * v_mps * cd * A_m
    return a_mps2 / 1000                      # Convert back to km/s^2

def geopotential(r_vec):
    """Computes the geopotential function including zonal harmonics up to J5."""
    x, y, z = r_vec
    r = np.linalg.norm(r_vec)
    sin_phi = z / r  # Sine of geocentric latitude

    # Legendre Polynomials (P2 to P5)
    P2 = 0.5 * (3 * sin_phi**2 - 1)
    P3 = 0.5 * (5 * sin_phi**3 - 3 * sin_phi)
    P4 = (35 * sin_phi**4 - 30 * sin_phi**2 + 3) / 8
    P5 = (63 * sin_phi**5 - 70 * sin_phi**3 + 15 * sin_phi) / 8

    # Perturbed geopotential potential
    U = -mu / r * (-J2 * (Re / r)**2 * P2 -
                    J3 * (Re / r)**3 * P3 -
                    J4 * (Re / r)**4 * P4 -
                    J5 * (Re / r)**5 * P5)
    return U

def J_acc(r_vec):
    """Numerical calculation of the potential gradient (Central Difference approach)."""
    eps = 1e-1  # Small perturbation step for finite difference
    acc = np.zeros(3)
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = eps
        U_plus = geopotential(r_vec + dr)
        U_minus = geopotential(r_vec - dr)
        acc[i] = -(U_plus - U_minus) / (2 * eps)  # Central derivative
    return acc

def RHS(state, mu, r_sun, J, Drag, control=None):
    """Right-Hand Side of the system of ODEs (Equations of Motion)."""
    x, y, z, vx, vy, vz = state
    r = np.linalg.norm([x, y, z])

    # Central body gravitational acceleration (Two-Body Problem)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3

    # Earth's oblateness perturbation (Zonal Harmonics)
    if J != 0:
        J_pert = J_acc([x, y, z])
        ax += J_pert[0]
        ay += J_pert[1]
        az += J_pert[2]

    # Atmospheric Drag perturbation
    if Drag != 0:
        da = drag([x, y, z], [vx, vy, vz])
        ax += da[0]
        ay += da[1]
        az += da[2]
    
    # Solar Radiation Pressure (SRP)
    if np.linalg.norm(r_sun) != 0:
        a_srp = compute_srp_force([x, y, z], r_sun)
        ax += a_srp[0]
        ay += a_srp[1]
        az += a_srp[2]

    # Control input accelerations (Thrust)
    if control is not None:
        ax += control[0]
        ay += control[1]
        az += control[2]

    # Return the derivative of the state vector [vx, vy, vz, ax, ay, az]
    return np.hstack([vx, vy, vz, ax, ay, az])

def integrate(state, control, dt, mu, r_sun, J=1, Drag=1):
    """Propagates the state vector using an 8th-order Runge-Kutta integrator."""
    return rk8_step(state, dt, RHS, mu, r_sun, J, Drag, control)