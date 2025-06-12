import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D

# =============== PARAMETERE ===============
g = 9.81
alpha = 0.3          # Rotasjonsparameter
h = 1.0              # Dypde
sigma = 1.0          # Overflatespenning
B_10 = 0.01          # Bølgeamplitude x-retning
B_01 = 0.01          # Bølgeamplitude y-retning

# =============== GRUNNLEGGENDE FUNKSJONER ===============
def k_mag(k, l):
    return np.sqrt(k**2 + l**2)

def compute_gamma(k, l):
    return np.sqrt(k_mag(k, l)**2 - alpha**2)

def phi_kl(k, l, z):
    gamma = compute_gamma(k, l)
    if abs(gamma) < 1e-10:
        return z / h
    else:
        return np.sinh(gamma * z) / np.sinh(gamma * h)

def dphi_dz(z, k, l):
    gamma = compute_gamma(k, l)
    if abs(gamma) < 1e-10:
        return 1 / h
    else:
        return gamma * np.cosh(gamma * z) / np.sinh(gamma * h)

def eta(x, y):
    return np.real(B_10 * np.exp(1j * x) + B_01 * np.exp(1j * y))

def lap_eta(x, y):
    return np.real(-B_10 * np.exp(1j * x) - B_01 * np.exp(1j * y))

def U0(z, c1, c2):
    return np.array([c1 * np.cos(alpha * (z - h)) + c2 * np.sin(alpha * (z - h)), 
                     -c1 * np.sin(alpha * (z - h)) + c2 * np.cos(alpha * (z - h)),
                     0])

def u(x, y, z, c1, c2):
    Uh = U0(h, c1, c2)[:2]
    modes = [{'k': np.array([1, 0]), 'phase': x, 'eta_hat': B_10},
             {'k': np.array([0, 1]), 'phase': y, 'eta_hat': B_01}]

    u_total = np.zeros(3, dtype=float)

    for mode in modes:
        k = mode['k']
        phase = mode['phase']
        eta_hat = mode['eta_hat']
        k_perp = np.array([-k[1], k[0]])
        k_dot_Uh = np.dot(k, Uh)
        k_norm2 = np.dot(k, k)

        if k_norm2 < 1e-10:
            continue

        phi = phi_kl(*k, z)
        dphi = dphi_dz(z, *k)
        dphi_h = dphi_dz(h, *k)

        if abs(dphi_h) < 1e-10:
            continue

        exp_term = np.exp(1j * phase)
        u_h = 1j * k_dot_Uh / k_norm2 * (k * dphi + alpha * k_perp * phi) * exp_term
        u_3 = 1j * k_dot_Uh / (k_norm2 * dphi_h) * phi * exp_term
        u_mode = eta_hat * np.concatenate([u_h, [u_3]])
        u_total += np.real(u_mode)

    return u_total

def dynamic_condition(x, y, z, c1, c2):
    u0 = U0(z, c1, c2)
    uf = u(x, y, z, c1, c2)
    return np.dot(u0, uf) + g * eta(x, y) - sigma * lap_eta(x, y)

def system_to_solve(c):
    c1, c2 = c
    eq1 = dynamic_condition(np.pi/2, 0, h, c1, c2)
    eq2 = dynamic_condition(0, np.pi/2, h, c1, c2)
    return [eq1, eq2]

# =============== PARTIKKELBEVEGELSE ===============
def particle_motion(t, pos, c1, c2):
    x, y, z = pos
    velocity = u(x, y, z, c1, c2)
    return velocity

def solve_particle_trajectory(x0, y0, z0, c1, c2, t_span, max_step=0.005):
    initial_pos = [x0, y0, z0]
    sol = solve_ivp(
        lambda t, pos: particle_motion(t, pos, c1, c2),
        t_span, initial_pos, max_step=max_step, dense_output=True,
        rtol=1e-8, atol=1e-10
    )
    return sol

def analyze_trajectory(sol):
    """Analyser banens egenskaper"""
    t_eval = np.linspace(sol.t[0], sol.t[-1], 2000)
    trajectory = sol.sol(t_eval)
    
    x_traj = trajectory[0]
    y_traj = trajectory[1] 
    z_traj = trajectory[2]
    
    print("\nBanens egenskaper:")
    print(f"x-område: [{np.min(x_traj):.2f}, {np.max(x_traj):.2f}] (spenn: {np.max(x_traj)-np.min(x_traj):.2f})")
    print(f"y-område: [{np.min(y_traj):.2f}, {np.max(y_traj):.2f}] (spenn: {np.max(y_traj)-np.min(y_traj):.2f})")
    print(f"z-område: [{np.min(z_traj):.3f}, {np.max(z_traj):.3f}] (spenn: {np.max(z_traj)-np.min(z_traj):.3f})")

def plot_trajectory_analysis(sol):
    """Plot banen for analyse"""
    t_eval = np.linspace(sol.t[0], sol.t[-1], 1500)
    trajectory = sol.sol(t_eval)
    
    fig = plt.figure(figsize=(14, 10))
    
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[0], trajectory[1], trajectory[2], 'b-', linewidth=1.5, alpha=0.8)
    ax.scatter(trajectory[0][0], trajectory[1][0], trajectory[2][0], c='lime', s=100, label='Start')
    ax.scatter(trajectory[0][-1], trajectory[1][-1], trajectory[2][-1], c='red', s=100, label='Slutt')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Partikkelbane i 3D')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_torus_trajectory(sol, R=3, r=1, n_points=2000):
    """Plot banen på en torus med korrekt wrapping"""
    t_eval = np.linspace(sol.t[0], sol.t[-1], n_points)
    trajectory = sol.sol(t_eval)
    
    x_traj = trajectory[0]
    y_traj = trajectory[1]
    z_traj = trajectory[2]
    
    # Modulo 2π for x og y
    x_traj = np.mod(x_traj, 2*np.pi)
    y_traj = np.mod(y_traj, 2*np.pi)
    
    # Finn hopp-punkter
    x_jumps = np.where(np.abs(np.diff(x_traj)) > np.pi)[0] + 1
    y_jumps = np.where(np.abs(np.diff(y_traj)) > np.pi)[0] + 1
    all_jumps = np.unique(np.concatenate((x_jumps, y_jumps)))
    all_jumps = np.sort(all_jumps)
    
    # Del opp i kontinuerlige segmenter
    segments = []
    start_idx = 0
    for jump_idx in all_jumps:
        segments.append((start_idx, jump_idx))
        start_idx = jump_idx
    segments.append((start_idx, len(x_traj)-1))
    
    # Konverter til torus-koordinater
    X_segments = []
    Y_segments = []
    Z_segments = []
    
    for seg_start, seg_end in segments:
        theta = x_traj[seg_start:seg_end]
        phi = y_traj[seg_start:seg_end]
        
        X = (R + r * np.cos(phi)) * np.cos(theta)
        Y = (R + r * np.cos(phi)) * np.sin(theta)
        Z = r * np.sin(phi) + 0.3 * z_traj[seg_start:seg_end]  # Inkluder litt z-variasjon
        
        X_segments.append(X)
        Y_segments.append(Y)
        Z_segments.append(Z)
    
    # Lag figur
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot hvert segment
    for X, Y, Z in zip(X_segments, Y_segments, Z_segments):
        ax.plot(X, Y, Z, 'b-', linewidth=2, alpha=0.8)
    
    # Marker start og slutt
    if X_segments:
        ax.scatter(X_segments[0][0], Y_segments[0][0], Z_segments[0][0], 
                  c='lime', s=100, label='Start')
        ax.scatter(X_segments[-1][-1], Y_segments[-1][-1], Z_segments[-1][-1], 
                  c='red', s=100, label='Slutt')
    
    # Lag torus-grid som referanse
    theta_grid = np.linspace(0, 2*np.pi, 30)
    phi_grid = np.linspace(0, 2*np.pi, 30)
    theta_grid, phi_grid = np.meshgrid(theta_grid, phi_grid)
    
    X_torus = (R + r * np.cos(phi_grid)) * np.cos(theta_grid)
    Y_torus = (R + r * np.cos(phi_grid)) * np.sin(theta_grid)
    Z_torus = r * np.sin(phi_grid)
    
    ax.plot_wireframe(X_torus, Y_torus, Z_torus, color='gray', 
                     alpha=0.15, rstride=2, cstride=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Partikkelbane på Torus (mod 2π)')
    ax.legend()
    
    # Sett visningsområde
    max_range = R + r + 0.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range/2, max_range/2)
    
    plt.tight_layout()
    plt.show()

# =============== HOVEDPROGRAM ===============
if __name__ == "__main__":
    print("PARTIKKELBANE PÅ TORUS")
    print("="*40)
    
    # 1. Løs ligningssystemet for c1 og c2
    print("\nLøser for c1 og c2...")
    sol = root(system_to_solve, [1.0, 1.0], method='hybr')
    
    if not sol.success:
        print(f"Feil: {sol.message}")
        exit()
    
    c1_opt, c2_opt = sol.x
    print(f"Løsning: c1 = {c1_opt:.4f}, c2 = {c2_opt:.4f}")
    
    # 2. Simuler partikkelbane
    print("\nSimulerer partikkelbane...")
    x0, y0, z0 = np.pi/2, np.pi/2, 0.2  # Startposisjon
    t_span = [0, 100]  # Simuleringstid
    
    trajectory_sol = solve_particle_trajectory(x0, y0, z0, c1_opt, c2_opt, t_span)
    
    if not trajectory_sol.success:
        print(f"Feil under simulering: {trajectory_sol.message}")
        exit()
    
    # 3. Analyser og plott
    analyze_trajectory(trajectory_sol)
       
    print("\nPlotter torus-visualisering...")
    plot_torus_trajectory(trajectory_sol)
    
    print("\nFerdig!")