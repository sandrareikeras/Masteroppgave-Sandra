import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fft2, fftshift

# =============== PARAMETERE ===============
g = 9.81
alpha = 0.3          # Rotasjonsparameter
h = 1.0              # Dypde
sigma = 1.0          # Overflatespenning
B_10 = 0.01          # Bølgeamplitude x-retning
B_01 = 0.01          # Bølgeamplitude y-retning

# =============== GRUNNLEGGENDE FUNKSJONER ===============
def k_mag(k, l):
    '''
    Find the magnitude of the wavenumber vector
    Input
        - k: wavenumber in x-direction
        - l: wavenumber in y-direction
    Returns
        - Magnitude of the wavenumber vector
    '''
    return np.sqrt(k**2 + l**2)

def compute_gamma(k, l):
    '''
    Compute the gamma value based on the wavenumbers k and l
    Input
        - k: wavenumber in x-direction
        - l: wavenumber in y-direction
    Returns
        - gamma: computed value based on the wavenumbers
    '''
    return np.sqrt(k_mag(k, l)**2 - alpha**2)

def phi_kl(k, l, z):
    '''
    Vertical shape function for the velocity field
    Input
        - k: wavenumber in x-direction
        - l: wavenumber in y-direction
        - z: height above the bottom
    Returns
        - phi: vertical shape function value at height z
    '''
    gamma = compute_gamma(k, l)
    if abs(gamma) < 1e-10:
        return z / h
    else:
        return np.sinh(gamma * z) / np.sinh(gamma * h)

def dphi_dz(z, k, l):
    '''
    Derivative of the vertical shape function with respect to z
    Input
        - z: height above the bottom
        - k: wavenumber in x-direction
        - l: wavenumber in y-direction
    Returns
        - dphi: derivative of the vertical shape function at height z
    '''
    gamma = compute_gamma(k, l)
    if abs(gamma) < 1e-10:
        return 1 / h
    else:
        return gamma * np.cosh(gamma * z) / np.sinh(gamma * h)

def eta(x, y):
    '''
    Compute the surface elevation eta at a given position using known Fourier coefficients B_10 and B_01
    Input
        - x: position in x-direction
        - y: position in y-direction
    Returns
        - eta: surface elevation at position (x, y)
    '''
    return np.real(B_10 * np.exp(1j * x) + B_01 * np.exp(1j * y))

def lap_eta(x, y):
    '''
    Compute the Laplacian of the surface elevation eta at a given position
    Input
        - x: position in x-direction
        - y: position in y-direction
    Returns
        - lap_eta: Laplacian of the surface elevation at position (x, y)
    '''
    return np.real(-B_10 * np.exp(1j * x) - B_01 * np.exp(1j * y))

def U0(z, c1, c2):
    '''
    Compute the initial velocity field at a given height z using Fourier coefficients c1 and c2
    Input
        - z: height above the bottom
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
    Returns
        - U0: velocity field at height z as a numpy array 
    '''
    return np.array([c1 * np.cos(alpha * (z - h)) + c2 * np.sin(alpha * (z - h)), 
                     -c1 * np.sin(alpha * (z - h)) + c2 * np.cos(alpha * (z - h)),
                     0])

def u(x, y, z, c1, c2):
    '''
    Compute the velocity field at a given position (x, y, z) using Fourier coefficients c1 and c2
    Input
        - x: position in x-direction
        - y: position in y-direction
        - z: height above the bottom
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
    Returns
        - u_total: total velocity field at position
    '''
    Uh = U0(h, c1, c2)[:2]
    modes = [{'k': np.array([1, 0]), 'phase': x, 'eta_hat': B_10},
             {'k': np.array([0, 1]), 'phase': y, 'eta_hat': B_01}]

    u_total = np.zeros(3, dtype=float)

    # Iterate over each mode
    for mode in modes:
        k = mode['k']
        phase = mode['phase']
        eta_hat = mode['eta_hat']
        k_perp = np.array([-k[1], k[0]])
        k_dot_Uh = np.dot(k, Uh)
        k_norm2 = np.dot(k, k)

        if k_norm2 < 1e-10:         # Avoid division by zero
            continue

        phi = phi_kl(*k, z)
        dphi = dphi_dz(z, *k)
        dphi_h = dphi_dz(h, *k)

        if abs(dphi_h) < 1e-10:    # Avoid division by zero
            continue

        exp_term = np.exp(1j * phase)
        u_h = 1j * k_dot_Uh / k_norm2 * (k * dphi + alpha * k_perp * phi) * exp_term
        u_3 = 1j * k_dot_Uh / (k_norm2 * dphi_h) * phi * exp_term
        u_mode = eta_hat * np.concatenate([u_h, [u_3]])
        u_total += np.real(u_mode)

    return u_total

def dynamic_condition(x, y, z, c1, c2):
    '''
    Compute the dynamic condition for the velocity field at a given position (x, y, z) using Fourier coefficients c1 and c2
    Input
        - x: position in x-direction
        - y: position in y-direction
        - z: height above the bottom
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
    Returns
        - dynamic_condition: value of the dynamic condition at position (x, y, z)
    '''
    u0 = U0(z, c1, c2)
    uf = u(x, y, z, c1, c2)
    return np.dot(u0, uf) + g * eta(x, y) - sigma * lap_eta(x, y)

def system_to_solve(c):
    '''
    System of equations to solve for c1 and c2
    Input
        - c: array containing coefficients c1 and c2
    Returns
        - List of equations to be solved
    '''
    c1, c2 = c
    eq1 = dynamic_condition(np.pi/2, 0, h, c1, c2)
    eq2 = dynamic_condition(0, np.pi/2, h, c1, c2)
    return [eq1, eq2]

# =============== PARTIKKELBEVEGELSE ===============
def particle_motion(t, pos, c1, c2):
    '''
    Define the particle motion in a moving reference frame
    Input
        - t: time (not used, but required by solve_ivp)
        - pos: current position of the particle as a list [x, y, z]
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
    Returns
        - velocity: velocity of the particle at the current position
    '''
    x, y, z = pos
    velocity = u(x, y, z, c1, c2)
    return velocity

def solve_particle_trajectory(x0, y0, z0, c1, c2, t_span, max_step=0.005):
    '''
    Solve the particle trajectory using the initial position and coefficients c1, c2
    Input
        - x0: initial x-position
        - y0: initial y-position
        - z0: initial z-position
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
        - t_span: time span for the simulation as a list [start_time, end_time]
        - max_step: maximum step size for the solver
    Returns
        - sol: solution object containing the particle trajectory
    '''
    sol = solve_ivp( lambda t, pos: particle_motion(t, pos, c1, c2), t_span, [x0, y0, z0], 
                    max_step=max_step, dense_output=True, rtol=1e-8, atol=1e-10)
    return sol

def analyze_trajectory(sol):
    '''
    Analyze the trajectory and print its properties
    Input
        - sol: solution object containing the particle trajectory
    Returns
        - Prints the trajectory properties
    '''
    t_eval = np.linspace(sol.t[0], sol.t[-1], 2000)
    trajectory = sol.sol(t_eval)
    
    x_traj = trajectory[0]
    y_traj = trajectory[1] 
    z_traj = trajectory[2]
    
    print("Paricle trajectory analysis:")
    print(f"x-area: [{np.min(x_traj):.2f}, {np.max(x_traj):.2f}] (range: {np.max(x_traj)-np.min(x_traj):.2f})")
    print(f"y-area: [{np.min(y_traj):.2f}, {np.max(y_traj):.2f}] (range: {np.max(y_traj)-np.min(y_traj):.2f})")
    print(f"z-are: [{np.min(z_traj):.3f}, {np.max(z_traj):.3f}] (range: {np.max(z_traj)-np.min(z_traj):.3f})")

def plot_trajectory_analysis(sol):
    '''
    Plot the trajectory in 3D space with start and end points highlighted
    Input
        - sol: solution object containing the particle trajectory
    Returns
        - 3D plot of the particle trajectory
    '''
    t_eval = np.linspace(sol.t[0], sol.t[-1], 1500)
    trajectory = sol.sol(t_eval)
    
    fig = plt.figure(figsize=(14, 10))
    
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[0], trajectory[1], trajectory[2], 'b-', linewidth=1.5, alpha=0.8)
    ax.scatter(trajectory[0][0], trajectory[1][0], trajectory[2][0], c='lime', s=100, label='Start')
    ax.scatter(trajectory[0][-1], trajectory[1][-1], trajectory[2][-1], c='red', s=100, label='End')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Particletrajectory in 3D')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_torus_trajectory(sol, R=3, r=1, n_points=2000):
    '''
    Plot a torus for comparison
    Input
        - sol: solution object from solve_ivp
        - R: major radius of the torus
        - r: minor radius of the torus
        - n_points: number of points to evaluate the trajectory
    Returns
        - 3D plot of the particle trajectory on a torus
    '''
    t_eval = np.linspace(sol.t[0], sol.t[-1], n_points)
    trajectory = sol.sol(t_eval)
    
    x_traj = trajectory[0]
    y_traj = trajectory[1]
    z_traj = trajectory[2]
    
    # Modulo 2π for x og y
    x_traj = np.mod(x_traj, 2*np.pi)
    y_traj = np.mod(y_traj, 2*np.pi)
    
    # Find jump-points
    x_jumps = np.where(np.abs(np.diff(x_traj)) > np.pi)[0] + 1
    y_jumps = np.where(np.abs(np.diff(y_traj)) > np.pi)[0] + 1
    all_jumps = np.unique(np.concatenate((x_jumps, y_jumps)))
    all_jumps = np.sort(all_jumps)
    
    # Split trajectory into segments based on jumps
    segments = []
    start_idx = 0

    # Iterate through all jumps and create segments
    for jump_idx in all_jumps:
        segments.append((start_idx, jump_idx))
        start_idx = jump_idx
    segments.append((start_idx, len(x_traj)-1))
    
    # Convert segments to 3D coordinates on the torus
    X_segments = []
    Y_segments = []
    Z_segments = []
    
    # Iterate through segments and convert to 3D coordinates
    for seg_start, seg_end in segments:
        theta = x_traj[seg_start:seg_end]
        phi = y_traj[seg_start:seg_end]
        
        X = (R + r * np.cos(phi)) * np.cos(theta)
        Y = (R + r * np.cos(phi)) * np.sin(theta)
        Z = r * np.sin(phi) + 0.3 * z_traj[seg_start:seg_end]  # Inkluder litt z-variasjon
        
        X_segments.append(X)
        Y_segments.append(Y)
        Z_segments.append(Z)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 
    for X, Y, Z in zip(X_segments, Y_segments, Z_segments):
        ax.plot(X, Y, Z, 'b-', linewidth=2, alpha=0.8)
    
    # Mark start and end points
    if X_segments:
        ax.scatter(X_segments[0][0], Y_segments[0][0], Z_segments[0][0], 
                  c='lime', s=100, label='Start')
        ax.scatter(X_segments[-1][-1], Y_segments[-1][-1], Z_segments[-1][-1], 
                  c='red', s=100, label='End')
    
    # Plot torus surface as wireframe
    theta_grid = np.linspace(0, 2*np.pi, 30)
    phi_grid = np.linspace(0, 2*np.pi, 30)
    theta_grid, phi_grid = np.meshgrid(theta_grid, phi_grid)
    
    X_torus = (R + r * np.cos(phi_grid)) * np.cos(theta_grid)
    Y_torus = (R + r * np.cos(phi_grid)) * np.sin(theta_grid)
    Z_torus = r * np.sin(phi_grid)
    
    ax.plot_wireframe(X_torus, Y_torus, Z_torus, color='gray', alpha=0.15, rstride=2, cstride=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Particletrajcetory (mod 2π)')
    ax.legend()
    
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
    x0, y0, z0 = np.pi/2, np.pi/2, 0.5  # Startposisjon
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







# ---------- SECOND ORDER -----------------
def find_coefficients_second_order(c1, c2):
    '''
    Find the Fourier coefficients for the second order surface elevation eta^(2)
    Input
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
    Returns
        - Dictionary with Fourier coefficients
    '''
    N = 100
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Beregn eta^(1)
    eta1 = np.real(B_10 * np.exp(1j * X) + B_01 * np.exp(1j * Y))

    # Beregn u^(1)_h = (u1_x, u1_y) ved z = h over rutenett
    u1 = np.zeros((3, N, N))
    for i in range(N):
        for j in range(N):
            u1[:, i, j] = u(X[i, j], Y[i, j], h, c1, c2)

    u1_h = u1[:2]  # kun horisontale komponenter

    # Beregn |u1_h|^2
    u1_norm_sq = u1_h[0]**2 + u1_h[1]**2

    # Beregn d/dz(U · u1_h) ved z = h
    dUdz = alpha * np.array([-c1 * np.sin(0) + c2 * np.cos(0),
            -c1 * np.cos(0) - c2 * np.sin(0)])
    dUu1_dz = dUdz[0] * u1_h[0] + dUdz[1] * u1_h[1]

    # Sett sammen høyresiden
    rhs = 0.5 * u1_norm_sq + eta1 * dUu1_dz

    # FFT og projeksjon
    rhs_hat = fftshift(fft2(rhs)) / (N**2)
    def idx(kx, ky): return N//2 + kx, N//2 + ky

    return { "B_20": np.real(rhs_hat[idx(2, 0)]),
            "B_11": np.real(rhs_hat[idx(1, 1)]),
            "B_m11": np.real(rhs_hat[idx(1, -1)]),
            "B_02": np.real(rhs_hat[idx(0, 2)]),
            "B_00": np.real(rhs_hat[idx(0, 0)]),}

a_vals = find_coefficients_second_order(c1_opt, c2_opt)
print("Second order koeffisienter:", a_vals)


