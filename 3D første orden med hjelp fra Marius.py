import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fft2, fftshift

# ---- PARAMETERS -------
g = 9.81                  # Gravitational acceleration 
h = 1.0                   # Water depth 
alpha = 0.3               # Beltrami constant
sigma = 1.0               # Surface tension
t_span = [0, 200]         # Time span for the simulation

# -------- FIRST ORDER ------------------------
B_10 = 0.01          # Wave amplitude x-direction
B_01 = 0.01          # Wave amplitude y-direction

# ------------ FIRST ORDER -------------
# -------- FUNCTIONS ------------------------
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
        u_h = - k_dot_Uh / k_norm2 * (k * dphi + alpha * k_perp * phi) * exp_term
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
    velocity = U0(z, c1, c2) + u(x, y, z, c1, c2)
    return velocity

#####################

def solve_particle_trajectory(x0, y0, z0, c1, c2, t_span, max_step=0.001):
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
    sol = solve_ivp(lambda t, pos: particle_motion(t, pos, c1, c2), t_span, [x0, y0, z0], 
                    max_step=max_step, dense_output=True, rtol=1e-8, atol=1e-10)
    return sol

def analyze_trajectory(sol):
    '''
    Analyze the trajectory and print its properties
    Input
        - sol: solution object containing the particle trajectory
    '''
    t_eval = np.linspace(sol.t[0], sol.t[-1], 2000)
    trajectory = sol.sol(t_eval)
    
    x_traj = trajectory[0]
    y_traj = trajectory[1] 
    z_traj = trajectory[2]


def plot_mod2pi_trajectory(sol, n_points=2000):
    '''
    Plot the particle trajectory with modulo 2π 
    Input
        - sol: solution object from solve_ivp
        - n_points: number of points to evaluate the trajectory
    Returns
        - 3D plot of the particle trajectory
    '''
    t_eval = np.linspace(sol.t[0], sol.t[-1], n_points)
    trajectory = sol.sol(t_eval)
    
    x_traj = trajectory[0]
    y_traj = trajectory[1]
    z_traj = trajectory[2]
    
    # Modulo 2π for x and y
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
    
    # Convert segments to 3D coordinates
    X_segments = []
    Y_segments = []
    Z_segments = []
    
    # Iterate through segments and convert to 3D coordinates
    for seg_start, seg_end in segments:
        X = x_traj[seg_start:seg_end]
        Y = y_traj[seg_start:seg_end]
        Z = z_traj[seg_start:seg_end]
        
        X_segments.append(X)
        Y_segments.append(Y)
        Z_segments.append(Z)
    
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each segment
    for X, Y, Z in zip(X_segments, Y_segments, Z_segments):
        ax.plot(X, Y, Z, 'b-', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Particle trajcetory (mod 2π)')
    
    plt.tight_layout()
    plt.show()


def find_poincare_crossings(sol, n_points=20000):
    '''
    Find (x,z) points where y ≡ 0 mod 2π (Poincaré section)
    Input:
        - sol: trajectory solution from solve_ivp
        - n_points: number of time points to sample
    Returns:
        - x_crossings: array of x (mod 2π) at crossings
        - z_crossings: array of z at crossings
    '''
    t_eval = np.linspace(sol.t[0], sol.t[-1], n_points)
    traj = sol.sol(t_eval)
    
    x, y, z = traj[0], traj[1], traj[2]
    
    # Apply mod 2π to y
    y_mod = np.mod(y, 2*np.pi)
    
    # Define crossings: detect zero crossings of y_mod near 0 or 2π
    crossings = []
    for i in range(1, len(y_mod)):
        if (y_mod[i-1] > np.pi and y_mod[i] < np.pi) and (y[i] > y[i-1]):   # Only capture crossings of particles moving downwards
            # Linear interpolation
            t1, t2 = t_eval[i-1], t_eval[i]
            y1, y2 = y_mod[i-1], y_mod[i]
            frac = (0 - y1) / (y2 - y1)
            x_cross = x[i-1] + frac * (x[i] - x[i-1])
            z_cross = z[i-1] + frac * (z[i] - z[i-1])
            crossings.append((np.mod(x_cross, 2*np.pi), z_cross))

    x_crossings, z_crossings = zip(*crossings) if crossings else ([], [])
    return np.array(x_crossings), np.array(z_crossings)

def plot_poincare_section(x_crossings, z_crossings):
    '''
    Plot the Poincaré section in x-z plane
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(x_crossings, z_crossings, 'k.', markersize=3)
    plt.xlabel("x (mod 2π)")
    plt.ylabel("z")
    plt.title("Poincaré Section at y ≡ 0 (mod 2π)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============== HOVEDPROGRAM ===============
if __name__ == "__main__":
    print("Particle trajectory for first order")
    print("="*40)
    
    # 1. Solve for c1 and c2
    print('Solve for c1 and c2...')
    sol = root(system_to_solve, [1.0, 1.0], method='hybr')
    
    if not sol.success:
        print(f'Error: {sol.message}')
        exit()
    
    c1, c2 = sol.x
    print(f'Solution: c1 = {c1:.4f}, c2 = {c2:.4f}')
    
    # 2. Simulate particle trajectory
    print('Simulate particle trajectory...')
    x0, y0, z0 = np.pi/2, np.pi/2, 0.5          # Initial position  
    
    trajectory_sol = solve_particle_trajectory(x0, y0, z0, c1, c2, t_span)
    
    if not trajectory_sol.success:
        print(f'Error: {trajectory_sol.message}')
        exit()
    
    # 3. Analyze and plot the trajectory
    analyze_trajectory(trajectory_sol)
       
    print('Plotting trajectory...')
    plot_mod2pi_trajectory(trajectory_sol)

    # 4. Find Poincaré crossings
    print('Computing Poincaré crossings...')
    x_cross, z_cross = find_poincare_crossings(trajectory_sol)
    print('Plotting Poincaré section...')
    plot_poincare_section(x_cross, z_cross)

    print('Done!')


# -------------- SECOND ORDER ----------------
"""

# notater fra hva Douglas foreslår

def u3_hat_order2 (eta_hat_order2 ):
    # bruk kinematisk betingelse ....

    return [u_20, u_11, u_m11, u_02]

def u_order2 (u_20, u_11, u_m11, u_02, pos):

def dynamic_boundary_condition_order2 (eta_hat_order2):
    #
    u_order2(u3_hat_order2 (eta_hat_order2), pos)

    return dyn

# Anbefaler å gjøre en slags ligning fremfor fft
def eq (eta_hat_order_2):
    # skal være et linært system med 5 ukjente

    return dyn_con2 #i 5 pukter)

# Får eta_hat_2 ved å løse eq, for det er 5 løsninger

#Starter med en startgjett på eta_hat_2 i solve_ivp 


# med en løsning for eta_hat_2 kan vi finne en løsning for u2
# legger til u2 i plotten: u0 + u + u2 --> kan bruke den funksjonen jeg allerede har bare legge til u2
"""
