import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

# --------- PARAMETERS ------------------------
g = 9.81                  # Gravitational acceleration 
h = 1.0                   # Water depth 
alpha = 0.3               # Beltrami constant
sigma = 1.0               # Surface tension
t_span = [0, 100]         # Time span for the simulation

# --------- FIRST ORDER ------------------------
B_10 = 0.01               # Wave amplitude x-direction
B_01 = 0.01               # Wave amplitude y-direction

# --------- FUNCTIONS ------------------------
def compute_gamma(k, l):
    '''
    Compute the gamma value based on the wavenumbers k and l
    Input
        - k: wavenumber in x-direction
        - l: wavenumber in y-direction
    Returns
        - gamma: computed value based on the wavenumbers
    '''
    return np.sqrt(k**2 + l**2 - alpha**2)

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


def plot_mod2pi_trajectory(sol, n_points=20000):
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

def find_poincare_crossings(sol, n_points=2000):
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
    
    # Detect wrap-around crossings of y_mod at y ≡ 0 (mod 2π), from above 2π to below 0
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
    Input
        - x_crossings: array of x (mod 2π) at crossings
        - z_crossings: array of z at crossings
    Returns
        - Displays a plot of the Poincaré section
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
    
    # 3. Plot the trajectory
    print('Plotting trajectory...')
    plot_mod2pi_trajectory(trajectory_sol)

    # 4. Find Poincaré crossings
    print('Computing Poincaré crossings...')
    x_cross, z_cross = find_poincare_crossings(trajectory_sol)
    print('Plotting Poincaré section...')
    plot_poincare_section(x_cross, z_cross)

    print('Done!')

# -------------- SECOND ORDER ----------------

# Second-order Fourier modes and evaluation points
kvecs_order2 = [(0, 0), (2, 0), (1, 1), (1, -1), (0, 2)]
eval_points = [(i * np.pi/6, j * np.pi/6) for i in range(1, 4) for j in range(1, 4)]

def compute_gamma(k, l):
    '''
    Compute the gamma value based on the wavenumbers k and l for second-order modes
    Input
        - k: wavenumber in x-direction
        - l: wavenumber in y-direction
    Returns
        - gamma: computed value based on the wavenumbers
    '''
    K2 = k**2 + l**2
    gamma2 = K2 - alpha**2
    return np.sqrt(gamma2) if gamma2 > 0 else 1e-6

def eta2_at_point(x, y, eta_hat_2):
    '''
    Compute the second-order surface elevation at a given point
    Input
        - x: position in x-direction
        - y: position in y-direction
        - eta_hat_2: Fourier coefficients for second-order modes
    Returns
        - eta2: second-order surface elevation at position
    '''
    return sum(a * np.cos(k*x + l*y) for (k, l), a in zip(kvecs_order2, eta_hat_2))           # Take the sum of the Fourier modes 

def lap_eta2_at_point(x, y, eta_hat_2):
    '''
    Compute the Laplacian of the second-order surface elevation at a given point
    Input
        - x: position in x-direction
        - y: position in y-direction
        - eta_hat_2: Fourier coefficients for second-order modes
    Returns
        - lap_eta2: Laplacian of the second-order surface elevation at position
    '''
    return sum(-(k**2 + l**2) * np.real(a) * np.cos(k*x + l*y)                                # Take the sum of second derivatives
               for (k, l), a in zip(kvecs_order2, eta_hat_2)
               if (k, l) != (0, 0))

def u_order2(x, y, z, eta_hat_2):
    '''
    Compute the second-order velocity field u^(2) at a given position (x, y, z)
    Input
        - x: position in x-direction
        - y: position in y-direction
        - z: height above the bottom
        - eta_hat_2: Fourier coefficients for second-order modes
    Returns
        - u_total: total second-order velocity field at position (x, y, z)
    '''
    u_total = np.zeros(3, dtype=complex)

    # Iterate over each second-order mode
    for (k, l), a in zip(kvecs_order2, eta_hat_2):
        u3_hat = -1j * 2 * a
        phi = phi_kl(k, l, z)
        exp_term = np.exp(1j * (k*x + l*y))
        grad_phi = 1j * np.array([k, l]) * phi * u3_hat
        u_total[:2] += grad_phi * exp_term
        u_total[2] += u3_hat * phi * exp_term
    return np.real(u_total)

def dynamic_bc_order2(x, y, eta_hat_2, c1, c2):
    '''
    Compute the dynamic boundary condition for the second-order velocity field
    Input
        - x: position in x-direction
        - y: position in y-direction
        - eta_hat_2: Fourier coefficients for second-order modes
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
    Returns
        - dynamic_condition: value of the dynamic boundary condition at position
    '''
    eta1 = eta(x, y)
    u1 = u(x, y, h, c1, c2)
    u2 = u_order2(x, y, h, eta_hat_2)
    Uh = U0(h, c1, c2)[:2]

    # Compute the dynamic condition
    duhz = 0.0
    modes = [{'k': np.array([1, 0]), 'eta_hat': B_10},
             {'k': np.array([0, 1]), 'eta_hat': B_01}]

    # Iterate over each mode
    for mode in modes:
        k = mode['k']
        eta_hat = mode['eta_hat']
        dphi = dphi_dz(h, *k)
        grad = np.array(k)
        Uh_dot_grad = np.dot(Uh, grad)
        cos_term = np.cos(k[0]*x + k[1]*y)
        duhz += eta1 * Uh_dot_grad * dphi * np.real(eta_hat) * cos_term

    # Compute the kinetic energy and Laplacian of the second-order surface elevation
    kinetic = 0.5 * np.sum(u1[:2]**2)
    eta2_val = eta2_at_point(x, y, eta_hat_2)
    lap_eta2 = lap_eta2_at_point(x, y, eta_hat_2)

    return kinetic + np.dot(Uh, u2[:2]) + duhz - g * eta2_val + 2 * sigma * lap_eta2

def kinematic_bc_order2(x, y, eta_hat_2, c1, c2):
    '''
    Compute the kinematic boundary condition for the second-order velocity field
    Input
        - x: position in x-direction
        - y: position in y-direction
        - eta_hat_2: Fourier coefficients for second-order modes
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
    Returns
        - kinematic_condition: value of the kinematic boundary condition at position
    '''
    # First-order terms
    eta1_val = eta(x, y)
    grad_eta1 = np.array([np.real(1j * B_10 * np.exp(1j * x)),
                          np.real(1j * B_01 * np.exp(1j * y))])
    u1 = u(x, y, h, c1, c2)
    Uh = U0(h, c1, c2)[:2]

    # Gradient of the second-order surface elevation
    grad_eta2 = np.zeros(2, dtype=float)
    for (k, l), a in zip(kvecs_order2, eta_hat_2):
        grad_eta2[0] += -a * k * np.sin(k * x + l * y)
        grad_eta2[1] += -a * l * np.sin(k * x + l * y)

    # Compute
    dz_u3 = 0.0
    modes = [{'k': np.array([1, 0]), 'eta_hat': B_10, 'phase': x},
             {'k': np.array([0, 1]), 'eta_hat': B_01, 'phase': y}]
    for mode in modes:
        k = mode['k']
        phase = mode['phase']
        eta_hat = np.real(mode['eta_hat'])
        k_dot_Uh = np.dot(k, Uh)
        k_norm2 = np.dot(k, k)
        gamma = compute_gamma(*k)
        dphi_h = dphi_dz(h, *k)
        ddphi_h = gamma**2
        coeff = 1j * k_dot_Uh / (k_norm2 * dphi_h) * ddphi_h
        dz_u3 += np.real(coeff * np.exp(1j * phase))

    # Left and right sides of the BC
    u2 = u_order2(x, y, h, eta_hat_2)
    lhs = u2[2]
    rhs = -np.dot(Uh, grad_eta2) + np.dot(u1[:2], grad_eta1) - eta1_val * dz_u3
    return lhs - rhs


def eq_order2(eta_hat_2, c1, c2):
    '''
    Compute the residuals for the second-order boundary conditions
    Input
        - eta_hat_2: Fourier coefficients for second-order modes
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
    Returns
        - residuals: array of residuals for dynamic and kinematic boundary conditions
    '''
    residuals = []
    # Evaluate the dynamic and kinematic boundary conditions at specified points
    for (x, y) in eval_points:
        res_dyn = dynamic_bc_order2(x, y, eta_hat_2, c1, c2)
        res_kin = kinematic_bc_order2(x, y, eta_hat_2, c1, c2)
        residuals.extend([res_dyn, res_kin])
    return np.array(residuals)

def solve_eta_hat_order2(c1, c2):
    '''
    Solve for the second-order Fourier coefficients using least-squares optimization
    Input
        - c1: Fourier coefficient for the x-direction
        - c2: Fourier coefficient for the y-direction
    Returns
        - eta_hat_2: optimized Fourier coefficients for second-order modes
    '''
    # Define the residual function for least-squares optimization
    def residual(eta_hat_2): return eq_order2(eta_hat_2, c1, c2)
    initial_guess = [1e-4, 2e-4, -1e-4, 3e-4, 4e-4]
    result = least_squares(residual, initial_guess)
    if not result.success:
        raise RuntimeError(f"Least-squares solution failed: {result.message}")
    return result.x



# Solve and print results
eta_hat_2 = solve_eta_hat_order2(c1, c2)
for (k, l), val in zip(kvecs_order2, eta_hat_2):
    print(f"eta_({k},{l}) = {val:.6e}")