import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---- PARAMETERS -------
g = 9.81                  # Gravitational acceleration 
h = 1.0                   # Water depth 
a = 0.1                   # Wave amplitude 
lambda_ = 2 * np.pi       # Wavelength
k = 2 * np.pi / lambda_   # Wave number 
c = np.sqrt(g * h)        # Linear wave speed 
f = k * c                 # Wave frequency 
s = 0.5                   # Integration constant

# Time domain
t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)

# --------------- FUNCTIONS -------------
def compute_Ak(k_mode, c, h, g, lambda_, omega, s):
    '''Compute the second-order Fourier coefficients for a given mode.'''
    if abs(k_mode) != 2:
        return 0
    
    sqrt_gh = np.sqrt(g * h)
    denom = np.sinh(k_mode**2 * h)

    # First term
    factor1 = (2 * np.pi**3 * h**2 * 1j * k_mode * (sqrt_gh * s - h * omega - c)) / (lambda_**2 * sqrt_gh * denom)
    inner1 = ((c - sqrt_gh * s + h * omega) * np.sinh(2 * h) -
              ((c - sqrt_gh * s + h * omega)**2) / (g * h * np.sinh(2 * h)**2))
    # Second term
    factor2 = (np.pi / (2 * 1j * sqrt_gh * denom))
    inner2 = (h * omega - 4 * np.pi * (c - sqrt_gh * s + h * omega) * (1 / np.tanh(2 * h)))

    # Final coefficient
    Ak = factor1 * inner1 + factor2 * inner2
    return Ak

def full_system_transformed(pos, omega, a, k, h, f, lambda_, g, A_k_dict):
    '''Defines the full ODE system in a moving reference fram including both first- and second-order effects.'''
    X, Y = pos  
    sqrt_gh = np.sqrt(g * h)
    A0 = a * (f + k * h * omega) / np.sinh(k * h)

    # Clip large values of Y to avoid overflow
    Y_clip = np.clip(Y, -100, 100)

    # First-order velocity components
    dX = A0 * k * np.cos(X) * np.cosh(Y_clip) - omega * (Y_clip / k) - f
    dY = A0 * k * np.sin(X) * np.sinh(Y_clip)

    # Second-ordeer velocity components
    A1 = a**2 * sqrt_gh / (h * lambda_)
    for n, Ak in A_k_dict.items():
        factor = (n * k)**2 * (Y_clip / k) / h 
        factor = np.clip(factor, -100, 100)
        dX += A1 * k * (Ak * np.cosh(factor)).real
        dY += A1 * k * (Ak * np.sinh(factor)).real

    return [dX, dY]

def solve_and_plot_trajcetories(start_values, omega, kryss_targets, *args):
    '''Solves the ODE system, plots resulting trajcetories and marks the Poincaré crossings'''
    plt.figure(figsize=(11, 7))

    for X0, Y0 in start_values:
        line_color = color_map_combo[(X0, Y0)]
        sol = solve_ivp(lambda t, y: full_system_transformed(y, omega, a, k, h, f, lambda_, g, A_k_dict),
                        t_span, [X0, Y0], t_eval=t_eval, method='RK45')
        # Plot trajectory
        plt.plot(sol.y[0], sol.y[1], label=f'Start: X0={X0:.2f}, Y0={Y0:.2f}', color=line_color)

    # Plot settings
    plt.title(f'Particle trajectories in a travelling frame for ω={omega}')
    plt.xlabel(r'$X = kx - ft$')
    plt.ylabel(r'$Y = ky$')
    plt.xlim(-lambda_*0.75, lambda_*1.75)
    plt.ylim(0, h+0.25)
    plt.grid()
    plt.legend(fontsize=8, markerscale=0.5)
    plt.tight_layout()
    plt.savefig(f'phase_portrait_omega_{omega}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_poincare_crossings(start_values, omega, kryss_targets, A_k_dict, poincare_colors):
    '''Solves trajectories and plots Poincaré crossings per target for a given ω.'''
    summary = {target: {'Y0': [], 'Y_cross': []} for target in kryss_targets}

    for X0, Y0 in start_values:
        sol = solve_ivp(lambda t, y: full_system_transformed(y, omega, a, k, h, f, lambda_, g, A_k_dict),
                        t_span, [X0, Y0], t_eval=t_eval, method='RK45')
        x, y_vals, t = sol.y[0], sol.y[1], sol.t

        for target in kryss_targets:
            for i in range(len(x) - 1):
                if (x[i] - target) * (x[i + 1] - target) < 0:
                    t_cross = t[i] + (t[i + 1] - t[i]) * (target - x[i]) / (x[i + 1] - x[i])
                    y_cross = y_vals[i] + (y_vals[i + 1] - y_vals[i]) * (t_cross - t[i]) / (t[i + 1] - t[i])
                    summary[target]['Y0'].append(Y0)
                    summary[target]['Y_cross'].append(y_cross)

    fig, axes = plt.subplots(1, len(kryss_targets), figsize=(18, 5), sharey=True)

    for i, target in enumerate(kryss_targets):
        ax = axes[i]
        Y0s = summary[target]['Y0']
        Ycross = summary[target]['Y_cross']
        if Y0s:
            ax.scatter(Y0s, Ycross,
                       alpha=0.7,
                       s=12,
                       color=poincare_colors.get(target, 'black'),
                       label=f'X = {target:.2f}')
        ax.set_title(f'X = {target:.2f}')
        ax.set_xlabel('Initial $Y_0$')
        if i == 0:
            ax.set_ylabel('Poincaré Crossing $Y$')
        ax.grid()
        ax.legend()

    plt.suptitle(f'Poincaré crossings for ω = {omega:.2f}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f'poincare_crossings_omega_{omega:.2f}_subplots.pdf', dpi=300)
    plt.show()
    plt.close(fig)

def vorticity_trajectories(omega_range):

    plt.figure(figsize=(12, 8))
    for omega in omega_range:
        if omega <= -25:
            y_vals = np.linspace(0.05, 0.25, 2)
        elif omega <= -20:
            y_vals = np.linspace(0.15, 0.35, 2)
        elif omega <= -15:
            y_vals = np.linspace(0.25, 0.45, 2)
        elif omega <= -10:
            y_vals = np.linspace(0.35, 0.55, 2)
        elif omega <= -5:
            y_vals = np.linspace(0.45, 0.65, 2)
        elif omega <= 0:
            y_vals = np.linspace(0.55, 0.8, 2)
        else:
            y_vals = np.linspace(0.7, h, 2)

        current_start_values_dict = {}
        current_start_values = [(2, y0) for y0 in y_vals]
        current_start_values_dict[omega] = start_values
    
        for X0, Y0 in current_start_values:
            sol = solve_ivp(lambda t, y: full_system_transformed(y, omega, a, k, h, f, lambda_, g, A_k_dict),
                            t_span, [X0, Y0], t_eval=t_eval, method='RK45')
            all_trajectories.append({'omega': omega, 'X': sol.y[0], 'Y': sol.y[1], 'X0': X0,'Y0': Y0})

    for traj in all_trajectories:
        plt.plot(traj['X'], traj['Y'], linewidth=0.7, label=f'ω={traj["omega"]:.2f}, X0={traj["X0"]:.2f}, Y0={traj["Y0"]:.2f}')

    plt.title('Particle trajectories for multiple vorticity values')
    plt.xlabel(r'$X = kx - ft$')
    plt.ylabel(r'$Y = ky$')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0, h + 0.2)
    plt.grid()
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('all_vorticities_custom_startvalues.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# -------- INITIAL CONDITIONS --------------
start_values = [(x0, y0) for x0 in np.linspace(0, lambda_, 2) for y0 in np.linspace(0.1, h, 3)]
start_values_crossings = [(x0, y0) for x0 in np.linspace(0, lambda_, 5) for y0 in np.linspace(0.1, h, 5)]
cross_targets = [0, np.pi/2, np.pi]

poincare_colors = {cross_targets[0]: 'red', cross_targets[1]: 'blue', cross_targets[2]: 'green'}
vorticities = [-10.0, -4.0, 1]
all_crossing_summaries = {}

omega_range = np.linspace(-30, 1, 15) 
all_trajectories = []

cmap = plt.get_cmap('Dark2')  
unique_combinations = list(sorted(set(start_values)))  # All unique (X0, Y0) pairs
color_map_combo = {combo: cmap(i % cmap.N) for i, combo in enumerate(unique_combinations)}


# ------------ MAIN LOOP ---------------
# Solve the system for each vorticity and plot results
for omega in vorticities:
    A_k_dict = {2: compute_Ak(2, c, h, g, lambda_, omega, s),
                -2: compute_Ak(-2, c, h, g, lambda_, omega, s)}
    solve_and_plot_trajcetories(start_values, omega, cross_targets, A_k_dict, a, k, h, f, lambda_, g)
    plot_poincare_crossings(start_values_crossings, omega, cross_targets, A_k_dict, poincare_colors)

vorticity_trajectories(omega_range)

