import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
K        = 1.0      # nonlinearity strength
Omega0   = 0.3      # drive for the time series plots
theta0   = 0.1      # initial phase
N_time   = 200      # length of both time series
N_steps  = 3000     # iterations per Omega for staircase
M        = 1000     # resolution in Omega for staircase


#gauss map
def gauss(x):
    return np.exp(-6.2 * x**2)

#ORIGINAL circle_map step
def original_step(phi, Omega, K=1.0):
    return phi + Omega - (K/(2*np.pi)) * np.sin(2*np.pi * (phi % 1))

#MODIFIED circle_map step
def modified_step(phi, Omega, K=1.0):
    base  = Omega - (K/(2*np.pi)) * np.sin(2*np.pi * (phi % 1))
    return phi + base + gauss(phi % 1)

#winding number compute
def compute_rotation_number(step_func, Omegas, K, N):
    W = np.zeros_like(Omegas)
    for i, Om in enumerate(Omegas):
        phi = 0.0
        for _ in range(N):
            phi = step_func(phi, Om, K)
        W[i] = phi / N
    return W

#base circle map time series
theta_orig = np.zeros(N_time)
theta_orig[0] = theta0
for n in range(1, N_time):
    theta_orig[n] = original_step(theta_orig[n-1], Omega0, K) % 1

#modified circle map time series
theta_mod = np.zeros(N_time)
theta_mod[0] = theta0
for n in range(1, N_time):
    # note: we wrap only the total so the unwrapped-step matches compute_rotation_number
    theta_mod[n] = modified_step(theta_mod[n-1], Omega0, K) % 1

#Original and Modified Staircases
Omegas  = np.linspace(0, 1, M)
W_orig  = compute_rotation_number(original_step, Omegas, K, N_steps)
W_mod   = compute_rotation_number(modified_step, Omegas, K, N_steps)

# Arnold tongues heatmap parameters
K_vals    = np.linspace(0, 1, 80)    # 80 values for K
Omega_vals = np.linspace(0, 1, 160)  # 160 values for Omega
N_grid    = 1000                 # iterations per grid point

# Compute heatmap grid of rotation numbers for original map
W_grid = np.zeros((len(K_vals), len(Omega_vals)))
for i, Kv in enumerate(K_vals):
    for j, Om in enumerate(Omega_vals):
        phi = 0.5
        for _ in range(N_grid):
            phi = original_step(phi, Om, Kv)
        W_grid[i, j] = phi / N_grid

# 2x3 Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

# Top-left: pure time series
ax1.plot(np.arange(N_time), theta_orig, '-')
ax1.set_title('Original map time series')
ax1.set_xlabel('$n$'); ax1.set_ylabel(r'$\theta_n$')
ax1.grid(True)

# Top-middle: modified time series
ax2.plot(np.arange(N_time), theta_mod, '-', color='C1')
ax2.set_title('Modified map time series')
ax2.set_xlabel('$n$'); ax2.set_ylabel(r'$\theta_n$')
ax2.grid(True)

# Top-right: original devil's staircase
ax3.plot(Omegas, W_orig, '-', color='C2')
ax3.set_title("Original map Devil's staircase")
ax3.set_xlabel(r'$\Omega$'); ax3.set_ylabel('W')
ax3.grid(True)

# Bottom-left: modified devil's staircase
ax4.plot(Omegas, W_mod, '-', color='C3')
ax4.set_title("Modified map Devil's staircase")
ax4.set_xlabel(r'$\Omega$'); ax4.set_ylabel('W')
ax4.grid(True)

# Bottom-middle: Arnold tongues (heatmap)
X, Y = np.meshgrid(Omega_vals, K_vals)
pcm = ax5.pcolormesh(X, Y, W_grid, shading='auto')
fig.colorbar(pcm, ax=ax5, label='W')
ax5.set_title("Arnold tongues heatmap")
ax5.set_xlabel(r'$\Omega$'); ax5.set_ylabel('$K$')
ax5.grid(False)

# Bottom-right: hide unused
ax6.axis('off')

plt.tight_layout()

plt.show()
