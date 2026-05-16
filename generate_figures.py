import argparse
import json
from datetime import datetime, timezone
import platform

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
DATA = ROOT / 'data'
WRITE_TEX = False


def set_root(root):
    global ROOT, FIG, DATA
    ROOT = Path(root).resolve()
    FIG = ROOT / 'figures'
    DATA = ROOT / 'data'
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

# Dimensionless units used throughout the numerical section
HBAR = 1.0
MASS = 1.0

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8.6,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.2,
    'figure.dpi': 160,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'lines.linewidth': 1.45,
})


def grid(L, N):
    dx = L / N
    x = -L / 2 + dx * np.arange(N)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    return x, k, dx


def normalize(psi, dx):
    nrm = np.sqrt(dx * np.sum(np.abs(psi)**2))
    return psi / nrm


def norm2(psi, dx):
    return float(dx * np.sum(np.abs(psi)**2))


def norm(psi, dx):
    return np.sqrt(norm2(psi, dx))


def inner(u, v, dx):
    return dx * np.vdot(u, v)


def gaussian(x, x0, sigma, k0):
    return np.exp(-((x - x0)**2) / (4 * sigma**2) + 1j * k0 * x)


def split_step(psi, V, k, dt):
    P = np.exp(-0.5j * V * dt / HBAR)
    K = np.exp(-1j * (HBAR**2 * k**2 / (2 * MASS)) * dt / HBAR)
    psi = P * psi
    ph = np.fft.fft(psi, norm='ortho')
    ph = K * ph
    psi = np.fft.ifft(ph, norm='ortho')
    psi = P * psi
    return psi


def split_step_yoshida4(psi, V, k, dt):
    cbrt2 = 2.0**(1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)
    psi = split_step(psi, V, k, w1 * dt)
    psi = split_step(psi, V, k, w0 * dt)
    psi = split_step(psi, V, k, w1 * dt)
    return psi


def split_step_stored_dtype(psi, V, k, dt, dtype):
    P = np.exp(-0.5j * V * dt / HBAR).astype(dtype)
    K = np.exp(-1j * (HBAR**2 * k**2 / (2 * MASS)) * dt / HBAR).astype(dtype)
    psi = (P * psi).astype(dtype)
    ph = np.fft.fft(psi, norm='ortho').astype(dtype)
    ph = (K * ph).astype(dtype)
    psi = np.fft.ifft(ph, norm='ortho').astype(dtype)
    psi = (P * psi).astype(dtype)
    return psi


def apply_hamiltonian(psi, V, k):
    ph = np.fft.fft(psi, norm='ortho')
    kinetic = np.fft.ifft((HBAR**2 * k**2 / (2 * MASS)) * ph, norm='ortho')
    return kinetic + V * psi


def energy(psi, V, k, dx):
    return float(np.real(inner(psi, apply_hamiltonian(psi, V, k), dx)))


def expectation_x(psi, x, dx):
    return float(dx * np.sum(x * np.abs(psi)**2))


def variance_x(psi, x, dx):
    center = expectation_x(psi, x, dx)
    return float(dx * np.sum((x - center)**2 * np.abs(psi)**2))


def phase_align(psi, reference, dx):
    overlap = inner(psi, reference, dx)
    if abs(overlap) == 0:
        return psi
    return psi * np.exp(1j * np.angle(overlap))


def analytic_free_gaussian(x, t, x0, sigma, k0):
    """Continuum free Gaussian for i psi_t = -0.5 psi_xx, up to domain truncation."""
    spread = 1.0 + 1j * t / (2.0 * sigma**2)
    prefactor = (1.0 / (2.0 * np.pi * sigma**2))**0.25 / np.sqrt(spread)
    center = x0 + k0 * t
    envelope = np.exp(-((x - center)**2) / (4.0 * sigma**2 * spread))
    carrier = np.exp(1j * k0 * x - 0.5j * k0**2 * t)
    return prefactor * envelope * carrier


def boundary_tail_mass_from_density(density, x, dx, strip_fraction=0.10):
    half_width = 0.5 * (x[-1] - x[0] + dx)
    strip_width = strip_fraction * (2.0 * half_width)
    mask = np.abs(x) >= (half_width - strip_width)
    return float(dx * np.sum(density[mask]))


def max_tail_mass_from_heat(heat, x, dx, strip_fraction=0.10):
    if heat.size == 0:
        return 0.0
    return float(max(boundary_tail_mass_from_density(row, x, dx, strip_fraction) for row in heat))


def square_barrier_transmission(k_values, V0, width):
    """Plane-wave transmission through a square barrier in units hbar=m=1."""
    k_abs = np.abs(np.asarray(k_values, dtype=float))
    E = 0.5 * k_abs**2
    T = np.zeros_like(E)
    below = (E > 1e-14) & (E < V0 - 1e-10)
    above = E > V0 + 1e-10
    near = np.abs(E - V0) <= 1e-10
    if np.any(below):
        kappa = np.sqrt(2.0 * (V0 - E[below]))
        denom = 1.0 + (V0**2 * np.sinh(kappa * width)**2) / (4.0 * E[below] * (V0 - E[below]))
        T[below] = 1.0 / denom
    if np.any(above):
        q = np.sqrt(2.0 * (E[above] - V0))
        denom = 1.0 + (V0**2 * np.sin(q * width)**2) / (4.0 * E[above] * (E[above] - V0))
        T[above] = 1.0 / denom
    if np.any(near):
        denom = 1.0 + 0.5 * V0 * width**2
        T[near] = 1.0 / denom
    return T


def weighted_barrier_transmission(psi0, k, V0, width):
    ph = np.fft.fft(psi0, norm='ortho')
    weights = np.abs(ph)**2
    weights = weights / np.sum(weights)
    return float(np.sum(weights * square_barrier_transmission(k, V0, width)))


def forward_euler(psi, V, k, dt):
    return psi - 1j * dt * apply_hamiltonian(psi, V, k) / HBAR


def rk4(psi, V, k, dt):
    # dpsi/dt = -i H psi / hbar; included only as an optional comparison baseline.
    def rhs(u):
        return -1j * apply_hamiltonian(u, V, k) / HBAR
    k1 = rhs(psi)
    k2 = rhs(psi + 0.5 * dt * k1)
    k3 = rhs(psi + 0.5 * dt * k2)
    k4 = rhs(psi + dt * k3)
    return psi + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def dft_matrix(N):
    j = np.arange(N)
    ell = np.arange(N)
    return (1/np.sqrt(N))*np.exp(-2j*np.pi*np.outer(ell, j)/N)


def spectral_hamiltonian_matrix(V, k):
    N = len(V)
    F = dft_matrix(N)
    kinetic = F.conj().T @ np.diag((HBAR**2 * k**2) / (2 * MASS)) @ F
    return kinetic + np.diag(V)


def crank_nicolson_step_matrix(V, k, dt):
    H = spectral_hamiltonian_matrix(V, k)
    I = np.eye(len(V), dtype=complex)
    A = I + 0.5j * dt * H / HBAR
    B = I - 0.5j * dt * H / HBAR
    return np.linalg.solve(A, B), H


def run_ssf(psi0, V, k, dx, dt, T, sample_every=None, snapshots=()):
    steps = int(round(T / dt))
    psi = psi0.copy()
    norms = np.empty(steps + 1)
    norms[0] = norm2(psi, dx)
    energies = np.empty(steps + 1)
    energies[0] = energy(psi, V, k, dx)
    snap_steps = {int(round(t / dt)): t for t in snapshots}
    snaps = {0: psi.copy()} if 0 in snap_steps else {}
    heat = []
    heat_t = []
    if sample_every is not None:
        heat.append(np.abs(psi)**2)
        heat_t.append(0.0)
    for n in range(1, steps + 1):
        psi = split_step(psi, V, k, dt)
        norms[n] = norm2(psi, dx)
        energies[n] = energy(psi, V, k, dx)
        if n in snap_steps:
            snaps[n] = psi.copy()
        if sample_every is not None and (n % sample_every == 0 or n == steps):
            heat.append(np.abs(psi)**2)
            heat_t.append(n * dt)
    return {
        'psi': psi,
        'norms': norms,
        'energies': energies,
        'times': np.arange(steps + 1) * dt,
        'snaps': snaps,
        'heat': np.array(heat),
        'heat_t': np.array(heat_t),
    }


def save_pdf(fig, filename):
    fig.tight_layout()
    fig.savefig(FIG / filename, bbox_inches='tight')
    plt.close(fig)


def plot_snapshots(filename, x, snaps, dt, snap_times, title, xlim=None, ylim=None, potential=None, potential_label=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.35))
    for t in snap_times:
        step = int(round(t / dt))
        psi = snaps[step]
        ax.plot(x, np.abs(psi)**2, label=rf'$t={t:g}$')
    if potential is not None and np.max(potential) > 0:
        scaled = 0.18 * potential / np.max(potential)
        ax.fill_between(x, 0, scaled, alpha=0.18, step='mid', label=potential_label or 'scaled $V$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$|\psi(x,t)|^2$')
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=min(len(snap_times), 5), frameon=True)
    save_pdf(fig, filename)


def plot_heatmap(filename, x, t, heat, title, xlim=None, barrier=None):
    fig, ax = plt.subplots(figsize=(6.55, 3.45))
    im = ax.imshow(heat, origin='lower', aspect='auto', extent=[x[0], x[-1], t[0], t[-1]], cmap='viridis')
    if barrier is not None:
        ax.axvspan(barrier[0], barrier[1], color='white', alpha=0.19, linewidth=0)
        ax.axvline(barrier[0], color='white', alpha=0.55, linewidth=0.8)
        ax.axvline(barrier[1], color='white', alpha=0.55, linewidth=0.8)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label(r'$|\psi|^2$')
    save_pdf(fig, filename)


def experiment_free():
    L, N = 80.0, 2048
    x, k, dx = grid(L, N)
    V = np.zeros_like(x)
    dt, T = 0.01, 12.0
    snap_times = [0, 3, 6, 9, 12]
    psi0 = normalize(gaussian(x, x0=-20.0, sigma=2.0, k0=2.0), dx)
    out = run_ssf(psi0, V, k, dx, dt, T, sample_every=5, snapshots=snap_times)
    plot_snapshots('fig_free_snapshots.pdf', x, out['snaps'], dt, snap_times,
                   'Free Gaussian wavepacket: translation and dispersion', xlim=(-32, 14), ylim=(0, 0.23))
    plot_heatmap('fig_free_heatmap.pdf', x, out['heat_t'], out['heat'],
                 'Free wavepacket density, computed by split-step Fourier', xlim=(-35, 18))
    tail_max = max_tail_mass_from_heat(out['heat'], x, dx)
    # analytic group center for visual check
    times = out['times']
    centers = np.empty_like(times)
    variances = np.empty_like(times)
    state_errors = np.empty_like(times)
    psi = psi0.copy()
    centers[0] = expectation_x(psi, x, dx)
    variances[0] = variance_x(psi, x, dx)
    exact0 = normalize(analytic_free_gaussian(x, 0.0, -20.0, 2.0, 2.0), dx)
    state_errors[0] = norm(phase_align(psi, exact0, dx) - exact0, dx)
    for n in range(1, len(times)):
        psi = split_step(psi, V, k, dt)
        centers[n] = expectation_x(psi, x, dx)
        variances[n] = variance_x(psi, x, dx)
        exact = normalize(analytic_free_gaussian(x, times[n], -20.0, 2.0, 2.0), dx)
        state_errors[n] = norm(phase_align(psi, exact, dx) - exact, dx)
    exact_centers = -20.0 + 2.0 * times
    exact_variances = 2.0**2 + times**2 / (4 * 2.0**2)
    center_error = np.max(np.abs(centers - exact_centers))
    variance_error = np.max(np.abs(variances - exact_variances))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].plot(times, centers, label='computed')
    axes[0].plot(times, exact_centers, '--', linewidth=1.1, label='analytic')
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$\langle x\rangle_h$')
    axes[0].legend(frameon=True)
    axes[1].plot(times, variances, label='computed')
    axes[1].plot(times, exact_variances, '--', linewidth=1.1, label='analytic')
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$\mathrm{Var}_h(x)$')
    axes[1].legend(frameon=True)
    save_pdf(fig, 'fig_free_moments.pdf')
    fig, ax = plt.subplots(figsize=(5.9, 3.25))
    ax.semilogy(times, np.maximum(state_errors, 1e-18))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'phase-aligned $L^2_h$ error')
    ax.set_title('Full-state analytic validation for free propagation')
    save_pdf(fig, 'fig_free_state_error.pdf')
    max_norm = np.max(np.abs(out['norms'] - out['norms'][0]))
    max_energy = np.max(np.abs(out['energies'] - out['energies'][0]))
    return {
        'free_L': L, 'free_N': N, 'free_dt': dt, 'free_T': T, 'free_steps': int(round(T/dt)),
        'free_norm_error': max_norm, 'free_energy_error': max_energy,
        'free_center_error': float(center_error), 'free_variance_error': float(variance_error),
        'free_state_error': float(np.max(state_errors)), 'free_tail_mass': tail_max,
    }


def experiment_barrier():
    L, N = 100.0, 2048
    x, k, dx = grid(L, N)
    V = np.zeros_like(x)
    V0, a, b = 3.0, 0.0, 3.0
    V[(x >= a) & (x <= b)] = V0
    dt, T = 0.006, 28.0
    snap_times = [0, 10, 16, 22, 28]
    psi0 = normalize(gaussian(x, x0=-30.0, sigma=2.5, k0=2.2), dx)
    out = run_ssf(psi0, V, k, dx, dt, T, sample_every=8, snapshots=snap_times)
    tail_max = max_tail_mass_from_heat(out['heat'], x, dx)
    plot_snapshots('fig_barrier_snapshots.pdf', x, out['snaps'], dt, snap_times,
                   'Finite barrier scattering: incident, reflected, and transmitted packets',
                   xlim=(-42, 45), ylim=(0, 0.22), potential=V, potential_label='scaled barrier')
    plot_heatmap('fig_barrier_heatmap.pdf', x, out['heat_t'], out['heat'],
                 'Barrier scattering density heatmap', xlim=(-42, 45), barrier=(a,b))
    dens = np.abs(out['psi'])**2
    R = dx * np.sum(dens[x < a])
    B = dx * np.sum(dens[(x >= a) & (x <= b)])
    Tr = dx * np.sum(dens[x > b])
    weighted_T = weighted_barrier_transmission(psi0, k, V0, b - a)
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.bar([0,1,2], [R,B,Tr], tick_label=[r'$R$', r'$B$', r'$T$'])
    ax.scatter([2], [weighted_T], marker='D', s=42, color='crimson', zorder=5, label='spectral estimate')
    ax.set_ylim(0, 1)
    ax.set_ylabel('probability mass')
    ax.set_title('Late-time probability partition in barrier test')
    for i,v in enumerate([R,B,Tr]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.legend(frameon=True, loc='upper right')
    save_pdf(fig, 'fig_barrier_partition.pdf')
    return {
        'barrier_L': L, 'barrier_N': N, 'barrier_dt': dt, 'barrier_Tfinal': T,
        'barrier_steps': int(round(T/dt)), 'barrier_norm_error': float(np.max(np.abs(out['norms'] - out['norms'][0]))),
        'barrier_R': float(R), 'barrier_B': float(B), 'barrier_Trans': float(Tr), 'barrier_total': float(R+B+Tr),
        'barrier_weighted_T': weighted_T, 'barrier_T_difference': float(abs(Tr - weighted_T)),
        'barrier_tail_mass': tail_max, 'barrier_V0': V0, 'barrier_a': a, 'barrier_b': b,
    }


def experiment_harmonic():
    L, N = 70.0, 2048
    x, k, dx = grid(L, N)
    omega = 0.4
    V = 0.5 * omega**2 * x**2
    # Ground-state width for the dimensionless harmonic oscillator is sqrt(hbar/(2m omega)) in the exp[-(x-x0)^2/(4 sigma^2)] convention.
    sigma = np.sqrt(HBAR / (2 * MASS * omega))
    dt, T = 0.01, 36.0
    snap_times = [0, 9, 18, 27, 36]
    psi0 = normalize(gaussian(x, x0=8.0, sigma=sigma, k0=0.0), dx)
    out = run_ssf(psi0, V, k, dx, dt, T, sample_every=5, snapshots=snap_times)
    tail_max = max_tail_mass_from_heat(out['heat'], x, dx)
    plot_snapshots('fig_harmonic_snapshots.pdf', x, out['snaps'], dt, snap_times,
                   'Harmonic oscillator: oscillation of a localized packet', xlim=(-13, 13), ylim=(0, 0.42))
    plot_heatmap('fig_harmonic_heatmap.pdf', x, out['heat_t'], out['heat'],
                 'Harmonic oscillator density heatmap', xlim=(-13, 13))
    # diagnostics
    times = out['times']
    centers = np.empty_like(times)
    psi = psi0.copy()
    centers[0] = expectation_x(psi, x, dx)
    for n in range(1, len(times)):
        psi = split_step(psi, V, k, dt)
        centers[n] = expectation_x(psi, x, dx)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    exact_centers = 8.0 * np.cos(omega * times)
    center_error = float(np.max(np.abs(centers - exact_centers)))
    ax.plot(times, centers, label=r'computed $\langle x\rangle_h$')
    ax.plot(times, exact_centers, '--', linewidth=1.1, label=r'$8\cos(\omega t)$ guide')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\langle x\rangle_h$')
    ax.set_title('Harmonic oscillator center-of-mass diagnostic')
    ax.legend(frameon=True)
    save_pdf(fig, 'fig_harmonic_center.pdf')
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(times, out['energies'] - out['energies'][0])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$E_h^n-E_h^0$')
    ax.set_title('Harmonic oscillator energy diagnostic')
    save_pdf(fig, 'fig_harmonic_energy.pdf')
    return {
        'harmonic_L': L, 'harmonic_N': N, 'harmonic_dt': dt, 'harmonic_T': T, 'harmonic_steps': int(round(T/dt)),
        'harmonic_norm_error': float(np.max(np.abs(out['norms'] - out['norms'][0]))),
        'harmonic_energy_error': float(np.max(np.abs(out['energies'] - out['energies'][0]))),
        'harmonic_center_error': center_error, 'harmonic_tail_mass': tail_max, 'harmonic_omega': omega,
    }


def experiment_norm_comparison():
    L, N = 80.0, 256
    x, k, dx = grid(L, N)
    V = np.zeros_like(x)
    dt, T = 0.004, 4.0
    steps = int(round(T / dt))
    times = np.arange(steps+1) * dt
    psi0 = normalize(gaussian(x, x0=-20.0, sigma=2.0, k0=2.0), dx)
    psi_s = psi0.copy()
    psi_fe = psi0.copy()
    psi_rk4 = psi0.copy()
    n_s = np.empty(steps+1); n_fe = np.empty(steps+1); n_rk4 = np.empty(steps+1)
    n_s[0] = norm2(psi_s, dx); n_fe[0] = norm2(psi_fe, dx); n_rk4[0] = norm2(psi_rk4, dx)
    for n in range(1, steps+1):
        psi_s = split_step(psi_s, V, k, dt)
        psi_fe = forward_euler(psi_fe, V, k, dt)
        # RK4 included to show accurate methods are not automatically exactly unitary.
        psi_rk4 = rk4(psi_rk4, V, k, dt)
        n_s[n] = norm2(psi_s, dx)
        n_fe[n] = norm2(psi_fe, dx)
        n_rk4[n] = norm2(psi_rk4, dx)
    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    ax.semilogy(times, np.maximum(np.abs(n_s - n_s[0]), 1e-18), label='split-step Fourier')
    ax.semilogy(times, np.maximum(np.abs(n_fe - n_fe[0]), 1e-18), label='forward Euler')
    ax.semilogy(times, np.maximum(np.abs(n_rk4 - n_rk4[0]), 1e-18), label='classical RK4')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$|\|\psi^n\|_h^2-\|\psi^0\|_h^2|$')
    ax.set_title('Norm error for unitary and non-unitary time steppers')
    ax.legend(frameon=True)
    save_pdf(fig, 'fig_norm_error_comparison.pdf')
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    ax.plot(times, n_s, label='split-step Fourier')
    ax.plot(times, n_fe, label='forward Euler')
    ax.plot(times, n_rk4, label='classical RK4')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\|\psi^n\|_h^2$')
    ax.set_title('Discrete norm squared over time')
    ax.legend(frameon=True)
    save_pdf(fig, 'fig_norm_squared_comparison.pdf')
    return {
        'compare_L': L, 'compare_N': N, 'compare_dt': dt, 'compare_T': T, 'compare_steps': steps,
        'compare_ssf_norm_error': float(np.max(np.abs(n_s-n_s[0]))),
        'compare_fe_final_norm': float(n_fe[-1]), 'compare_fe_norm_error': float(np.max(np.abs(n_fe-n_fe[0]))),
        'compare_rk4_norm_error': float(np.max(np.abs(n_rk4-n_rk4[0]))),
    }


def experiment_convergence():
    L, N = 60.0, 1024
    x, k, dx = grid(L, N)
    omega = 0.35
    V = 0.5 * omega**2 * x**2
    psi0 = normalize(gaussian(x, x0=5.0, sigma=1.2, k0=0.3), dx)
    T = 3.0
    dt_ref = 0.0009375  # 3200 steps
    steps_ref = int(round(T/dt_ref))
    psi_ref = psi0.copy()
    for _ in range(steps_ref):
        psi_ref = split_step(psi_ref, V, k, dt_ref)
    dts = np.array([0.06, 0.03, 0.015, 0.0075, 0.00375])
    errors = []
    for dt in dts:
        steps = int(round(T/dt))
        psi = psi0.copy()
        for _ in range(steps):
            psi = split_step(psi, V, k, dt)
        # Remove a global phase before comparing, because phase is physically irrelevant and can obscure state-shape error.
        phase = inner(psi, psi_ref, dx)
        if abs(phase) > 0:
            psi_aligned = psi * np.exp(-1j * np.angle(phase))
        else:
            psi_aligned = psi
        errors.append(norm(psi_aligned - psi_ref, dx))
    errors = np.array(errors)
    coeff = np.polyfit(np.log(dts), np.log(errors), 1)
    slope = coeff[0]
    fig, ax = plt.subplots(figsize=(5.65, 3.3))
    ax.loglog(dts, errors, 'o-', label=rf'observed slope ${slope:.2f}$')
    # reference slope 2 line through last point
    ref_line = errors[-1] * (dts / dts[-1])**2
    ax.loglog(dts, ref_line, '--', label=r'reference slope $2$')
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'phase-aligned $L^2_h$ error at $T=3$')
    ax.set_title('Temporal convergence diagnostic for Strang splitting')
    ax.legend(frameon=True)
    ax.invert_xaxis()
    save_pdf(fig, 'fig_time_convergence.pdf')
    np.savetxt(DATA / 'time_convergence.csv', np.column_stack([dts, errors]), delimiter=',', header='dt,error', comments='')
    return {
        'conv_L': L, 'conv_N': N, 'conv_T': T, 'conv_dt_ref': dt_ref,
        'conv_slope': float(slope),
        'conv_err_0': float(errors[0]), 'conv_err_last': float(errors[-1])
    }


def experiment_spatial_convergence():
    L, T = 80.0, 4.0
    dt = 0.01
    Ns = np.array([128, 256, 512, 1024, 2048])
    x0, sigma, k0 = -18.0, 0.9, 3.0

    def exact_free(x, t):
        spread = 1.0 + 1j * t / (2.0 * sigma**2)
        prefactor = (1.0 / (2.0 * np.pi * sigma**2))**0.25 / np.sqrt(spread)
        center = x0 + k0 * t
        envelope = np.exp(-((x - center)**2) / (4.0 * sigma**2 * spread))
        phase = np.exp(1j * k0 * (x - x0) - 0.5j * k0**2 * t)
        return prefactor * envelope * phase

    errors = []
    for N in Ns:
        x, k, dx = grid(L, int(N))
        V = np.zeros_like(x)
        psi = normalize(gaussian(x, x0=x0, sigma=sigma, k0=k0), dx)
        for _ in range(int(round(T / dt))):
            psi = split_step(psi, V, k, dt)
        exact = normalize(exact_free(x, T), dx)
        psi_aligned = phase_align(psi, exact, dx)
        errors.append(norm(psi_aligned - exact, dx))
    errors = np.array(errors)
    fig, ax = plt.subplots(figsize=(5.65, 3.3))
    ax.semilogy(Ns, errors, 'o-', label='phase-aligned error')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$L^2_h$ error at $T=4$')
    ax.set_title('Spatial refinement diagnostic for a free Gaussian packet')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=True)
    save_pdf(fig, 'fig_spatial_convergence.pdf')
    np.savetxt(DATA / 'spatial_convergence.csv', np.column_stack([Ns, errors]), delimiter=',', header='N,error', comments='')
    return {
        'spatial_L': L, 'spatial_T': T, 'spatial_dt': dt, 'spatial_N_ref': int(Ns[-1]),
        'spatial_err_0': float(errors[0]), 'spatial_err_last': float(errors[-1]),
        'spatial_ratio': float(errors[0] / errors[-1]) if errors[-1] > 0 else float('inf'),
    }


def experiment_long_time_norm():
    L, N = 80.0, 1024
    x, k, dx = grid(L, N)
    omega = 0.25
    V = 0.5 * omega**2 * x**2
    dt, T = 0.01, 120.0
    steps = int(round(T / dt))
    sample_every = 10
    psi = normalize(gaussian(x, x0=7.0, sigma=np.sqrt(HBAR / (2 * MASS * omega)), k0=0.0), dx)
    norm0 = norm2(psi, dx)
    sampled_t = [0.0]
    sampled_err = [0.0]
    max_err = 0.0
    for n in range(1, steps + 1):
        psi = split_step(psi, V, k, dt)
        err = abs(norm2(psi, dx) - norm0)
        max_err = max(max_err, err)
        if n % sample_every == 0 or n == steps:
            sampled_t.append(n * dt)
            sampled_err.append(err)
    fig, ax = plt.subplots(figsize=(6.1, 3.2))
    ax.semilogy(sampled_t, np.maximum(sampled_err, 1e-18))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$|\|\psi^n\|_h^2-\|\psi^0\|_h^2|$')
    ax.set_title('Long-time norm preservation diagnostic')
    save_pdf(fig, 'fig_long_time_norm.pdf')
    return {
        'long_L': L, 'long_N': N, 'long_dt': dt, 'long_T': T,
        'long_steps': steps, 'long_norm_error': float(max_err),
    }


def experiment_unitarity_defect():
    # Small explicit matrix test: build S and compute ||S^*S-I||_2 and ||.||_F.
    L, N = 10.0, 64
    x, k, dx = grid(L, N)
    V = 0.2*np.cos(2*np.pi*x/L) + 0.1*x**2/(L**2)
    dt = 0.07
    j = np.arange(N)
    ell = np.arange(N)
    F = (1/np.sqrt(N))*np.exp(-2j*np.pi*np.outer(ell, j)/N)
    P = np.diag(np.exp(-0.5j*V*dt/HBAR))
    K = np.diag(np.exp(-1j*(HBAR**2*k**2/(2*MASS))*dt/HBAR))
    S = P @ F.conj().T @ K @ F @ P
    defect = S.conj().T @ S - np.eye(N)
    fro = np.linalg.norm(defect, ord='fro')
    two = np.linalg.norm(defect, ord=2)
    rng = np.random.default_rng(42)
    trials = 200
    errs = []
    for _ in range(trials):
        z = rng.normal(size=N) + 1j*rng.normal(size=N)
        z = z / np.sqrt(dx*np.sum(np.abs(z)**2))
        Sz = S @ z
        errs.append(abs(norm2(Sz, dx) - norm2(z, dx)))
    errs = np.array(errs)
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.hist(errs, bins=30, alpha=0.85)
    ax.set_xlabel(r'$|\|S z\|_h^2-\|z\|_h^2|$')
    ax.set_ylabel('random trials')
    ax.set_title('Floating-point unitarity defect for explicit small-matrix test')
    save_pdf(fig, 'fig_unitarity_defect_hist.pdf')
    return {
        'unitarity_N': N, 'unitarity_dt': dt, 'unitarity_fro': float(fro), 'unitarity_two': float(two),
        'unitarity_trial_max': float(np.max(errs)), 'unitarity_trial_median': float(np.median(errs))
    }


def experiment_time_reversal():
    cases = []
    for name, L, N, dt, T, potential_kind in [
        ('Free', 60.0, 1024, 0.01, 8.0, 'free'),
        ('Barrier', 80.0, 1024, 0.008, 12.0, 'barrier'),
        ('Harmonic', 60.0, 1024, 0.01, 16.0, 'harmonic'),
    ]:
        x, k, dx = grid(L, N)
        if potential_kind == 'free':
            V = np.zeros_like(x)
            psi0 = normalize(gaussian(x, x0=-16.0, sigma=2.0, k0=2.0), dx)
        elif potential_kind == 'barrier':
            V = np.zeros_like(x)
            V[(x >= 0.0) & (x <= 2.5)] = 2.5
            psi0 = normalize(gaussian(x, x0=-22.0, sigma=2.3, k0=2.1), dx)
        else:
            omega = 0.35
            V = 0.5 * omega**2 * x**2
            psi0 = normalize(gaussian(x, x0=6.0, sigma=np.sqrt(HBAR / (2 * MASS * omega)), k0=0.0), dx)
        steps = int(round(T / dt))
        psi = psi0.copy()
        for _ in range(steps):
            psi = split_step(psi, V, k, dt)
        for _ in range(steps):
            psi = split_step(psi, V, k, -dt)
        err = norm(phase_align(psi, psi0, dx) - psi0, dx)
        cases.append((name, err))
    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    ax.bar([c[0] for c in cases], [c[1] for c in cases])
    ax.set_yscale('log')
    ax.set_ylabel(r'phase-aligned recovery error')
    ax.set_title('Forward-backward time-reversal diagnostic')
    save_pdf(fig, 'fig_time_reversal.pdf')
    return {
        'reverse_free_error': float(cases[0][1]),
        'reverse_barrier_error': float(cases[1][1]),
        'reverse_harmonic_error': float(cases[2][1]),
    }


def experiment_method_comparison():
    L, N = 40.0, 96
    x, k, dx = grid(L, N)
    omega = 0.35
    V = 0.5 * omega**2 * x**2
    dt, T = 0.01, 1.2
    steps = int(round(T / dt))
    psi0 = normalize(gaussian(x, x0=4.0, sigma=np.sqrt(HBAR / (2 * MASS * omega)), k0=0.25), dx)
    C_cn, H = crank_nicolson_step_matrix(V, k, dt)
    evals, evecs = np.linalg.eigh(H)
    coeff = evecs.conj().T @ psi0
    psi_exact = evecs @ (np.exp(-1j * evals * T / HBAR) * coeff)

    methods = {
        'Split-step Fourier': psi0.copy(),
        'Yoshida 4 split': psi0.copy(),
        'Crank-Nicolson': psi0.copy(),
        'RK4': psi0.copy(),
        'Forward Euler': psi0.copy(),
    }
    for _ in range(steps):
        methods['Split-step Fourier'] = split_step(methods['Split-step Fourier'], V, k, dt)
        methods['Yoshida 4 split'] = split_step_yoshida4(methods['Yoshida 4 split'], V, k, dt)
        methods['Crank-Nicolson'] = C_cn @ methods['Crank-Nicolson']
        methods['RK4'] = rk4(methods['RK4'], V, k, dt)
        methods['Forward Euler'] = forward_euler(methods['Forward Euler'], V, k, dt)

    rows = []
    for name, psi in methods.items():
        aligned = phase_align(psi, psi_exact, dx)
        rows.append({
            'method': name,
            'norm_error': float(abs(norm2(psi, dx) - norm2(psi0, dx))),
            'state_error': float(norm(aligned - psi_exact, dx)),
            'center_error': float(abs(expectation_x(psi, x, dx) - expectation_x(psi_exact, x, dx))),
        })

    names = [row['method'] for row in rows]
    state_errors = [row['state_error'] for row in rows]
    norm_errors = [max(row['norm_error'], 1e-18) for row in rows]
    pos = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.45, 3.35))
    width = 0.38
    ax.bar(pos - width/2, state_errors, width, label='state error')
    ax.bar(pos + width/2, norm_errors, width, label='norm error')
    ax.set_yscale('log')
    ax.set_xticks(pos)
    ax.set_xticklabels(names, rotation=18, ha='right')
    ax.set_ylabel('error at final time')
    ax.set_title('Finite-dimensional method comparison')
    ax.legend(frameon=True)
    save_pdf(fig, 'fig_method_comparison.pdf')

    if WRITE_TEX:
        with open(DATA / 'method_comparison_table.tex', 'w') as f:
            f.write('\\begin{tabular}{lrrr}\n')
            f.write('\\toprule\n')
            f.write('Method & norm error & state error & center error \\\\ \n')
            f.write('\\midrule\n')
            for row in rows:
                f.write(f"{row['method']} & ${format_sci(row['norm_error'])}$ & ${format_sci(row['state_error'])}$ & ${format_sci(row['center_error'])}$ \\\\ \n")
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}\n')

    return {
        'method_L': L, 'method_N': N, 'method_dt': dt, 'method_T': T,
        'method_ssf_norm_error': rows[0]['norm_error'],
        'method_ssf_state_error': rows[0]['state_error'],
        'method_y4_norm_error': rows[1]['norm_error'],
        'method_y4_state_error': rows[1]['state_error'],
        'method_cn_norm_error': rows[2]['norm_error'],
        'method_cn_state_error': rows[2]['state_error'],
        'method_rk4_norm_error': rows[3]['norm_error'],
        'method_rk4_state_error': rows[3]['state_error'],
        'method_fe_norm_error': rows[4]['norm_error'],
        'method_fe_state_error': rows[4]['state_error'],
    }


def experiment_precision_sensitivity():
    L, N = 60.0, 512
    x, k, dx = grid(L, N)
    omega = 0.3
    V = 0.5 * omega**2 * x**2
    dt, T = 0.01, 40.0
    steps = int(round(T / dt))
    psi0 = normalize(gaussian(x, x0=6.0, sigma=np.sqrt(HBAR / (2 * MASS * omega)), k0=0.1), dx)
    rows = []
    for label, dtype in [('complex128', np.complex128), ('complex64 storage', np.complex64)]:
        psi = psi0.astype(dtype)
        norm0 = norm2(psi, dx)
        max_norm_error = 0.0
        for _ in range(steps):
            psi = split_step_stored_dtype(psi, V, k, dt, dtype)
            max_norm_error = max(max_norm_error, abs(norm2(psi, dx) - norm0))
        rows.append((label, max_norm_error))

    fig, ax = plt.subplots(figsize=(5.1, 3.1))
    ax.bar([row[0] for row in rows], [row[1] for row in rows])
    ax.set_yscale('log')
    ax.set_ylabel(r'max norm error')
    ax.set_title('Precision sensitivity for stored split-step states')
    save_pdf(fig, 'fig_precision_sensitivity.pdf')

    if WRITE_TEX:
        with open(DATA / 'precision_table.tex', 'w') as f:
            f.write('\\begin{tabular}{lr}\n')
            f.write('\\toprule\n')
            f.write('Storage model & max norm error \\\\ \n')
            f.write('\\midrule\n')
            for label, err in rows:
                f.write(f'{label} & ${format_sci(err)}$ \\\\ \n')
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}\n')

    return {
        'precision_L': L, 'precision_N': N, 'precision_dt': dt, 'precision_T': T,
        'precision_complex128_norm_error': float(rows[0][1]),
        'precision_complex64_norm_error': float(rows[1][1]),
    }


def make_parameter_table(results):
    rows = [
        ('Free packet', results['free_L'], results['free_N'], results['free_dt'], results['free_T']),
        ('Barrier', results['barrier_L'], results['barrier_N'], results['barrier_dt'], results['barrier_Tfinal']),
        ('Harmonic oscillator', results['harmonic_L'], results['harmonic_N'], results['harmonic_dt'], results['harmonic_T']),
        ('Norm comparison', results['compare_L'], results['compare_N'], results['compare_dt'], results['compare_T']),
        ('Convergence', results['conv_L'], results['conv_N'], np.nan, results['conv_T']),
        ('Spatial refinement', results['spatial_L'], results['spatial_N_ref'], results['spatial_dt'], results['spatial_T']),
        ('Long-time norm', results['long_L'], results['long_N'], results['long_dt'], results['long_T']),
        ('Method comparison', results['method_L'], results['method_N'], results['method_dt'], results['method_T']),
        ('Precision sensitivity', results['precision_L'], results['precision_N'], results['precision_dt'], results['precision_T']),
    ]
    with open(DATA / 'parameter_table.tex', 'w') as f:
        f.write('\\begin{tabular}{lrrrr}\n')
        f.write('\\toprule\n')
        f.write('Experiment & $L$ & $N$ & $\\Delta t$ & final time \\\\ \n')
        f.write('\\midrule\n')
        for name,L,N,dt,T in rows:
            dts = '$\\{0.06,\\ldots,0.00375\\}$' if np.isnan(dt) else f'{dt:g}'
            f.write(f'{name} & {L:g} & {int(N)} & {dts} & {T:g} \\\\ \n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')


def make_diagnostic_table(results):
    def math_sci(value):
        return f'${format_sci(value)}$'
    with open(DATA / 'diagnostic_table.tex', 'w') as f:
        f.write('\\small\n')
        f.write('\\begin{tabular}{L{0.22\\linewidth}L{0.31\\linewidth}L{0.33\\linewidth}}\n')
        f.write('\\toprule\n')
        f.write('Diagnostic & Quantity & Value \\\\ \n')
        f.write('\\midrule\n')
        rows = [
            ('Free packet', 'max norm error', math_sci(results['free_norm_error'])),
            ('Free packet', 'max full-state error', math_sci(results['free_state_error'])),
            ('Free packet', 'max center error', math_sci(results['free_center_error'])),
            ('Free packet', 'max variance error', math_sci(results['free_variance_error'])),
            ('Barrier', '$T$ and spectral estimate', f"{results['barrier_Trans']:.3f}, {results['barrier_weighted_T']:.3f}"),
            ('Harmonic oscillator', 'max center error', math_sci(results['harmonic_center_error'])),
            ('Boundary tails', 'free/barrier/harmonic maxima', f"{math_sci(results['free_tail_mass'])}, {math_sci(results['barrier_tail_mass'])}, {math_sci(results['harmonic_tail_mass'])}"),
            ('Harmonic oscillator', 'max energy error', math_sci(results['harmonic_energy_error'])),
            ('Norm comparison', 'forward Euler final norm', f"{results['compare_fe_final_norm']:.3f}"),
            ('Method comparison', 'CN state error', math_sci(results['method_cn_state_error'])),
            ('Method comparison', 'Yoshida 4 state error', math_sci(results['method_y4_state_error'])),
            ('Precision sensitivity', 'complex128/complex64 norm errors', f"{math_sci(results['precision_complex128_norm_error'])}, {math_sci(results['precision_complex64_norm_error'])}"),
            ('Time reversal', 'free/barrier/harmonic errors', f"{math_sci(results['reverse_free_error'])}, {math_sci(results['reverse_barrier_error'])}, {math_sci(results['reverse_harmonic_error'])}"),
            ('Temporal convergence', 'observed slope', f"{results['conv_slope']:.2f}"),
            ('Spatial refinement', 'error ratio', math_sci(results['spatial_ratio'])),
            ('Long-time norm', 'max norm error', math_sci(results['long_norm_error'])),
            ('Explicit matrix', 'Frobenius defect', math_sci(results['unitarity_fro'])),
        ]
        for exp, quantity, value in rows:
            f.write(f'{exp} & {quantity} & {value} \\\\ \n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')


def format_sci(x):
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10**exp)
    return rf'{mant:.2f}\times 10^{{{exp}}}'


def make_macros(results):
    def sci_macro(name, x):
        return rf'\newcommand{{\{name}}}{{${format_sci(float(x))}$}}' + '\n'
    def dec_macro(name, x, digits=3):
        return rf'\newcommand{{\{name}}}{{{float(x):.{digits}f}}}' + '\n'
    with open(ROOT / 'results_macros.tex', 'w') as f:
        f.write('% Automatically generated by generate_figures.py.\n')
        f.write('% Raw values are stored in data/results_summary.json.\n')
        f.write(sci_macro('FreeNormError', results['free_norm_error']))
        f.write(sci_macro('FreeEnergyError', results['free_energy_error']))
        f.write(sci_macro('FreeCenterError', results['free_center_error']))
        f.write(sci_macro('FreeVarianceError', results['free_variance_error']))
        f.write(sci_macro('FreeStateError', results['free_state_error']))
        f.write(sci_macro('FreeTailMass', results['free_tail_mass']))
        f.write(sci_macro('BarrierNormError', results['barrier_norm_error']))
        f.write(dec_macro('BarrierR', results['barrier_R']))
        f.write(dec_macro('BarrierB', results['barrier_B'], 5))
        f.write(dec_macro('BarrierT', results['barrier_Trans']))
        f.write(dec_macro('BarrierTotal', results['barrier_total']))
        f.write(dec_macro('BarrierWeightedT', results['barrier_weighted_T']))
        f.write(sci_macro('BarrierTDifference', results['barrier_T_difference']))
        f.write(sci_macro('BarrierTailMass', results['barrier_tail_mass']))
        f.write(sci_macro('HarmonicNormError', results['harmonic_norm_error']))
        f.write(sci_macro('HarmonicEnergyError', results['harmonic_energy_error']))
        f.write(sci_macro('HarmonicCenterError', results['harmonic_center_error']))
        f.write(sci_macro('HarmonicTailMass', results['harmonic_tail_mass']))
        f.write(sci_macro('CompareSSFNormError', results['compare_ssf_norm_error']))
        f.write(dec_macro('CompareFEFinalNorm', results['compare_fe_final_norm']))
        f.write(sci_macro('CompareFENormError', results['compare_fe_norm_error']))
        f.write(sci_macro('CompareRKNormError', results['compare_rk4_norm_error']))
        f.write(sci_macro('ReverseFreeError', results['reverse_free_error']))
        f.write(sci_macro('ReverseBarrierError', results['reverse_barrier_error']))
        f.write(sci_macro('ReverseHarmonicError', results['reverse_harmonic_error']))
        f.write(sci_macro('MethodSSFStateError', results['method_ssf_state_error']))
        f.write(sci_macro('MethodYFourStateError', results['method_y4_state_error']))
        f.write(sci_macro('MethodCNStateError', results['method_cn_state_error']))
        f.write(sci_macro('MethodRKStateError', results['method_rk4_state_error']))
        f.write(sci_macro('MethodFEStateError', results['method_fe_state_error']))
        f.write(sci_macro('MethodSSFNormError', results['method_ssf_norm_error']))
        f.write(sci_macro('MethodYFourNormError', results['method_y4_norm_error']))
        f.write(sci_macro('MethodCNNormError', results['method_cn_norm_error']))
        f.write(sci_macro('MethodRKNormError', results['method_rk4_norm_error']))
        f.write(sci_macro('MethodFENormError', results['method_fe_norm_error']))
        f.write(sci_macro('PrecisionDoubleNormError', results['precision_complex128_norm_error']))
        f.write(sci_macro('PrecisionSingleNormError', results['precision_complex64_norm_error']))
        f.write(dec_macro('ConvergenceSlope', results['conv_slope'], 2))
        f.write(sci_macro('SpatialFirstError', results['spatial_err_0']))
        f.write(sci_macro('SpatialLastError', results['spatial_err_last']))
        f.write(sci_macro('SpatialRatio', results['spatial_ratio']))
        f.write(sci_macro('LongTimeNormError', results['long_norm_error']))
        f.write(sci_macro('UnitaryFroDefect', results['unitarity_fro']))
        f.write(sci_macro('UnitaryTwoDefect', results['unitarity_two']))
        f.write(sci_macro('UnitaryTrialMax', results['unitarity_trial_max']))
        f.write(sci_macro('UnitaryTrialMedian', results['unitarity_trial_median']))


def write_results_json(results):
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'matplotlib': matplotlib.__version__,
        'results': results,
    }
    with open(DATA / 'results_summary.json', 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def experiment_registry():
    return {
        'free': experiment_free,
        'barrier': experiment_barrier,
        'harmonic': experiment_harmonic,
        'norm_comparison': experiment_norm_comparison,
        'convergence': experiment_convergence,
        'spatial_convergence': experiment_spatial_convergence,
        'long_time_norm': experiment_long_time_norm,
        'unitarity_defect': experiment_unitarity_defect,
        'time_reversal': experiment_time_reversal,
        'method_comparison': experiment_method_comparison,
        'precision_sensitivity': experiment_precision_sensitivity,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Regenerate figures and numerical summaries for the split-step Fourier experiments.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='Project root where figures/ and data/ should be written.')
    parser.add_argument('--experiment', choices=['all', *experiment_registry().keys()], default='all',
                        help='Run a single experiment for development, or all experiments for manuscript regeneration.')
    parser.add_argument('--quick', action='store_true',
                        help='Run a quick subset for smoke-checking figure generation without rewriting aggregate tables.')
    parser.add_argument('--metadata', action='store_true',
                        help='Print environment metadata and exit without running experiments.')
    parser.add_argument('--write-tex', action='store_true',
                        help='Also write manuscript helper tables and macros.')
    return parser.parse_args()


def main():
    global WRITE_TEX
    args = parse_args()
    set_root(args.root)
    WRITE_TEX = args.write_tex
    if args.metadata:
        payload = {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'matplotlib': matplotlib.__version__,
            'platform': platform.platform(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    results = {}
    registry = experiment_registry()
    if args.quick:
        selected = ['free', 'unitarity_defect', 'time_reversal', 'method_comparison', 'precision_sensitivity']
    elif args.experiment == 'all':
        selected = list(registry.keys())
    else:
        selected = [args.experiment]
    experiments = [registry[name] for name in selected]
    for fn in experiments:
        print(f'Running {fn.__name__}...')
        results.update(fn())
    if args.write_tex and args.experiment == 'all' and not args.quick:
        make_parameter_table(results)
        make_diagnostic_table(results)
        make_macros(results)
    write_results_json(results)
    with open(DATA / 'results_summary.txt', 'w') as f:
        for k, v in sorted(results.items()):
            f.write(f'{k} = {v}\n')
    print('Done. Summary:')
    for k, v in sorted(results.items()):
        print(f'{k}: {v}')

if __name__ == '__main__':
    main()
