import numpy as np

import generate_figures as gf


def test_split_step_norm_preservation():
    L, N = 20.0, 128
    x, k, dx = gf.grid(L, N)
    V = 0.2 * np.cos(2 * np.pi * x / L)
    psi = gf.normalize(np.exp(-x**2 / 6.0 + 1.5j * x), dx)
    norm0 = gf.norm2(psi, dx)
    for _ in range(250):
        psi = gf.split_step(psi, V, k, 0.01)
    assert abs(gf.norm2(psi, dx) - norm0) < 1e-12


def test_forward_euler_inflates_norm_for_generic_state():
    L, N = 20.0, 128
    x, k, dx = gf.grid(L, N)
    V = np.zeros_like(x)
    psi = gf.normalize(gf.gaussian(x, x0=-4.0, sigma=1.2, k0=2.0), dx)
    norm0 = gf.norm2(psi, dx)
    psi_next = gf.forward_euler(psi, V, k, 0.01)
    assert gf.norm2(psi_next, dx) > norm0


def test_phase_alignment_removes_global_phase():
    L, N = 10.0, 64
    x, _, dx = gf.grid(L, N)
    reference = gf.normalize(np.exp(-x**2 + 0.2j * x), dx)
    shifted = np.exp(0.73j) * reference
    aligned = gf.phase_align(shifted, reference, dx)
    assert gf.norm(aligned - reference, dx) < 1e-12


def test_normalized_dft_matrix_is_unitary():
    F = gf.dft_matrix(32)
    defect = F.conj().T @ F - np.eye(32)
    assert np.linalg.norm(defect, ord=2) < 1e-12


def test_split_step_time_reversal_recovers_state():
    L, N = 20.0, 128
    x, k, dx = gf.grid(L, N)
    V = 0.3 * np.cos(2 * np.pi * x / L)
    psi0 = gf.normalize(gf.gaussian(x, x0=-3.0, sigma=1.1, k0=1.7), dx)
    psi = psi0.copy()
    for _ in range(100):
        psi = gf.split_step(psi, V, k, 0.01)
    for _ in range(100):
        psi = gf.split_step(psi, V, k, -0.01)
    psi = gf.phase_align(psi, psi0, dx)
    assert gf.norm(psi - psi0, dx) < 1e-12


def test_square_barrier_transmission_is_physical():
    k = np.linspace(0.1, 5.0, 50)
    T = gf.square_barrier_transmission(k, V0=3.0, width=2.0)
    assert np.all(T >= 0.0)
    assert np.all(T <= 1.0)


def test_barrier_spectral_breakdown_is_consistent():
    L, N = 100.0, 512
    x, k, dx = gf.grid(L, N)
    psi0 = gf.normalize(gf.gaussian(x, x0=-30.0, sigma=2.5, k0=2.2), dx)
    breakdown = gf.barrier_spectral_breakdown(psi0, k, V0=3.0, width=3.0)
    total = breakdown['above_contribution'] + breakdown['below_contribution']
    assert abs(total - breakdown['weighted_T']) < 1e-14
    assert 0.0 <= breakdown['above_mass'] <= 1.0
    assert breakdown['above_contribution'] >= 0.0
    assert breakdown['below_contribution'] >= 0.0


def test_harmonic_ground_state_width_is_stable_over_short_run():
    L, N = 50.0, 512
    x, k, dx = gf.grid(L, N)
    omega = 0.4
    sigma = np.sqrt(gf.HBAR / (2 * gf.MASS * omega))
    V = 0.5 * omega**2 * x**2
    psi = gf.normalize(gf.gaussian(x, x0=4.0, sigma=sigma, k0=0.0), dx)
    for _ in range(200):
        psi = gf.split_step(psi, V, k, 0.01)
    assert abs(gf.variance_x(psi, x, dx) - sigma**2) < 1e-3


def test_yoshida4_split_preserves_norm():
    L, N = 20.0, 128
    x, k, dx = gf.grid(L, N)
    V = 0.2 * np.cos(2 * np.pi * x / L) + 0.03 * x**2
    psi = gf.normalize(gf.gaussian(x, x0=-2.0, sigma=1.0, k0=1.3), dx)
    norm0 = gf.norm2(psi, dx)
    for _ in range(50):
        psi = gf.split_step_yoshida4(psi, V, k, 0.01)
    assert abs(gf.norm2(psi, dx) - norm0) < 1e-12


def test_stored_dtype_split_step_preserves_shape_and_dtype():
    L, N = 20.0, 64
    x, k, dx = gf.grid(L, N)
    V = np.zeros_like(x)
    psi = gf.normalize(gf.gaussian(x, x0=-3.0, sigma=1.2, k0=1.0), dx).astype(np.complex64)
    updated = gf.split_step_stored_dtype(psi, V, k, 0.01, np.complex64)
    assert updated.shape == psi.shape
    assert updated.dtype == np.complex64


if __name__ == '__main__':
    test_split_step_norm_preservation()
    test_forward_euler_inflates_norm_for_generic_state()
    test_phase_alignment_removes_global_phase()
    test_normalized_dft_matrix_is_unitary()
    test_split_step_time_reversal_recovers_state()
    test_square_barrier_transmission_is_physical()
    test_barrier_spectral_breakdown_is_consistent()
    test_harmonic_ground_state_width_is_stable_over_short_run()
    test_yoshida4_split_preserves_norm()
    test_stored_dtype_split_step_preserves_shape_and_dtype()
    print('All numerical regression tests passed.')
