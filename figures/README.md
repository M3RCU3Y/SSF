# Generated Figures

This directory contains the vector PDF outputs produced by:

```powershell
python generate_figures.py --root .
```

The README previews use PNG copies stored under `assets/readme/`, but the PDFs here are the canonical figure outputs for inspection and reuse.

Figure groups:

- `fig_free_*`: free Gaussian propagation and analytic checks.
- `fig_barrier_*`: finite barrier scattering and probability partitioning.
- `fig_harmonic_*`: harmonic oscillator trajectory, heatmap, snapshots, and energy behavior.
- `fig_method_comparison.pdf`: comparison across SSF, Yoshida, Crank-Nicolson, RK4, and forward Euler.
- `fig_time_convergence.pdf` and `fig_spatial_convergence.pdf`: refinement diagnostics.
- `fig_time_reversal.pdf`, `fig_long_time_norm.pdf`, `fig_precision_sensitivity.pdf`, and `fig_unitarity_defect_hist.pdf`: structural and floating-point diagnostics.
