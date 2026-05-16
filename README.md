# Split-Step Fourier Reproducibility Package

This repository contains the reproducibility code and generated numerical outputs for a one-dimensional split-step Fourier study of the time-dependent Schrodinger equation.

The package focuses on the numerical experiments behind the reported figures and diagnostics: discrete norm preservation, analytic free-packet checks, barrier scattering, harmonic-oscillator evolution, time reversal, temporal and spatial refinement, precision sensitivity, long-time drift, and comparisons with other propagators.

## Repository Contents

```text
.
├── generate_figures.py      # Numerical experiments and figure/data generation
├── test_numerics.py         # Lightweight numerical regression checks
├── requirements.txt         # Minimal pip dependencies
├── environment.yml          # Optional conda environment
├── data/                    # Generated result summaries and convergence data
└── figures/                 # Generated vector PDF figures
```

The repository intentionally excludes manuscript source files and build artifacts. It is meant to reproduce and inspect the computational results, not to rebuild the paper text.

## Setup

Python 3.11 or newer is recommended.

Using `pip`:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Using `conda`:

```bash
conda env create -f environment.yml
conda activate ssf-paper
```

## Reproduce the Results

Run the full numerical workflow from the repository root:

```bash
python generate_figures.py --root .
```

This regenerates:

- PDF figures in `figures/`
- `data/results_summary.json`
- `data/results_summary.txt`
- `data/time_convergence.csv`
- `data/spatial_convergence.csv`

For a faster smoke check:

```bash
python generate_figures.py --root . --quick
```

To run one experiment:

```bash
python generate_figures.py --root . --experiment method_comparison
```

Available experiments are:

- `free`
- `barrier`
- `harmonic`
- `norm_comparison`
- `convergence`
- `spatial_convergence`
- `long_time_norm`
- `unitarity_defect`
- `time_reversal`
- `method_comparison`
- `precision_sensitivity`

## Regression Checks

Run the numerical regression checks with:

```bash
python test_numerics.py
```

The tests check split-step norm preservation, Yoshida fourth-order composition norm preservation, forward Euler norm growth on a generic state, phase alignment, normalized DFT unitarity, time-reversal recovery, precision-storage behavior, and physical bounds for the square-barrier transmission formula.

## Result Snapshot

The current generated result summary is stored in `data/results_summary.json` and `data/results_summary.txt`. A few headline diagnostics from the included outputs are:

- Free Gaussian maximum norm error: `3.305133944309091e-13`
- Free Gaussian full-state error: `3.958179842672649e-12`
- Harmonic oscillator maximum norm error: `6.514788708500419e-13`
- Long-time norm error over `12000` steps: `2.2341017924532025e-12`
- Observed temporal convergence slope: `2.020697537500877`
- Explicit DFT unitarity Frobenius defect: `1.0427091626200353e-13`
- Barrier transmission estimate: `0.04584802956943435`

The JSON file is the preferred audit trail for exact numerical values and package-version metadata.

## Figure Inventory

The `figures/` directory contains vector PDF figures for:

- Free-packet density snapshots, heatmap, moments, and state error
- Barrier scattering heatmap, snapshots, and probability partition
- Harmonic-oscillator heatmap, snapshots, center trajectory, and energy drift
- Norm preservation and norm-error comparisons
- Method comparison across propagators
- Precision sensitivity
- Temporal and spatial convergence
- Time-reversal recovery
- Long-time norm drift
- Explicit unitarity-defect diagnostics

## Citation

If using this code or the generated diagnostics, cite the accompanying manuscript and this repository.
