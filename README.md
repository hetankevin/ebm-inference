# InferableEBMRegressor Implementation

This repository contains an implementation of **Algorithm 1: Inferable EBM** from the paper "Statistical Inference for Explainable Boosting Machines".

## Overview

The `InferableEBMRegressor` extends the standard Explainable Boosting Machine (EBM) with statistical inference capabilities through:

1. **Algorithm 1 Implementation**: Subsampling, mean-centering, and truncation
2. **Statistical Inference**: Prediction intervals and variable importance tests
3. **Structure Matrices**: Kernel matrices and influence vectors for uncertainty quantification

## Installation

The implementation is integrated into InterpretML and installed in development mode:

```bash
# Navigate to the project directory
cd /Users/Jonathan/research/giles/aistats2025

# Activate virtual environment
source venv/bin/activate

# The package is already installed in development mode
```

## Quick Start

```python
from interpret.glassbox import InferableEBMRegressor
import numpy as np
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)

# Create and fit model
model = InferableEBMRegressor(
    max_rounds=100,
    subsample_rate=1.0,    # Algorithm 1 parameter ξ
    truncation=3.0,        # Algorithm 1 parameter M
    random_state=42
)

model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Get prediction intervals
lower, upper, predictions = model.predict_intervals(X, level=0.95)

# Test variable importance for each feature
for i in range(X.shape[1]):
    result = model.variable_importance_test(X, y, groups=[i], level=0.95)
    print(f"Feature {i}: p-value = {result['p_value']:.4f}")
```

## Algorithm 1 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subsample_rate` (ξ) | 1.0 | Probability for subsampling samples for each feature |
| `truncation` (M) | 3.0 | Truncation parameter for limiting predictions to [-M, M] |
| `honest` | False | Whether to use honest estimation for statistical inference |
| `min_bin_count` | None | Minimum samples per bin for statistical inference (auto: max(3, floor(n^(1/3)))) |

## Statistical Inference Methods

### Prediction Intervals
```python
lower, upper, predictions = model.predict_intervals(X, level=0.95)
# Returns: (lower_bounds, upper_bounds, predictions) arrays
```

### Variable Importance Tests
```python
result = model.variable_importance_test(X, y, groups=[0], level=0.95)
# Returns: dict with 'stat', 'df', 'p_value' for testing feature significance
```

### Influence Vectors
```python
influence_vec = model._r_vector(x)
# Returns: influence vector for a single prediction
```

## Files Structure

```
aistats2025/
├── interpret/python/interpret-core/interpret/glassbox/_ebm/_ebm.py  # Main implementation
├── experiments/                                                     # Experiment notebooks and scripts
├── jobs/                                                           # Experiment execution scripts
├── papers/                                                         # Paper and documentation
└── README.md                                                       # This file
```

## Implementation Details

### Algorithm 1 Steps
1. **Initialize**: β₀ = (1/n) * Σᵢ yᵢ, f_k^(0) = 0 for all k
2. **For each boosting round b = 1 to B:**
   - **For each feature k = 1 to p:**
     - Sample G_{b,k} ⊂ {1,...,n} i.i.d. each i w.p. ξ
     - Compute residuals: r_{i,k} = y_i - (β₀ + Σ_{ℓ≠k} f_ℓ^{(b-1)}(x_i^{(ℓ)}))
     - Fit tree t^{(b,k)} to {(x_i^{(k)}, r_{i,k})}_{i∈G_{b,k}}
     - Compute mean: μ_{b,k} = (1/n) * Σᵢ t^{(b,k)}(x_i^{(k)})
     - Center: t̃^{(b,k)}(x) = t^{(b,k)}(x) - μ_{b,k}
     - Truncate: t̃^{(b,k)}(x) = max{-M, min{t̃^{(b,k)}(x), M}}
     - Update intercept: β₀ = β₀ + μ_{b,k}
     - Update feature function: f_k^{(b)} = (b-1)/b * f_k^{(b-1)} + 1/b * t̃^{(b,k)}

### Statistical Inference Theory
- **Structure Matrices**: S^(k) - orthogonal projectors onto vectors constant on each leaf
- **Kernel Vectors**: k^(k)(x) - expected influence of training points
- **Influence Vectors**: r^(k)(x) = k^(k)(x)^T (I - E[S^(k)])^† (I - 1/n 11^T) [I - (I + K)^† K]
- **Asymptotic Normality**: (f̂^(k)(x) - r^(k)(x)^T f(X)) / ||r^(k)(x)|| → N(0, σ²)

## Running Experiments

Run the experiment scripts to test the implementation:

```bash
# Activate virtual environment
source venv/bin/activate

# Run 1D interval experiments
python jobs/one_d_ci_pi_ri_plot.py

# Run coverage rate analysis
python jobs/coverage_rate.py

# Run MSE comparison
python jobs/mse_comparison.py
```

## Limitations

- **Native Library**: Full functionality requires the built InterpretML native library
- **Computational Cost**: Statistical inference methods add computational overhead
- **Memory Usage**: Structure matrices can be memory-intensive for large datasets

## References

- Original paper: "Statistical Inference for Explainable Boosting Machines"
- InterpretML: https://github.com/interpretml/interpret
- Algorithm 1: Page 1 of the paper

## Contact

For questions about this implementation, please refer to the original paper and InterpretML documentation.