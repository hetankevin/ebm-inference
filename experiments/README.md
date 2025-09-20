# InferableEBM Experiments - Complete Setup

## 📁 Directory Structure

```
aistats2025/
├── interpret/                          # InterpretML with InferableEBM
│   └── python/interpret-core/
│       └── interpret/glassbox/
│           └── _ebm/_ebm.py            # InferableEBMRegressor implementation
├── experiments/                        # Experiment scripts and results
│   ├── 1d_intervals.py                # 1D interval visualization
│   ├── mse_comparison.py              # MSE comparison analysis
│   ├── coverage_rates.py              # Coverage rate analysis
│   ├── qq_plot.py                     # QQ plot for residual normality
│   ├── variable_importance.py         # Variable importance tests
│   ├── optuna_mse.py                  # Hyperparameter optimization
│   ├── matchups.py                    # Model comparison script
│   ├── matchups.csv                   # Model comparison results
│   ├── 1d/                            # 1D experiments
│   │   ├── data/                      # Data files and results
│   │   └── plots/                     # Generated plots
│   ├── coverage/                      # Coverage rate experiments
│   │   ├── data/                      # Coverage analysis results
│   │   └── plots/                     # Coverage plots
│   ├── mse_results/                   # MSE comparison results
│   ├── cache/                         # Cached computation results
│   └── README.md                      # Detailed experiment documentation
├── jobs/                              # Experiment scripts
│   ├── one_d_ci_pi_ri_plot.py        # 1D intervals experiment
│   ├── coverage_rate.py               # Coverage rate analysis
│   └── mse_comparison.py              # MSE comparison
├── plotting/                          # Plotting utilities
│   └── mse_plotting.py                # MSE visualization functions
├── run_experiments.py                 # Main experiment runner
├── venv/                              # Virtual environment
└── EXPERIMENTS_README.md              # This file
```

## 🚀 How to Run Experiments

### Quick Start

```bash
# From the aistats2025 directory
source venv/bin/activate

# Run all experiments
python run_experiments.py --experiment all

# Run specific experiments
python run_experiments.py --experiment 1d
python run_experiments.py --experiment coverage
python run_experiments.py --experiment mse
```

### Individual Experiments

```bash
# 1D intervals visualization
python jobs/one_d_ci_pi_ri_plot.py

# Coverage rate analysis
python jobs/coverage_rate.py

# MSE comparison
python jobs/mse_comparison.py
```

### Running Individual Scripts

```bash
# Run individual experiment scripts
python experiments/1d_intervals.py
python experiments/mse_comparison.py
python experiments/coverage_rates.py
python experiments/qq_plot.py
python experiments/variable_importance.py
python experiments/optuna_mse.py
python experiments/matchups.py
```

## 📊 Experiments Overview

### 1. 1D Interval Visualization
- **File**: `experiments/1d_intervals.py`
- **Purpose**: Demonstrate Algorithm 1's prediction intervals on f(X) = sin(2πX) + 0.5X²
- **Key Features**: Interval visualization, coverage analysis, parameter sensitivity

### 2. MSE Comparison
- **File**: `experiments/mse_comparison.py`
- **Purpose**: Compare InferableEBM vs Standard EBM prediction accuracy
- **Datasets**: Synthetic, Diabetes, Boston Housing
- **Key Features**: Multi-dataset comparison, parameter effects, statistical analysis

### 3. Coverage Rate Analysis
- **File**: `experiments/coverage_rates.py` + `jobs/coverage_rate.py`
- **Purpose**: Analyze prediction interval coverage rates
- **Key Features**: Parameter sensitivity, multiple α levels, statistical validity

### 4. QQ Plot Analysis
- **File**: `experiments/qq_plot.py`
- **Purpose**: Diagnose residual normality for statistical inference validity
- **Key Features**: Normality assessment, Algorithm 1 parameter effects

### 5. Variable Importance Tests
- **File**: `experiments/variable_importance.py`
- **Purpose**: Implement statistical variable importance tests
- **Key Features**: Significance testing, comparison with standard EBM

### 6. Hyperparameter Optimization
- **File**: `experiments/optuna_mse.py`
- **Purpose**: Optimize Algorithm 1 and EBM parameters using Optuna
- **Key Features**: Multi-objective optimization, cross-validation

### 7. Model Matchups
- **File**: `experiments/matchups.py` + `experiments/matchups.csv`
- **Purpose**: Compare InferableEBM against various standard methods
- **Key Features**: Multi-dataset benchmarking, uncertainty quantification advantages