"""
Multi-Objective Bayesian Optimization for HEA Design
Uses BoTorch with Expected Hypervolume Improvement (EHVI)

Objectives:
1. MINIMIZE ln(Kp) - oxidation resistance
2. MAXIMIZE Temperature - operating temperature

Features: 16 elements + 5 CALPHAD descriptors (21 total)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

# BoTorch imports
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from botorch.sampling import SobolQMCNormalSampler

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

warnings.filterwarnings('ignore')

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ATOMIC_WEIGHTS = {
    'NI': 58.69, 'CR': 52.00, 'CO': 58.93, 'MO': 95.95, 'W': 183.84,
    'AL': 26.98, 'TI': 47.87, 'FE': 55.85, 'CU': 63.55, 'NB': 92.91,
    'C': 12.01, 'HF': 178.49, 'SI': 28.09, 'MN': 54.94, 'ZR': 91.22,
    'TA': 180.95
}

ELEMENT_COLS = ['NI', 'CR', 'CO', 'MO', 'W', 'AL', 'TI', 'FE', 'CU', 'NB',
                'C', 'HF', 'SI', 'MN', 'ZR', 'TA']

FEATURE_COLS = ELEMENT_COLS + ['FCC_amount', 'FCC_CR/AL_Mole_Fraction',
                               'FCC_L12_amount', 'FCC_L12_CR/AL_Mole_Fraction',
                               'temperature']


def normalize_composition(X: Tensor, element_indices: range) -> Tensor:
    """Normalize element compositions to sum to 1"""
    X_normalized = X.clone()
    mass_sum = X[:, element_indices].sum(dim=1, keepdim=True)
    X_normalized[:, element_indices] = X[:, element_indices] / (mass_sum + 1e-12)
    return X_normalized


def load_and_prepare_data(csv_path: str):
    """Load data and prepare bounds"""
    print(f"\n{'=' * 60}\nDATA LOADING\n{'=' * 60}")

    data = pd.read_csv(csv_path)

    # Check if data is in percentage format
    element_sum = data[ELEMENT_COLS].sum(axis=1).mean()
    if np.abs(element_sum - 100.0) < 1.0:
        print("Converting % to fractions")
        data[ELEMENT_COLS] = data[ELEMENT_COLS] / 100.0

    X = data[FEATURE_COLS].values
    y = data['ln(Kp)'].values

    print(f"Samples: {len(y)}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"ln(Kp) range: [{y.min():.4f}, {y.max():.4f}]")

    # Define bounds for optimization
    bounds = torch.zeros(2, len(FEATURE_COLS), dtype=torch.float64)

    # Elements: 0 to 1 (will be normalized to sum to 1)
    bounds[0, :16] = 0.0
    bounds[1, :16] = 1.0

    # CALPHAD features: use data range
    for i, col in enumerate(FEATURE_COLS[16:], start=16):
        bounds[0, i] = data[col].min()
        bounds[1, i] = data[col].max()

    print(f"\nCALPHAD feature bounds:")
    for i, col in enumerate(FEATURE_COLS[16:], start=16):
        print(f"  {col}: [{bounds[0, i]:.4f}, {bounds[1, i]:.4f}]")

    return X, y, bounds, data


def create_gp_model(train_X: Tensor, train_Y: Tensor):
    """Create a Gaussian Process model with Matern kernel"""
    # Add output dimension if needed
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        ),
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=train_Y.shape[-1]),
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


def optimize_ehvi_and_get_candidates(
        models: list,
        train_X: Tensor,
        train_Y: Tensor,
        bounds: Tensor,
        batch_size: int = 3,
        num_restarts: int = 10,
        raw_samples: int = 512,
        sampler_samples: int = 256,
) -> Tensor:
    """
    Optimize EHVI acquisition function to propose new candidates

    Args:
        models: List of GP models (one per objective) OR a single GP model.
        train_X: Current training inputs
        train_Y: Current training outputs (n x 2: [ln(kp), -temperature])
        bounds: Input bounds (2 x d) tensor
    """
    # Ensure train_Y is 2D tensor
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    # Device & dtype consistency
    device = train_X.device
    dtype = train_X.dtype
    bounds = bounds.to(device=device, dtype=dtype)
    train_Y = train_Y.to(device=device, dtype=dtype)

    # Compute reference point (nadir point with slight offset)
    y_max = train_Y.max(dim=0).values
    y_min = train_Y.min(dim=0).values
    ref_point = (y_max + 0.1 * (y_max - y_min)).to(device=device, dtype=dtype)

    # Partitioning for hypervolume computation
    partitioning = DominatedPartitioning(
        ref_point=ref_point,
        Y=train_Y,
    )

    # If the user passed a list of models, make a ModelListGP
    if isinstance(models, (list, tuple)):
        if len(models) == 1:
            model_for_acq = models[0]
        else:
            # make sure models are on same device/dtype
            # ModelListGP accepts already-fitted models
            model_for_acq = ModelListGP(*models)
    else:
        model_for_acq = models

    # Use a QMC Normal sampler for EHVI Monte Carlo estimation
    # Pass sample_shape as a torch.Size to match botorch versions that require it
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([sampler_samples]))

    # EHVI acquisition function
    acq_func = qExpectedHypervolumeImprovement(
        model=model_for_acq,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    # Optimize acquisition function
    candidates, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # Normalize compositions (do this after optimization so the composition sums to 1)
    candidates = normalize_composition(candidates, range(16))

    return candidates


def evaluate_candidates_with_oracle(candidates: Tensor, data_df: pd.DataFrame) -> Tensor:
    """
    Evaluate candidates using actual CALPHAD data (oracle)

    In practice, this would call Thermo-Calc. Here we simulate by finding
    nearest neighbors or returning placeholder values for demonstration.

    Returns: Tensor of shape (n_candidates, 2) with [ln(kp), temperature]
    """
    print(f"\n{'=' * 60}")
    print("ORACLE EVALUATION (placeholder - replace with Thermo-Calc)")
    print(f"{'=' * 60}")

    # Convert candidates to DataFrame
    candidates_np = candidates.cpu().numpy()
    candidates_df = pd.DataFrame(candidates_np, columns=FEATURE_COLS)

    # Placeholder: Return random values in training range for demonstration
    # REPLACE THIS with actual Thermo-Calc calculations
    n_candidates = len(candidates_df)
    lnkp_values = np.random.uniform(
        data_df['ln(Kp)'].min(),
        data_df['ln(Kp)'].max(),
        size=n_candidates
    )
    temp_values = candidates_np[:, -1]  # Temperature from candidates

    print(f"\nEvaluated {n_candidates} candidates:")
    for i in range(n_candidates):
        print(f"  Candidate {i + 1}:")
        print(f"    ln(Kp): {lnkp_values[i]:.4f}")
        print(f"    Temperature: {temp_values[i]:.1f} K")
        top_elements = [(ELEMENT_COLS[j], candidates_np[i, j])
                        for j in range(16) if candidates_np[i, j] > 0.01]
        top_elements.sort(key=lambda x: x[1], reverse=True)
        print(f"    Composition: " + ", ".join([f"{e}:{v * 100:.1f}%" for e, v in top_elements[:5]]))

    # Return as tensor: [ln(kp), temperature]
    oracle_Y = torch.tensor(
        np.column_stack([lnkp_values, temp_values]),
        dtype=torch.float64
    )

    return oracle_Y


def run_mobo_optimization(
        initial_X: Tensor,
        initial_Y_raw: Tensor,
        bounds: Tensor,
        data_df: pd.DataFrame,
        n_iterations: int = 10,
        batch_size: int = 3,
):
    """
    Main MOBO loop with EHVI

    Args:
        initial_X: Initial training inputs (from dataset)
        initial_Y_raw: Initial outputs [ln(kp), temperature]
        bounds: Input bounds
        data_df: Original dataframe (for oracle simulation)
        n_iterations: Number of BO iterations
        batch_size: Candidates per iteration
    """
    print(f"\n{'=' * 60}")
    print("STARTING MULTI-OBJECTIVE BAYESIAN OPTIMIZATION")
    print(f"{'=' * 60}")
    print(f"Initial samples: {len(initial_X)}")
    print(f"Iterations: {n_iterations}")
    print(f"Batch size: {batch_size}")
    print(f"Total new evaluations: {n_iterations * batch_size}")

    # Convert objectives: [ln(kp), -temperature] for minimization
    train_X = initial_X.clone()
    train_Y = torch.column_stack([
        initial_Y_raw[:, 0],  # ln(kp) - minimize
        -initial_Y_raw[:, 1]  # -temperature - minimize (maximize temp)
    ])

    # Track Pareto fronts over iterations
    pareto_history = []

    # Main BO loop
    for iteration in range(n_iterations):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'=' * 60}")
        print(f"Current dataset size: {len(train_X)}")

        # Train separate GP for each objective
        model_lnkp = create_gp_model(train_X, train_Y[:, 0:1])
        model_temp = create_gp_model(train_X, train_Y[:, 1:2])
        models = [model_lnkp, model_temp]

        # Optimize EHVI to get new candidates
        print(f"\nOptimizing EHVI acquisition function...")
        new_candidates = optimize_ehvi_and_get_candidates(
            models=models,
            train_X=train_X,
            train_Y=train_Y,
            bounds=bounds,
            batch_size=batch_size,
        )

        # Evaluate with oracle (Thermo-Calc)
        new_Y_raw = evaluate_candidates_with_oracle(new_candidates, data_df)
        new_Y = torch.column_stack([
            new_Y_raw[:, 0],
            -new_Y_raw[:, 1]
        ])

        # Add to training set
        train_X = torch.cat([train_X, new_candidates], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        # Compute current Pareto front
        pareto_mask = is_non_dominated(train_Y)
        pareto_X = train_X[pareto_mask]
        pareto_Y = train_Y[pareto_mask]

        pareto_history.append({
            'iteration': iteration + 1,
            'n_pareto': pareto_mask.sum().item(),
            'pareto_X': pareto_X.clone(),
            'pareto_Y': pareto_Y.clone(),
        })

        print(f"\nCurrent Pareto front size: {pareto_mask.sum().item()}")
        print(f"Best ln(Kp): {train_Y[:, 0].min().item():.4f}")
        print(f"Best Temperature: {(-train_Y[:, 1].max()).item():.1f} K")

    return train_X, train_Y, pareto_history


def plot_pareto_front(train_Y: Tensor, save_path: str = 'pareto_front.png'):
    """Plot the Pareto front"""
    # Convert back to original objectives
    lnkp = train_Y[:, 0].cpu().numpy()
    temp = -train_Y[:, 1].cpu().numpy()

    # Find Pareto front
    pareto_mask = is_non_dominated(train_Y)

    plt.figure(figsize=(10, 8))
    plt.scatter(lnkp[~pareto_mask], temp[~pareto_mask],
                alpha=0.5, s=50, label='Non-Pareto')
    plt.scatter(lnkp[pareto_mask], temp[pareto_mask],
                alpha=0.9, s=100, label='Pareto Front', marker='*')

    plt.xlabel('ln(Kp) [minimize]', fontsize=12)
    plt.ylabel('Temperature (K) [maximize]', fontsize=12)
    plt.title('Multi-Objective Optimization: Pareto Front', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved Pareto front plot: {save_path}")


def save_pareto_solutions(pareto_X: Tensor, pareto_Y: Tensor,
                          save_path: str = 'pareto_solutions.csv'):
    """Save Pareto optimal solutions to CSV"""
    # Convert to numpy
    pareto_X_np = pareto_X.cpu().numpy()
    pareto_Y_np = pareto_Y.cpu().numpy()

    # Create DataFrame
    df = pd.DataFrame(pareto_X_np, columns=FEATURE_COLS)
    df['ln(Kp)'] = pareto_Y_np[:, 0]
    df['Temperature(K)'] = -pareto_Y_np[:, 1]

    # Sort by ln(Kp)
    df = df.sort_values('ln(Kp)').reset_index(drop=True)

    df.to_csv(save_path, index=False)
    print(f"\nSaved Pareto solutions: {save_path}")

    # Print top solutions
    print(f"\n{'=' * 60}")
    print("TOP 10 PARETO SOLUTIONS")
    print(f"{'=' * 60}")

    for idx in range(min(10, len(df))):
        row = df.iloc[idx]
        print(f"\nSolution {idx + 1}:")
        print(f"  ln(Kp): {row['ln(Kp)']:.4f}")
        print(f"  Temperature: {row['Temperature(K)']:.1f} K")
        print(f"  FCC: {row['FCC_amount']:.3f}, FCC_L12: {row['FCC_L12_amount']:.3f}")

        top_elements = [(e, row[e]) for e in ELEMENT_COLS if row[e] > 0.01]
        top_elements.sort(key=lambda x: x[1], reverse=True)
        print(f"  Composition: " + ", ".join([f"{e}:{v * 100:.1f}%" for e, v in top_elements[:5]]))


def main():
    # Configuration
    CSV_PATH = os.environ.get(
        "HEA_MOBO_CSV",
        "/Users/zaifmohammed/PycharmProjects/HEA_MOBO/.venv/CALPHAD_included_dataset.csv",
    )
    N_ITERATIONS = 10  # Number of BO iterations
    BATCH_SIZE = 3  # Candidates per iteration

    # Load data
    X, y, bounds, data_df = load_and_prepare_data(CSV_PATH)

    # Prepare tensors
    initial_X = torch.tensor(X, dtype=torch.float64)

    # Initial Y: [ln(kp), temperature]
    initial_Y_raw = torch.tensor(
        np.column_stack([y, X[:, -1]]),  # X[:, -1] is temperature
        dtype=torch.float64
    )

    # Normalize compositions in initial data
    initial_X = normalize_composition(initial_X, range(16))

    print(f"\nInitial data:")
    print(f"  X shape: {initial_X.shape}")
    print(f"  Y shape: {initial_Y_raw.shape}")
    print(f"  ln(Kp) range: [{initial_Y_raw[:, 0].min():.4f}, {initial_Y_raw[:, 0].max():.4f}]")
    print(f"  Temperature range: [{initial_Y_raw[:, 1].min():.1f}, {initial_Y_raw[:, 1].max():.1f}] K")

    # Run MOBO
    final_X, final_Y, pareto_history = run_mobo_optimization(
        initial_X=initial_X,
        initial_Y_raw=initial_Y_raw,
        bounds=bounds,
        data_df=data_df,
        n_iterations=N_ITERATIONS,
        batch_size=BATCH_SIZE,
    )

    # Get final Pareto front
    pareto_mask = is_non_dominated(final_Y)
    pareto_X = final_X[pareto_mask]
    pareto_Y = final_Y[pareto_mask]

    print(f"\n{'=' * 60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total evaluations: {len(final_X)}")
    print(f"Final Pareto front size: {pareto_mask.sum().item()}")
    print(f"Best ln(Kp): {final_Y[:, 0].min().item():.4f}")
    print(f"Best Temperature: {(-final_Y[:, 1].max()).item():.1f} K")

    # Plot and save results
    plot_pareto_front(final_Y, 'pareto_front_mobo.png')
    save_pareto_solutions(pareto_X, pareto_Y, 'pareto_solutions_mobo.csv')

    print(f"\n{'=' * 60}")
    print("IMPORTANT: Replace oracle evaluation with Thermo-Calc!")
    print(f"{'=' * 60}")
    print("The evaluate_candidates_with_oracle() function currently returns")
    print("placeholder values. Integrate your Thermo-Calc Python API calls there.")


if __name__ == "__main__":
    main()


