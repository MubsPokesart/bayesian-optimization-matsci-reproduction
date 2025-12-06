import os
import random
import time
import math
import copy
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import plotly.express as px

from sklearn import preprocessing
from sklearn import decomposition
from scipy.stats import norm

# === SUPPRESS ALL WARNINGS ===
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress logging from all libraries
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('GP').setLevel(logging.ERROR)
logging.getLogger('paramz').setLevel(logging.ERROR)

# Disable GPy's internal warnings

# --- Constants and Configuration ---
N_ENSEMBLE = 1  # Smoke test - change to 50 for full run
N_INITIAL = 2
TOP_K_PERCENT = 0.05
ACQUISITION_LCB_RATIO = 1.0
SEED_LIST = [
    4295, 8508, 326, 3135, 1549, 2528, 1274, 6545, 5971, 6269, 2422,
    4287, 9320, 4932, 951, 4304, 1745, 5956, 7620, 4545, 6003, 9885,
    5548, 9477, 30, 8992, 7559, 5034, 9071, 6437, 3389, 9816, 8617,
    3712, 3626, 1660, 3309, 2427, 9872, 938, 5156, 7409, 7672, 3411,
    3559, 9966, 7331, 8273, 8484, 5127
]

# Surrogate configuration
SURROGATES_TO_RUN = ['GP-sklearn', 'RF']  # Toggle which surrogates to run
N_RESTARTS_GP = 10  # Number of restarts for sklearn GP optimization
N_ESTIMATORS_RF = 100  # Number of trees in Random Forest

DATASETS = {
    'Crossed barrel': {
        'path': 'original_paper_dataset/Crossed barrel_dataset.csv',
        'negate_objective': True
    },
    'AgNP': {
        'path': 'original_paper_dataset/AgNP_dataset.csv',
        'negate_objective': False
    }
}

# --- Data Loading and Processing ---
def load_and_process_data(dataset_config):
    """Loads a dataset, negates objective if needed, and groups unique features."""
    raw_dataset = pd.read_csv(dataset_config['path'])
    feature_names = list(raw_dataset.columns)[:-1]
    objective_name = list(raw_dataset.columns)[-1]

    ds = copy.deepcopy(raw_dataset)
    if dataset_config['negate_objective']:
        ds[objective_name] = -raw_dataset[objective_name].values

    ds_grouped = ds.groupby(feature_names)[objective_name].agg(lambda x: x.unique().mean()).to_frame().reset_index()
    X_feature = ds_grouped[feature_names].values
    y_objective = ds_grouped[objective_name].values

    assert len(ds_grouped) == len(X_feature) == len(y_objective)
    return ds_grouped, feature_names, objective_name, X_feature, y_objective


# --- Surrogate Model Builders ---
def build_sklearn_gp_surrogate(X_train_norm, y_train_norm, n_features, seed):
    """
    Builds sklearn GP with Matérn 5/2 kernel and ARD.

    Args:
        X_train_norm: Normalized training features
        y_train_norm: Normalized training targets
        n_features: Number of feature dimensions
        seed: Random seed for reproducibility

    Returns:
        gp_model: Fitted GaussianProcessRegressor
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    # Matérn 5/2 with ARD + noise modeling
    kernel = (
        Matern(
            length_scale=[1.0] * n_features,  # ARD: separate lengthscale per dimension
            length_scale_bounds=(0.1, 10.0),
            nu=2.5  # Matérn 5/2
        )
        + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-5, 0.1)
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=N_RESTARTS_GP,  # Multi-start optimization
        optimizer='fmin_l_bfgs_b',
        alpha=0.0,  # Noise handled by WhiteKernel
        normalize_y=False,  # We handle normalization externally
        random_state=seed
    )

    # Fit automatically optimizes hyperparameters
    try:
        gp.fit(X_train_norm, y_train_norm.ravel())
    except Exception as e:
        # Fallback: add jitter for numerical stability
        gp.alpha = 1e-5
        gp.fit(X_train_norm, y_train_norm.ravel())

    return gp


def build_random_forest_surrogate(X_train_norm, y_train_norm, seed):
    """
    Builds Random Forest regressor for greedy acquisition.

    Args:
        X_train_norm: Normalized training features
        y_train_norm: Normalized training targets
        seed: Random seed for reproducibility

    Returns:
        rf_model: Fitted RandomForestRegressor
    """
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS_RF,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=seed,
        n_jobs=1
    )

    rf.fit(X_train_norm, y_train_norm.ravel())

    return rf


# --- Benchmark Frameworks ---
def run_benchmark_generic(X_feature, y_objective, top_indices, n_total, seed_list, surrogate_type='GP-sklearn'):
    """Runs BO benchmark with specified surrogate model.

    Args:
        X_feature: Feature matrix
        y_objective: Objective values
        top_indices: Indices of top-k elite materials
        n_total: Total number of samples
        seed_list: List of random seeds
        surrogate_type: Type of surrogate ('GP-sklearn', 'RF')

    Returns:
        np.array: Coverage trajectories (n_ensemble x n_total)
    """
    top_k_coverage_collection = []

    for seed_idx, seed in enumerate(seed_list[:N_ENSEMBLE]):
        print(f"Running seed {seed_idx + 1}/{N_ENSEMBLE} (seed={seed})...")
        random.seed(seed)
        np.random.seed(seed)

        indices = list(np.arange(n_total))
        indices_learn = random.sample(indices, N_INITIAL)
        indices_pool = [i for i in indices if i not in indices_learn]

        top_k_count = sum(1 for i in indices_learn if i in top_indices)
        run_coverage = [(top_k_count / len(top_indices))]

        for iter_num in range(len(indices_pool)):
            X_train_raw = X_feature[indices_learn]
            y_train_raw = y_objective[indices_learn]

            # 1. Normalize data
            scaler_x = preprocessing.StandardScaler().fit(X_train_raw)
            X_train_norm = scaler_x.transform(X_train_raw)

            y_train_raw_reshaped = y_train_raw.reshape(-1, 1)
            scaler_y = preprocessing.StandardScaler().fit(y_train_raw_reshaped)
            y_train_norm = scaler_y.transform(y_train_raw_reshaped)

            # 2. Train Surrogate
            try:
                if surrogate_type == 'GP-sklearn':
                    model = build_sklearn_gp_surrogate(X_train_norm, y_train_norm, X_feature.shape[1], seed)
                elif surrogate_type == 'RF':
                    model = build_random_forest_surrogate(X_train_norm, y_train_norm, seed)
                else:
                    raise ValueError(f"Unknown surrogate type: {surrogate_type}")

            except (np.linalg.LinAlgError, ValueError) as e:
                # Silent fallback to random selection
                next_index = random.choice(indices_pool)

                indices_learn.append(next_index)
                indices_pool.remove(next_index)
                if next_index in top_indices:
                    top_k_count += 1
                run_coverage.append(top_k_count / len(top_indices))
                continue

            # 3. Acquisition Function - FULLY VECTORIZED
            try:
                X_pool_norm = scaler_x.transform(X_feature[indices_pool])

                if surrogate_type == 'GP-sklearn':
                    # GP: LCB acquisition with uncertainty
                    mean_pool, std_pool = model.predict(X_pool_norm, return_std=True)
                    ac_values = -mean_pool + (ACQUISITION_LCB_RATIO * std_pool)
                elif surrogate_type == 'RF':
                    # RF: Greedy acquisition (maximize predicted value)
                    mean_pool = model.predict(X_pool_norm)
                    ac_values = -mean_pool  # Negate because we maximize objective

                best_idx_in_pool = np.argmax(ac_values)
                next_index = indices_pool[best_idx_in_pool]

            except Exception as e:
                # Silent fallback to random selection
                next_index = random.choice(indices_pool)

            # 4. Update Sets
            indices_learn.append(next_index)
            indices_pool.remove(next_index)

            if next_index in top_indices:
                top_k_count += 1
            
            run_coverage.append(top_k_count / len(top_indices))

        # Ensure coverage matches total length
        final_coverage = [run_coverage[0]] * N_INITIAL + run_coverage
        top_k_coverage_collection.append(final_coverage[:n_total])

    return np.array(top_k_coverage_collection)

def run_random_selection_benchmark(y_objective, top_indices, n_total, seed_list):
    """Runs the random selection benchmark across multiple seeds."""
    top_k_coverage_collection = []

    for seed in seed_list[:N_ENSEMBLE]:
        random.seed(seed)
        np.random.seed(seed)
        
        indices = list(np.arange(n_total))
        random.shuffle(indices)
        
        run_coverage = []
        top_k_count = 0
        for i, idx in enumerate(indices):
            if idx in top_indices:
                top_k_count += 1
            run_coverage.append(top_k_count / len(top_indices))
            
        top_k_coverage_collection.append(run_coverage)
        
    return np.array(top_k_coverage_collection)


# --- Metrics Computation ---
def compute_ef_curve(bo_coverage, random_coverage):
    """
    Computes Enhancement Factor at each iteration.

    EF_i = Coverage_BO(i) / Coverage_Random(i)

    Args:
        bo_coverage: (n_seeds, n_iterations) array
        random_coverage: (n_seeds, n_iterations) array

    Returns:
        ef_median: (n_iterations,) median EF across seeds
        ef_q25, ef_q75: 25th/75th percentiles
    """
    ef_curves = []
    for seed_idx in range(bo_coverage.shape[0]):
        bo_cov = bo_coverage[seed_idx, :]
        rand_cov = random_coverage[seed_idx, :]

        # Handle division by zero: if rand_cov=0, set EF=1
        ef = np.where(rand_cov > 1e-6, bo_cov / rand_cov, 1.0)
        ef_curves.append(ef)

    ef_curves = np.array(ef_curves)

    ef_median = np.median(ef_curves, axis=0)
    ef_q25 = np.percentile(ef_curves, 25, axis=0)
    ef_q75 = np.percentile(ef_curves, 75, axis=0)

    return ef_median, ef_q25, ef_q75


def compute_af_metric(bo_coverage, random_coverage, target=0.80):
    """
    Computes Acceleration Factor with conservative handling.

    AF = N_Random_to_80% / N_BO_to_80%

    Args:
        bo_coverage: (n_seeds, n_iterations) array
        random_coverage: (n_seeds, n_iterations) array
        target: Coverage threshold (default 0.80 for 80%)

    Returns:
        af_median: scalar, median AF across valid seeds
        af_q25, af_q75: scalars, quartiles
        n_valid: number of seeds where both methods reached target
    """
    af_values = []

    for seed_idx in range(bo_coverage.shape[0]):
        bo_cov = bo_coverage[seed_idx, :]
        rand_cov = random_coverage[seed_idx, :]

        # Find first iteration where coverage >= target
        bo_iter = np.where(bo_cov >= target)[0]
        rand_iter = np.where(rand_cov >= target)[0]

        # Conservative: only compute if BOTH reach target
        if len(bo_iter) > 0 and len(rand_iter) > 0:
            n_bo = bo_iter[0] + 1  # +1 for 1-indexed
            n_rand = rand_iter[0] + 1
            af_values.append(n_rand / n_bo)

    if len(af_values) == 0:
        return np.nan, np.nan, np.nan, 0

    af_values = np.array(af_values)
    return (
        np.median(af_values),
        np.percentile(af_values, 25),
        np.percentile(af_values, 75),
        len(af_values)
    )


def save_metrics_csv(iterations, bo_coverage, random_coverage, dataset_name, surrogate_name):
    """Exports coverage and EF data to CSV."""
    ef_median, ef_q25, ef_q75 = compute_ef_curve(bo_coverage, random_coverage)

    df = pd.DataFrame({
        'iteration': iterations,
        'bo_coverage_median': np.median(bo_coverage, axis=0),
        'bo_coverage_q25': np.percentile(bo_coverage, 25, axis=0),
        'bo_coverage_q75': np.percentile(bo_coverage, 75, axis=0),
        'random_coverage_median': np.median(random_coverage, axis=0),
        'random_coverage_q25': np.percentile(random_coverage, 25, axis=0),
        'random_coverage_q75': np.percentile(random_coverage, 75, axis=0),
        'ef_median': ef_median,
        'ef_q25': ef_q25,
        'ef_q75': ef_q75
    })

    filename = f'outputs/metrics/{dataset_name}_{surrogate_name}_metrics.csv'
    df.to_csv(filename, index=False)
    print(f"Saved metrics to {filename}")


# --- Visualization ---
def plot_discovery_rate(gp_coverage, random_coverage, n_total, dataset_name, surrogate_name='GP-ARD'):
    """Plots the median and IQR of top-k coverage and EF for BO vs Random."""
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10),
                                    gridspec_kw={'height_ratios': [2, 1]})

    iterations = np.arange(1, n_total + 1)

    # --- Top Panel: Coverage ---
    # BO Plotting
    gp_median = np.median(gp_coverage, axis=0)
    gp_q25 = np.percentile(gp_coverage, 25, axis=0)
    gp_q75 = np.percentile(gp_coverage, 75, axis=0)
    ax1.plot(iterations, gp_median, 'b-', label=surrogate_name)
    ax1.fill_between(iterations, gp_q25, gp_q75, color='b', alpha=0.2)

    # Random Selection Plotting
    random_median = np.median(random_coverage, axis=0)
    random_q25 = np.percentile(random_coverage, 25, axis=0)
    random_q75 = np.percentile(random_coverage, 75, axis=0)
    ax1.plot(iterations, random_median, 'k--', label='Random')
    ax1.fill_between(iterations, random_q25, random_q75, color='k', alpha=0.2)

    ax1.set_ylabel(f'Top {TOP_K_PERCENT*100:.0f}% Coverage', fontsize=14)
    ax1.set_title(f'Discovery Rate: {dataset_name}', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, n_total)

    # --- Bottom Panel: Enhancement Factor ---
    ef_median, ef_q25, ef_q75 = compute_ef_curve(gp_coverage, random_coverage)

    ax2.plot(iterations, ef_median, 'g-', label='Enhancement Factor', linewidth=2)
    ax2.fill_between(iterations, ef_q25, ef_q75, color='g', alpha=0.2)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Baseline (EF=1)')

    ax2.set_xlabel('Number of Experiments', fontsize=14)
    ax2.set_ylabel('EF (BO/Random)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, n_total)

    plt.tight_layout()
    plt.savefig(f"outputs/plots/{dataset_name.replace(' ', '_')}_discovery_rate_{surrogate_name}.png", dpi=300)
    plt.close(fig)

def plot_manifold(ds_grouped, feature_names, objective_name, dataset_name_str):
    """Performs PCA and creates a 3D scatter plot."""
    s_scaler = preprocessing.StandardScaler()
    ds_normalized_values = s_scaler.fit_transform(ds_grouped[feature_names + [objective_name]].values)
    ds_normalized = pd.DataFrame(ds_normalized_values, columns=feature_names + [objective_name])

    pca = decomposition.PCA(n_components=3, random_state=42)
    X_pca_values = pca.fit_transform(ds_normalized[feature_names])

    X_pca = pd.DataFrame()
    X_pca['PC1'] = X_pca_values[:, 0]
    X_pca['PC2'] = X_pca_values[:, 1]
    X_pca['PC3'] = X_pca_values[:, 2]
    X_pca[objective_name] = ds_normalized[objective_name]

    fig = px.scatter_3d(X_pca, x='PC1', y='PC2', z='PC3',
                        color=objective_name, opacity=0.7,
                        title=f'PCA Manifold Visualization: {dataset_name_str}')

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    fig.write_image(f"outputs/plots/{dataset_name_str.replace(' ', '_')}_manifold.png", scale=2)


# --- Main Execution Block ---
def main():
    """Main function to run benchmarks and generate plots for all datasets."""
    overall_start = time.time()
    af_summary = []  # Collect AF results for final table

    # Create output directories
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)

    for name, config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {name}")
        print(f"{'='*60}")

        # Load and process data
        ds_grouped, f_names, obj_name, X, y = load_and_process_data(config)
        n_total = len(ds_grouped)
        n_top = int(math.ceil(n_total * TOP_K_PERCENT))
        top_indices = list(ds_grouped.sort_values(obj_name).head(n_top).index)

        print(f"Total samples: {n_total}, Top {TOP_K_PERCENT*100:.0f}% samples: {n_top}")

        # Run Random baseline ONCE (shared across surrogates)
        print("\n--- Running Random Selection benchmark ---")
        start_time = time.time()
        random_coverage = run_random_selection_benchmark(y, top_indices, n_total, SEED_LIST)
        random_time = time.time() - start_time
        print(f"Random finished in {random_time:.2f}s")

        # Run each surrogate
        for surrogate_type in SURROGATES_TO_RUN:
            print(f"\n--- Running {surrogate_type} benchmark ---")
            start_time = time.time()

            bo_coverage = run_benchmark_generic(
                X, y, top_indices, n_total, SEED_LIST,
                surrogate_type=surrogate_type
            )

            bo_time = time.time() - start_time
            print(f"{surrogate_type} finished in {bo_time:.2f}s ({bo_time/60:.2f} min)")

            # Compute AF metric
            af_med, af_q25, af_q75, n_valid = compute_af_metric(bo_coverage, random_coverage)
            af_summary.append({
                'dataset': name,
                'surrogate': surrogate_type,
                'af_median': af_med,
                'af_q25': af_q25,
                'af_q75': af_q75,
                'n_valid_seeds': n_valid
            })
            print(f"AF: {af_med:.2f} [{af_q25:.2f}, {af_q75:.2f}] (n={n_valid}/{N_ENSEMBLE})")

            # Save metrics CSV
            iterations = np.arange(1, n_total + 1)
            save_metrics_csv(iterations, bo_coverage, random_coverage,
                           name.replace(' ', '_'), surrogate_type)

            # Generate plots
            print("Generating discovery rate plot...")
            plot_discovery_rate(bo_coverage, random_coverage, n_total, name, surrogate_type)

        # Manifold plot (once per dataset)
        print("Generating manifold plot...")
        plot_manifold(ds_grouped, f_names, obj_name, name)
        print(f"All plots saved for {name}")

    # Save AF summary table
    af_df = pd.DataFrame(af_summary)
    af_df.to_csv('outputs/metrics/af_summary.csv', index=False)
    print(f"\n{'='*60}")
    print("AF Summary:")
    print(f"{'='*60}")
    print(af_df.to_string(index=False))
    print(f"\nAF summary saved to outputs/metrics/af_summary.csv")

    print(f"\n{'='*60}")
    print(f"ALL DATASETS COMPLETE - Total time: {(time.time() - overall_start)/60:.2f} min")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()