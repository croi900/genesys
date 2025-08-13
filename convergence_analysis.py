import argparse
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plots will be skipped")

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, normal fitting will be skipped")

from models import (
    M1V0,
    M1VEXP,
    M1VPHI2,
    M1VPHI24,
    M2V0,
    M2VEXP,
    M2VPHI2,
    M2VPHI24,
    M3V0,
    M3VEXP,
    M3VPHI2,
    M3VPHI24,
)
from models.potential import PotentialModel


@dataclass
class DataConfig:
    data_dir: str
    min_samples_per_runid: int
    max_runids: int
    random_seed: int


def compute_split_rhat(chains: np.ndarray) -> np.ndarray:
    """
    Compute split-chain Gelman–Rubin R-hat per parameter.

    Parameters
    ----------
    chains : np.ndarray
        Array of shape (nchains, nsteps, ndim), AFTER burn-in and thinning.

    Returns
    -------
    np.ndarray
        R-hat values of shape (ndim,).
    """
    if chains.ndim != 3:
        raise ValueError("chains must be (nchains, nsteps, ndim)")

    nchains, nsteps, ndim = chains.shape
    if nsteps < 4:
        raise ValueError("Too few steps after burn-in to compute R-hat (need >= 4)")

    half = nsteps // 2
    if half < 2:
        raise ValueError("Too few steps per split-chain (need >= 2)")

    split_chains = np.concatenate([chains[:, :half, :], chains[:, -half:, :]], axis=0)
    m = split_chains.shape[0]
    n = split_chains.shape[1]

    chain_means = split_chains.mean(axis=1)

    chain_vars = split_chains.var(axis=1, ddof=1)

    grand_means = chain_means.mean(axis=0)

    B = n * ((chain_means - grand_means) ** 2).sum(axis=0) / (m - 1)
    W = chain_vars.mean(axis=0)

    var_hat = ((n - 1) / n) * W + (B / n)

    rhat = np.sqrt(var_hat / W)
    return rhat


def load_sql_dump_to_memory(sql_file: str) -> sqlite3.Connection:
    """Load SQL dump into in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    with open(sql_file, "r") as f:
        sql_content = f.read()

    sql_content = re.sub(r"INSERT INTO \w+\.", "INSERT INTO ", sql_content)
    conn.executescript(sql_content)
    return conn


def get_database_columns(model_name: str) -> List[str]:
    """Get the actual database column names for a given model."""
    if "vexp" in model_name:
        return ["x", "z"]
    elif "vphi24" in model_name:
        return ["x", "z"]
    elif "vphi2" in model_name:
        return ["x"]
    else:
        return []


def get_display_names(model_name: str) -> List[str]:
    """Get the display names for parameters of a given model."""
    if "vexp" in model_name:
        return ["σ", "μ"]
    elif "vphi24" in model_name:
        return ["γ", "δ"]
    elif "vphi2" in model_name:
        return ["m"]
    else:
        return []


def get_actual_prior_bounds(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get the actual prior bounds used in the MCMC runs from main.py."""
    prior_bounds_map = {
        "m1vphi2": ([-1], [1]),
        "m1vphi24": ([-3, -3], [3, 3]),
        "m1vexp": ([-5, -5], [5, 5]),
        "m2vphi2": ([-0.5], [0.5]),
        "m2vphi24": ([-4, -4], [4, 4]),
        "m2vexp": ([-5, -5], [5, 5]),
        "m3vphi2": ([-0.5], [0.5]),
        "m3vphi24": ([-4, -4], [4, 4]),
        "m3vexp": ([-5, -5], [5, 5]),
    }

    if model_name not in prior_bounds_map:
        raise ValueError(f"Unknown model: {model_name}")

    lower, upper = prior_bounds_map[model_name]
    return np.array(lower), np.array(upper)


def load_mcmc_chains(
    model_name: str, data_dir: str, config: DataConfig
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Load MCMC chains from SQL dump files.

    Returns:
        chains: array of shape (nchains, nsteps, ndim) where each chain corresponds to a runid
        prior_bounds: dict with 'lower' and 'upper' bounds inferred from data
    """
    sql_file = os.path.join(data_dir, f"{model_name}.sql")
    if not os.path.exists(sql_file):
        raise FileNotFoundError(f"SQL file not found: {sql_file}")

    conn = load_sql_dump_to_memory(sql_file)
    db_cols = get_database_columns(model_name)

    if not db_cols:
        raise ValueError(f"Model {model_name} has no free parameters")

    cursor = conn.execute(f"SELECT DISTINCT runid FROM {model_name} ORDER BY runid")
    runids = [row[0] for row in cursor.fetchall()]

    if len(runids) > config.max_runids:
        rng = np.random.default_rng(config.random_seed)
        runids = rng.choice(runids, size=config.max_runids, replace=False).tolist()

    chains = []
    all_params = []

    for runid in runids:
        param_str = ", ".join(db_cols)
        query = f"SELECT {param_str} FROM {model_name} WHERE runid = ? ORDER BY id"
        cursor = conn.execute(query, (runid,))
        samples = np.array(cursor.fetchall())

        if len(samples) >= config.min_samples_per_runid:
            chains.append(samples)
            all_params.append(samples)

    conn.close()

    if not chains:
        raise ValueError("No chains with sufficient samples found")

    min_length = min(len(chain) for chain in chains)
    chains = [chain[:min_length] for chain in chains]
    chains = np.array(chains)

    lower_bounds, upper_bounds = get_actual_prior_bounds(model_name)

    prior_bounds = {"lower": lower_bounds, "upper": upper_bounds}

    return chains, prior_bounds


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_representative_trace(
    chains: np.ndarray,
    outdir: str,
    model_name: str,
    param_names: List[str],
    max_walkers_to_plot: int = 8,
) -> Optional[str]:
    """
    Plot representative trace plots for all parameters.
    For single-parameter models: one plot
    For two-parameter models: two subplots, one above the other
    Returns the file path of the generated figure, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping trace plot (matplotlib not available)")
        return None

    nchains, nsteps, ndim = chains.shape
    idxs = np.linspace(0, nchains - 1, num=min(nchains, max_walkers_to_plot), dtype=int)

    if ndim == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes = [axes]
    else:
        fig, axes = plt.subplots(ndim, 1, figsize=(10, 4 * ndim))
        if ndim == 1:
            axes = [axes]

    for d in range(ndim):
        ax = axes[d]
        for idx in idxs:
            ax.plot(
                chains[idx, :, d],
                alpha=0.7,
                lw=0.8,
                label=f"Chain {idx + 1}" if d == 0 else "",
            )

        ax.set_title(f"{model_name.upper()} — Parameter {param_names[d]}", fontsize=15)
        ax.set_xlabel("Step", fontsize=12.5)
        ax.set_ylabel(param_names[d], fontsize=12.5)
        ax.grid(True, alpha=0.3)

        if d == 0 and len(idxs) > 1:
            ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()

    outpath = os.path.join(outdir, f"{model_name}_trace_representative.eps")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def create_rhat_table(
    all_models: List[str], data_dir: str, config: DataConfig, outdir: str
) -> Tuple[str, str]:
    """
    Create a comprehensive table of R-hat values for all models.

    Args:
        all_models: List of model names to analyze
        data_dir: Directory containing SQL dump files
        outdir: Output directory for the table

    Returns:
        Tuple of (png_table_path, csv_table_path)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping R-hat table (matplotlib not available)")
        return ""

    table_data = []

    for model_name in all_models:
        try:
            chains, _ = load_mcmc_chains(model_name, data_dir, config)
            chains_post, _ = process_loaded_chains(chains, 0.25)

            rhat = compute_split_rhat(chains_post)

            display_names = get_display_names(model_name)

            row = [model_name.upper()]
            for i, param in enumerate(display_names):
                if i < len(rhat):
                    row.append(f"{rhat[i]:.3f}")
                else:
                    row.append("N/A")

            while len(row) < 4:
                row.append("")

            table_data.append(row)

        except Exception as e:
            print(f"Warning: Could not process model {model_name}: {e}")

            row = [model_name.upper(), "ERROR", "ERROR", ""]
            table_data.append(row)

    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.5 + 2))
    ax.axis("tight")
    ax.axis("off")

    headers = ["Model", "Param 1", "Param 2", "Param 3"]

    table = ax.table(
        cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for i, row in enumerate(table_data):
        for j, cell in enumerate(row[1:3]):
            if cell != "ERROR" and cell != "":
                try:
                    rhat_val = float(cell)
                    if rhat_val < 1.05:
                        table[(i + 1, j + 1)].set_facecolor("#d4edda")
                    elif rhat_val < 1.1:
                        table[(i + 1, j + 1)].set_facecolor("#fff3cd")
                    else:
                        table[(i + 1, j + 1)].set_facecolor("#f8d7da")
                except ValueError:
                    pass

    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#e9ecef")
        table[(0, j)].set_text_props(weight="bold")

    ax.set_title(
        "MCMC Convergence Diagnostics: Gelman-Rubin R-hat Values",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="#d4edda", label="Excellent (R-hat < 1.05)"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="#fff3cd", label="Warning (1.05 ≤ R-hat < 1.1)"
        ),
        plt.Rectangle((0, 0), 1, 1, facecolor="#f8d7da", label="Poor (R-hat ≥ 1.1)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 0.98))

    png_outpath = os.path.join(outdir, "all_models_rhat_table.eps")
    fig.savefig(png_outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    csv_outpath = os.path.join(outdir, "all_models_rhat_table.csv")
    import csv

    with open(csv_outpath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            ["Model", "Param_1", "Param_2", "Param_3", "Convergence_Status"]
        )

        for row in table_data:
            model_name = row[0]
            param1 = row[1] if len(row) > 1 and row[1] != "" else "N/A"
            param2 = row[2] if len(row) > 2 and row[2] != "" else "N/A"
            param3 = row[3] if len(row) > 3 and row[3] != "" else "N/A"

            status = "ERROR"
            if param1 != "ERROR" and param1 != "N/A":
                try:
                    rhat_val = float(param1)
                    if rhat_val < 1.05:
                        status = "EXCELLENT"
                    elif rhat_val < 1.1:
                        status = "WARNING"
                    else:
                        status = "POOR"
                except ValueError:
                    pass

            writer.writerow([model_name, param1, param2, param3, status])

    return png_outpath, csv_outpath


def plot_prior_posterior_overlays(
    posterior_samples: np.ndarray,
    prior_bounds: Dict[str, np.ndarray],
    outdir: str,
    model_name: str,
    param_names: List[str],
) -> List[str]:
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping prior/posterior overlay plots (matplotlib not available)")
        return []

    if not SCIPY_AVAILABLE:
        print("Skipping prior/posterior overlay plots (scipy not available)")
        return []

    paths: List[str] = []
    ndim = posterior_samples.shape[1]

    if ndim == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(ndim, 1, figsize=(8, 4 * ndim))
        if ndim == 1:
            axes = [axes]

    for d in range(ndim):
        ax = axes[d]

        lower_bound = prior_bounds["lower"][d]
        upper_bound = prior_bounds["upper"][d]

        x_range = upper_bound - lower_bound
        x_min = lower_bound - 0.1 * x_range
        x_max = upper_bound + 0.1 * x_range
        x = np.linspace(x_min, x_max, 1000)

        prior_height = 1.0 / (upper_bound - lower_bound)
        ax.fill_between(
            [lower_bound, upper_bound],
            [0, 0],
            [prior_height, prior_height],
            alpha=0.3,
            color="#1f77b4",
            label="Prior (Uniform)",
            step="mid",
        )
        ax.plot(
            [lower_bound, lower_bound, upper_bound, upper_bound],
            [0, prior_height, prior_height, 0],
            color="#1f77b4",
            linewidth=2,
        )

        posterior_param = posterior_samples[:, d]
        mu, sigma = stats.norm.fit(posterior_param)

        posterior_pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(
            x,
            posterior_pdf,
            color="#ff7f0e",
            linewidth=2,
            label=f"Posterior (Normal)\nμ={mu:.4f}, σ={sigma:.4f}",
        )
        ax.fill_between(x, 0, posterior_pdf, alpha=0.3, color="#ff7f0e")

        ax.set_xlabel(f"{param_names[d]}", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        ax.axvline(lower_bound, color="#1f77b4", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(upper_bound, color="#1f77b4", linestyle="--", alpha=0.7, linewidth=1)

    fig.tight_layout()

    outpath = os.path.join(outdir, f"{model_name}_prior_posterior_overlay.eps")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(outpath)

    return paths


MODEL_REGISTRY: Dict[str, PotentialModel.__class__] = {
    "m1v0": M1V0,
    "m1vexp": M1VEXP,
    "m1vphi2": M1VPHI2,
    "m1vphi24": M1VPHI24,
    "m2v0": M2V0,
    "m2vexp": M2VEXP,
    "m2vphi2": M2VPHI2,
    "m2vphi24": M2VPHI24,
    "m3v0": M3V0,
    "m3vexp": M3VEXP,
    "m3vphi2": M3VPHI2,
    "m3vphi24": M3VPHI24,
}


def infer_ndim(model_class: PotentialModel.__class__) -> int:
    name = (
        model_class.to_string()
        if hasattr(model_class, "to_string")
        else model_class.__name__
    )
    if "vexp" in name:
        return 2
    if "phi24" in name:
        return 2
    if "phi2" in name:
        return 1
    return 0


def process_loaded_chains(
    chains: np.ndarray, burn_in_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process loaded chains by removing burn-in and creating flat samples.

    Args:
        chains: array of shape (nchains, nsteps, ndim)
        burn_in_frac: fraction of chain to discard as burn-in

    Returns:
        chains_post: chains after burn-in removal
        flat_samples: flattened posterior samples
    """
    nchains, nsteps, ndim = chains.shape
    burn_in_steps = int(burn_in_frac * nsteps)

    if burn_in_steps >= nsteps:
        raise ValueError("burn-in fraction too large")

    chains_post = chains[:, burn_in_steps:, :]
    flat_samples = chains_post.reshape(-1, ndim)

    return chains_post, flat_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convergence diagnostics and prior-vs-posterior analysis using existing MCMC data: "
            "computes Gelman–Rubin R-hat, generates a representative trace plot, and "
            "overlays prior and posterior distributions."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="m1vexp",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model to analyze",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing SQL dump files",
    )
    parser.add_argument(
        "--min_samples", type=int, default=50, help="Minimum samples per chain (runid)"
    )
    parser.add_argument(
        "--max_chains", type=int, default=10, help="Maximum number of chains to load"
    )
    parser.add_argument(
        "--burnin_frac",
        type=float,
        default=0.25,
        help="Fraction of chain to discard as burn-in",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("conv_output", "convergence"),
        help="Directory for saving plots",
    )
    parser.add_argument(
        "--create_table",
        action="store_true",
        help="Create comprehensive R-hat table for all models",
    )

    args = parser.parse_args()

    model_class = MODEL_REGISTRY[args.model]

    cfg = DataConfig(
        data_dir=args.data_dir,
        min_samples_per_runid=args.min_samples,
        max_runids=args.max_chains,
        random_seed=args.seed,
    )

    ensure_outdir(args.outdir)

    print(f"Loading chains for model {args.model} from {args.data_dir}...")
    chains, prior_bounds = load_mcmc_chains(args.model, args.data_dir, cfg)
    print(
        f"Loaded {chains.shape[0]} chains with {chains.shape[1]} samples each, {chains.shape[2]} parameters"
    )

    chains_post, flat_samples = process_loaded_chains(chains, 0.1)
    print(f"After burn-in removal: {chains_post.shape[1]} samples per chain")

    rhat = compute_split_rhat(chains_post)

    display_names = get_display_names(args.model)
    if not display_names:
        raise ValueError(f"Model {args.model} has no free parameters")

    trace_path = plot_representative_trace(
        chains_post, args.outdir, args.model, display_names
    )

    overlay_paths = plot_prior_posterior_overlays(
        posterior_samples=flat_samples,
        prior_bounds=prior_bounds,
        outdir=args.outdir,
        model_name=args.model,
        param_names=display_names,
    )

    print("\n=== Convergence diagnostics ===")
    for name, val in zip(display_names, rhat):
        print(f"R-hat[{name}] = {val:.3f}")
    print(
        "(Rule of thumb: R-hat close to 1.0 indicates convergence; < 1.05 is commonly acceptable.)"
    )

    import json

    rhat_results = {
        "model": args.model,
        "parameters": display_names,
        "rhat_values": rhat.tolist(),
        "convergence_status": [
            "excellent" if r < 1.05 else "warning" if r < 1.1 else "poor" for r in rhat
        ],
    }

    rhat_file = os.path.join(args.outdir, f"{args.model}_rhat.json")
    with open(rhat_file, "w") as f:
        json.dump(rhat_results, f, indent=2)
    print(f"R-hat results saved to: {rhat_file}")

    print("\n=== Representative trace plot ===")
    if trace_path:
        print(f"Saved: {trace_path}")
    else:
        print("Skipped (matplotlib not available)")

    print("\n=== Priors specification ===")
    print("Actual priors used in MCMC (from main.py):")
    for i, (name, lower, upper) in enumerate(
        zip(display_names, prior_bounds["lower"], prior_bounds["upper"])
    ):
        print(f"  {name}: uniform U({lower}, {upper})")
    print(f"Plus model-specific constraints via {args.model}.mcmc_constraints()")

    print("\n=== Prior vs Posterior overlays ===")
    if overlay_paths:
        for pth in overlay_paths:
            print(f"Saved: {pth}")
    else:
        print("Skipped (matplotlib not available)")

    print(
        "\nInterpretation notes: If the posterior density is concentrated away from the prior bounds and "
        "differs substantially from the (uniform) prior histogram, this indicates that the data "
        "dominate the posterior (i.e., results are not prior-dominated). For sensitivity analysis, "
        "compare results with different prior ranges."
    )

    if args.create_table:
        print("\n=== Creating comprehensive R-hat table ===")
        all_models = sorted(MODEL_REGISTRY.keys())
        table_paths = create_rhat_table(all_models, args.data_dir, cfg, args.outdir)
        if table_paths:
            png_path, csv_path = table_paths
            print(f"R-hat table (PNG) saved to: {png_path}")
            print(f"R-hat table (CSV) saved to: {csv_path}")
        else:
            print("R-hat table creation failed")


if __name__ == "__main__":
    main()
