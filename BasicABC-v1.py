"""
Basic Rejection Approximate Bayesian Computation (ABC)

This script performs Basic Rejection ABC. It:
1. Loads the observed data and formats it into arrays.
2. Computes summary statistics for the observed data (averaged across the 40 observations).
3. Loads the 1,000,000 simulated priors from 'full_reference_table.npz'.
4. Computes the summary statistics across all 1,000,000 simulations.
5. Normalizes the stats and computes Euclidean distances.
6. Rejects all but the top 0.1% closest simulations.
7. Plots the resulting approximate posterior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# Hyperparameters
# =====================================================================
REFERENCE_TABLE_PATH = "full_reference_table.npz"
QUANTILES_TOLERANCE = 0.001  # Keep the top 0.1% of simulations


def load_observed_data():
    """
    Reads the observed data CSVs and pivots them from 'long' format 
    into 2D NumPy matrices of shape (40_replicates, time_steps/degrees).
    """
    print("Loading observed data...")
    df_inf = pd.read_csv("data/infected_timeseries.csv")
    df_rew = pd.read_csv("data/rewiring_timeseries.csv")
    df_deg = pd.read_csv("data/final_degree_histograms.csv")

    # Pivot rows into grids
    obs_inf = df_inf.pivot(index="replicate_id", columns="time", values="infected_fraction").values
    obs_rew = df_rew.pivot(index="replicate_id", columns="time", values="rewire_count").values
    obs_deg = df_deg.pivot(index="replicate_id", columns="degree", values="count").values
    
    return obs_inf, obs_rew, obs_deg


def compute_summary_statistics(infected, rewires, degrees):
    """
    Computes a vector of summary statistics for the data.
    
    Parameters
    ----------
    infected : np.ndarray, shape (N, 201)
    rewires  : np.ndarray, shape (N, 201)
    degrees  : np.ndarray, shape (N, 31)
    
    Returns
    -------
    stats : np.ndarray, shape (N, 4)
        The computed summary statistics:[max_infection, total_infection, total_rewires, mean_degree]
    """

    # 1. Peak infection fraction
    max_inf = np.max(infected, axis=1)
    
    # 2. Total infection burden
    sum_inf = np.sum(infected, axis=1)
    
    # 3. Total number of rewiring events
    sum_rew = np.sum(rewires, axis=1)
    
    # 4. Mean final degree of the network
    # We multiply the bin index (0 to 30) by the counts in that bin
    bins = np.arange(31)
    total_nodes = np.sum(degrees, axis=1) # Always 200, but good to be robust
    mean_deg = np.sum(degrees * bins, axis=1) / total_nodes
    
    # Stack them horizontally into an (N, 4) matrix
    return np.column_stack([max_inf, sum_inf, sum_rew, mean_deg])


def main():
    # -----------------------------------------------------------------
    # 1. Prepare Observed Statistics
    # -----------------------------------------------------------------
    obs_inf, obs_rew, obs_deg = load_observed_data()
    
    # Compute stats for all 40 real replicates
    obs_stats_all = compute_summary_statistics(obs_inf, obs_rew, obs_deg)
    
    # Take the mean across the 40 replicates to get our TARGET statistic
    target_stat = np.mean(obs_stats_all, axis=0)
    print(f"Target Summary Stats (Max Inf, Sum Inf, Sum Rew, Mean Deg): \n{target_stat}\n")

    # -----------------------------------------------------------------
    # 2. Load Simulated Reference Table
    # -----------------------------------------------------------------
    print(f"Loading simulated reference table from '{REFERENCE_TABLE_PATH}'...")
    data = np.load(REFERENCE_TABLE_PATH)
    
    sim_betas = data["betas"]
    sim_gammas = data["gammas"]
    sim_rhos = data["rhos"]
    
    # Compute stats for all 1,000,000 simulations simultaneously
    print("Computing summary statistics for all simulated data...")
    sim_stats = compute_summary_statistics(data["infected"], data["rewires"], data["degrees"])
    
    # Free up memory
    del data 
    
    # -----------------------------------------------------------------
    # 3. Normalisation and Distance Calculation
    # -----------------------------------------------------------------
    print("Normalising statistics (computing Z-scores) and calculating distances...")
    # Calculate mean and std of the simulated statistics
    stat_means = np.mean(sim_stats, axis=0)
    stat_stds = np.std(sim_stats, axis=0)
    
    # Z-score normalisation
    sim_stats_norm = (sim_stats - stat_means) / stat_stds
    target_stat_norm = (target_stat - stat_means) / stat_stds
    
    # Calculate Euclidean distance (L2 norm) across the rows
    distances = np.linalg.norm(sim_stats_norm - target_stat_norm, axis=1)
    
    # -----------------------------------------------------------------
    # 4. ABC Rejection Step
    # -----------------------------------------------------------------
    # Find the distance threshold corresponding to the top 0.1%
    threshold = np.quantile(distances, QUANTILES_TOLERANCE)
    
    # Create a boolean mask of accepted simulations
    accepted_mask = distances <= threshold
    
    post_betas = sim_betas[accepted_mask]
    post_gammas = sim_gammas[accepted_mask]
    post_rhos = sim_rhos[accepted_mask]
    
    num_accepted = len(post_betas)
    print(f"Accepted {num_accepted} samples out of {len(distances)}")
    print(f"Distance Threshold (\u03b5): {threshold:.4f}\n")
    
    print("Posterior Means:")
    print(f"  Beta:  {np.mean(post_betas):.4f}")
    print(f"  Gamma: {np.mean(post_gammas):.4f}")
    print(f"  Rho:   {np.mean(post_rhos):.4f}")
    
    # -----------------------------------------------------------------
    # 5. Plotting the Approximate Posterior
    # -----------------------------------------------------------------
    print("Plotting posterior distributions...")
    
    # Put data into a DataFrame for easy seaborn plotting
    df_posterior = pd.DataFrame({
        r"$\beta$ (Infection Prob)": post_betas,
        r"$\gamma$ (Recovery Prob)": post_gammas,
        r"$\rho$ (Rewiring Prob)": post_rhos
    })
    
    # Create a PairGrid (pairwise scatter plots and marginal histograms)
    g = sns.PairGrid(df_posterior, corner=True, diag_sharey=False)
    
    # Upper/Lower corner: KDE density plots to show correlations
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues", alpha=0.8)
    g.map_lower(sns.scatterplot, s=5, color=".15", alpha=0.3)
    
    # Diagonal: Marginal histograms
    g.map_diag(sns.histplot, kde=True, color="steelblue", bins=30)
    
    g.fig.suptitle("Basic Rejection ABC: Approximate Posteriors", y=1.02, fontsize=16)
    
    plt.tight_layout()
    plt.savefig("./diagrams/posterior_pairplot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()