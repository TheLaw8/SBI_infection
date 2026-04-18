"""
Basic Rejection Approximate Bayesian Computation (ABC) with Regression Adjustment

This script performs Basic Rejection ABC and applies a local linear regression 
adjustment (Beaumont et al., 2002). It:
1. Loads the observed data and formats it into arrays.
2. Computes summary statistics for the observed data (averaged across the 40 observations).
3. Loads the 1,000,000 simulated priors from 'full_reference_table.npz'.
4. Computes the summary statistics across all 1,000,000 simulations.
5. Computes the Mahalanobis distance to appropriately handle correlated summary stats.
6. Rejects all but the top 0.1% closest simulations.
7. NEW ------> Applies Epanechnikov-weighted local linear regression to adjust accepted parameters.
8. Plots the resulting approximate posterior.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.linear_model import LinearRegression

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
    Computes an optimised 4-dimensional vector of summary statistics.
    
    This specific combination of statistics was chosen to break the structural 
    unidentifiability between beta (transmission) and rho (rewiring) while 
    extracting a clean signal for gamma (recovery) and avoiding the stochastic 
    noise inherent in small (N=200) random graphs.
    
    Parameters
    ----------
    infected : np.ndarray, shape (N, 201)
        Fraction of infected nodes at each time step.
    rewires  : np.ndarray, shape (N, 201)
        Count of successful rewiring events at each time step.
    degrees  : np.ndarray, shape (N, 31)
        Histogram of final degrees (counts per bin 0-30).
    
    Returns
    -------
    stats : np.ndarray, shape (N, 4)[max_inf, sum_inf, max_rew, gini_deg]
    """

    # =================================================================
    # 1. PEAK INFECTION FRACTION (Main target: Beta, paritally Rho)
    # =================================================================
    # Purpose: Captures the maximum severity (the "height") of the epidemic.
    # Contribution: This is the primary signal for \beta. A highly infectious 
    # disease will produce a massive peak, while a weak disease will stay low.
    # Note: It is partially coupled with \rho, as a high rho (lots of social 
    # distancing) can also suppress the peak.
    max_inf = np.max(infected, axis=1)
    
    # =================================================================
    # 2. TOTAL INFECTION BURDEN (Main target: Gamma)
    # =================================================================
    # Purpose: Calculates the area under the infection curve (total infected-days).
    # Contribution: This is an effective signal for \gamma. Because \gamma 
    # dictates the duration of an infection (1/\gamma), it strictly controls 
    # how long the epidemic lingers. A low \gamma means infections last longer, 
    # directly inflating this sum regardless of network topology.
    sum_inf = np.sum(infected, axis=1)
    
    # =================================================================
    # 3. PEAK REWIRING INTENSITY (Main target: Rho)
    # =================================================================
    # Purpose: Measures the maximum 'avoidance' response in a single time step.
    # Contribution: This heavily targets \rho and decouples it from \beta. 
    # While total rewiring might look similar for a fast/short epidemic and a 
    # slow/long epidemic, a high \rho typically leads to a large spike 
    # in rewiring events which max_rew captures.
    max_rew = np.max(rewires, axis=1)
    
    # =================================================================
    # 4. DEGREE GINI COEFFICIENT (Main target: Rho / Structural Inequality)
    # =================================================================
    # Purpose: Measures how unequal the final degree distribution is.
    # Contribution: A standard Erdős-Rényi graph has low inequality (low Gini). 
    # As \rho increases, susceptible nodes flee infected neighbors, creating 
    # isolated infected nodes (low degree) and highly connected safe hubs 
    # (high degree). The Gini coefficient captures this structural inequality.
    
    n_nodes = 200
    bins = np.arange(31)
    
    # Total edges and mean degree for each simulation
    total_k = np.sum(degrees * bins, axis=1)
    mean_k = total_k / n_nodes
    
    # Matrix of absolute differences between all pairs of degree bins: |i - j|
    bin_diffs = np.abs(bins[:, None] - bins[None, :])

    # For formula see: https://en.wikipedia.org/wiki/Gini_coefficient#:~:text=%2C%20the-,Gini%20coefficient%20is,-%3A
    # Calculate the Gini numerator: sum_{i,j} (count_i * count_j * |bin_i - bin_j|)
    # We use vectorized matrix multiplication (np.dot) to do this instantly for all simulations.
    count_prod_diff = np.sum(np.dot(degrees, bin_diffs) * degrees, axis=1)
    
    # Final Gini formula: Numerator / (2 * N^2 * mean)
    # We add 1e-6 to the denominator to prevent division by zero in edge cases.
    gini_deg = count_prod_diff / (2 * (n_nodes**2) * mean_k + 1e-6)

    # Return summary statistics
    return np.column_stack([max_inf, sum_inf, max_rew, gini_deg])



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
    # 3. Mahalanobis Distance Calculation
    # -----------------------------------------------------------------
    print("Computing covariance matrix and Mahalanobis distances...")
    
    # Step A: Compute the covariance matrix of the simulated summary statistics
    # rowvar=False ensures it treats columns as variables
    cov_matrix = np.cov(sim_stats, rowvar=False)

    # Step B: Invert the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    # Step C: Calculate the difference between every simulation and the target
    diffs = sim_stats - target_stat 
    
    # Step D: Compute the Mahalanobis distance D = sqrt( diffs^T * inv_cov * diffs )
    # We use np.dot and element-wise multiplication to vectorize this over all rows instantly
    distances = np.sqrt(np.sum(np.dot(diffs, inv_cov_matrix) * diffs, axis=1))
    
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
    
    # -----------------------------------------------------------------
    # NEW ------> 5. Regression Adjustment (Beaumont et al., 2002)
    # See report for mathematical derivation of formulae used.
    # -----------------------------------------------------------------
    print("Applying Beaumont Regression Adjustment...")
    
    # Extract the summary statistics for the accepted samples
    S_accepted = sim_stats[accepted_mask]
    
    # Calculate the discrepancy between accepted stats and the target
    delta_S = S_accepted - target_stat
    
    # Calculate Epanechnikov kernel weights based on distance
    d_acc = distances[accepted_mask]
    weights = 1.0 - (d_acc / threshold)**2
    
    # Arrays to hold our new adjusted parameters
    adj_betas = np.zeros_like(post_betas)
    adj_gammas = np.zeros_like(post_gammas)
    adj_rhos = np.zeros_like(post_rhos)
    
    raw_params = [post_betas, post_gammas, post_rhos]
    prior_bounds = [(0.05, 0.50), (0.02, 0.20), (0.0, 0.8)]
    
    # Apply Weighted Least Squares for each parameter separately
    for i, param in enumerate(raw_params):
        reg = LinearRegression()
        reg.fit(delta_S, param, sample_weight=weights)
        
        # Adjust the parameter: theta_adj = theta - (s - s_obs) * beta
        param_adj = param - reg.predict(delta_S) + reg.intercept_ # Note: intercept here cancels out with the intercept subtracted through `reg.predict()` function
        
        # Ensure the linear adjustment doesn't push values outside prior bounds
        param_adj = np.clip(param_adj, prior_bounds[i][0], prior_bounds[i][1])
        
        if i == 0: adj_betas = param_adj
        elif i == 1: adj_gammas = param_adj
        elif i == 2: adj_rhos = param_adj

    print("Posterior Means (Rejection -> Adjusted):")
    print(f"  Beta:  {np.mean(post_betas):.4f} -> {np.mean(adj_betas):.4f}")
    print(f"  Gamma: {np.mean(post_gammas):.4f} -> {np.mean(adj_gammas):.4f}")
    print(f"  Rho:   {np.mean(post_rhos):.4f} -> {np.mean(adj_rhos):.4f}\n")
    
    # -----------------------------------------------------------------
    # 6. Plotting the Approximate Posterior & Diagnostics
    # -----------------------------------------------------------------
    print("Plotting posterior distributions and diagnostics...")
    
    os.makedirs("./diagrams", exist_ok=True)  # Ensure the directory exists!

    # Define labels and known prior bounds (from README)
    param_labels =[r"$\beta$ (Infection)", r"$\gamma$ (Recovery)", r"$\rho$ (Rewiring)"]
    prior_bounds =[(0.05, 0.50), (0.02, 0.20), (0.0, 0.8)]
    
    # USE ADJUSTED PARAMS HERE so the charts display the new data
    post_params =[adj_betas, adj_gammas, adj_rhos]
    
    # all_params: The full prior samples (needed for the diagnostic cloud)
    all_params = [sim_betas, sim_gammas, sim_rhos]
    
    top_pct = QUANTILES_TOLERANCE * 100  # Format tolerance for title (e.g., 0.1)
    
    df_posterior = pd.DataFrame({
        param_labels[0]: adj_betas,
        param_labels[1]: adj_gammas,
        param_labels[2]: adj_rhos
    })
    
    # Calculate Posterior Means and 95% Credible Intervals (CI)
    post_means =[np.mean(p) for p in post_params]
    post_cis =[(np.percentile(p, 2.5), np.percentile(p, 97.5)) for p in post_params]

    # =================================================================
    # FIGURE 1: PairPlot (The Main Posterior)
    # =================================================================
    sns.set_theme(style="white") 
    g = sns.PairGrid(df_posterior, corner=True, diag_sharey=False)
    
    # 1. Map the Diagonal (Histograms)
    g.map_diag(sns.histplot, kde=True, color="steelblue", bins=30, edgecolor='white', alpha=0.7)
    
    # 2. Map the Lower Triangle (KDE and Scatter)
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues", alpha=0.7, levels=6)
    g.map_lower(sns.scatterplot, s=10, color=".2", alpha=0.15, linewidth=0)
    
    # 3. Customize Axes (Loop through the grid)
    for i in range(len(param_labels)):
        for j in range(i + 1):  # Iterate only over active subplots
            ax = g.axes[i, j]
            if ax is None:
                continue
            
            # Force tick marks and labels on every single subplot for maximum readability
            ax.tick_params(labelbottom=True, labelleft=True, direction='out', length=4, width=1)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            
            # Explicit axis labels for every subplot
            ax.set_xlabel(param_labels[j], fontsize=11, fontweight='bold')
            if i == j:
                ax.set_ylabel("Density", fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel(param_labels[i], fontsize=11, fontweight='bold')
            
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Enhancements for the Diagonal Marginal Plots
            if i == j:
                mean_val = post_means[i]
                ax.axvline(mean_val, color='crimson', linestyle='-', linewidth=2, 
                           label=f'Mean: {mean_val:.3f}')
                
                ci_low, ci_high = post_cis[i]
                ax.axvline(ci_low, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label=f'95% CI: ({ci_low:.3f}, {ci_high:.3f})')
                ax.axvline(ci_high, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
                
                # Visual guide for prior bounds
                bound_low, bound_high = prior_bounds[i]
                ax.axvspan(bound_low, bound_high, color='gray', alpha=0.08, label='Prior')
                
                ax.legend(loc='upper right', fontsize=7, frameon=True, facecolor='white')
                
            # Enhancements for Scatter Plots: Add Correlation Coefficients
            else:
                r = np.corrcoef(post_params[j], post_params[i])[0, 1]
                ax.annotate(f"$r = {r:.2f}$", xy=(0.1, 0.9), xycoords='axes fraction', 
                            ha='left', va='top', fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="navy", alpha=0.7))

    g.fig.suptitle(f"Regression Adjusted ABC: Approximate Posteriors\n(Top {top_pct:g}% / Distance Threshold $\epsilon \leq {threshold:.3f}$)", 
                   fontsize=16, fontweight='bold')
    g.fig.subplots_adjust(top=0.88, hspace=0.3, wspace=0.3) # Increased spacing
    
    plt.savefig("./diagrams/posterior_pairplot_regression.png", dpi=300, bbox_inches='tight')
    plt.show()

    # =================================================================
    # FIGURE 2: ABC Diagnostics (Distance vs. Parameters)
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    # Subset 20,000 random prior samples for the "cloud" so the plot isn't sluggish
    num_cloud = min(20000, len(sim_betas))
    idx_subset = np.random.choice(len(sim_betas), num_cloud, replace=False)
    
    # Cap the Y-axis at the 95th percentile to focus on the 'funnel' area
    max_dist_plot = np.percentile(distances, 95)
    
    for i, ax in enumerate(axes):
        # 1. Plot the "Cloud" of all prior simulations (Rejected points)
        ax.scatter(all_params[i][idx_subset], distances[idx_subset], 
                   c='gray', s=4, alpha=0.1, label='Rejected Samples')
        
        # 2. Highlight the "Accepted" points in red (using ADJUSTED points so you can see the shift)
        ax.scatter(post_params[i], distances[accepted_mask], 
                   c='crimson', s=12, alpha=0.6, edgecolor='white', linewidth=0.2, 
                   label=f'Accepted ($d \leq \epsilon$)')
        
        # 3. Draw the threshold line
        ax.axhline(threshold, color='black', linestyle='--', linewidth=1.5)
        
        ax.set_ylim(0, max_dist_plot)
        ax.set_xlim(prior_bounds[i])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        
        ax.set_xlabel(param_labels[i], fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            ax.set_ylabel("Mahalanobis Distance ($D_M$)", fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9, frameon=True)

    fig.suptitle(f"ABC Diagnostics: Parameter Convergence (Tolerance: {top_pct:g}%)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./diagrams/abc_diagnostics_regression.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()