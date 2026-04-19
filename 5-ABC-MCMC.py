"""
Markov Chain Monte Carlo Approximate Bayesian Computation (ABC-MCMC)

This script implements the Marjoram et al. (2003) ABC-MCMC algorithm. It:
1. Loads the observed data and computes target summary statistics.
2. Loads the reference table temporarily to establish the exact same distance 
   threshold (epsilon) and Mahalanobis covariance matrix as the Rejection ABC.
3. Initializes the Markov Chain at the best known parameter set.
4. Performs a Gaussian Random Walk, running the simulator on-the-fly.
5. Accepts/Rejects based on the epsilon threshold.
6. Plots the MCMC Trace diagnostics and the resulting approximate posterior.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Import the simulator
from simulator import simulate

# =====================================================================
# Hyperparameters
# =====================================================================
REFERENCE_TABLE_PATH = "full_reference_table.npz"
QUANTILES_TOLERANCE = 0.001  # To match the previous \epsilon threshold
NUM_MCMC_STEPS = 100_000      # Increased to 100,000 for better mixing


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
    # Contribution: This is the primary signal for beta. A highly infectious 
    # disease will produce a massive peak, while a weak disease will stay low.
    # Note: It is partially coupled with rho, as a high rho (lots of social 
    # distancing) can also suppress the peak.
    max_inf = np.max(infected, axis=1)
    
    # =================================================================
    # 2. TOTAL INFECTION BURDEN (Main target: Gamma)
    # =================================================================
    # Purpose: Calculates the area under the infection curve (total infected-days).
    # Contribution: This is an effective signal for gamma. Because gamma 
    # dictates the duration of an infection (1/gamma), it strictly controls 
    # how long the epidemic lingers. A low gamma means infections last longer, 
    # directly inflating this sum regardless of network topology.
    sum_inf = np.sum(infected, axis=1)
    
    # =================================================================
    # 3. PEAK REWIRING INTENSITY (Main target: Rho)
    # =================================================================
    # Purpose: Measures the maximum 'avoidance' response in a single time step.
    # Contribution: This heavily targets rho and decouples it from beta. 
    # While total rewiring might look similar for a fast/short epidemic and a 
    # slow/long epidemic, a high rho typically leads to a large spike 
    # in rewiring events which max_rew captures.
    max_rew = np.max(rewires, axis=1)
    
    # =================================================================
    # 4. DEGREE GINI COEFFICIENT (Main target: Rho / Structural Inequality)
    # =================================================================
    # Purpose: Measures how unequal the final degree distribution is.
    # Contribution: A standard Erdős-Rényi graph has low inequality (low Gini). 
    # As rho increases, susceptible nodes flee infected neighbors, creating 
    # isolated infected nodes (low degree) and highly connected safe hubs 
    # (high degree). The Gini coefficient captures this structural inequality.
    
    n_nodes = 200
    bins = np.arange(31)
    
    # Total edges and mean degree for each simulation
    total_k = np.sum(degrees * bins, axis=1)
    mean_k = total_k / n_nodes
    
    # Matrix of absolute differences between all pairs of degree bins: |i - j|
    bin_diffs = np.abs(bins[:, None] - bins[None, :])

    # Calculate the Gini numerator: sum_{i,j} (count_i * count_j * |bin_i - bin_j|)
    count_prod_diff = np.sum(np.dot(degrees, bin_diffs) * degrees, axis=1)
    
    # Final Gini formula: Numerator / (2 * N^2 * mean)
    gini_deg = count_prod_diff / (2 * (n_nodes**2) * mean_k + 1e-6)

    # Return summary statistics
    return np.column_stack([max_inf, sum_inf, max_rew, gini_deg])


def main():
    # -----------------------------------------------------------------
    # 1. Prepare Observed Target
    # -----------------------------------------------------------------
    obs_inf, obs_rew, obs_deg = load_observed_data()
    obs_stats_all = compute_summary_statistics(obs_inf, obs_rew, obs_deg)
    
    # Take the mean across the 40 replicates to get our TARGET statistic
    target_stat = np.mean(obs_stats_all, axis=0)
    
    # -----------------------------------------------------------------
    # 2. Establish Baseline metrics from the Reference Table
    # -----------------------------------------------------------------
    # We do this so the MCMC uses the EXACT same distance metric and threshold
    # as the Rejection ABC, ensuring a fair comparison for your report.
    print(f"Loading reference table to calculate baseline Mahalanobis metrics...")
    data = np.load(REFERENCE_TABLE_PATH)
    sim_stats = compute_summary_statistics(data["infected"], data["rewires"], data["degrees"])
    
    # Compute the covariance matrix of the simulated summary statistics
    cov_matrix = np.cov(sim_stats, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    # Calculate the difference between every simulation and the target
    diffs = sim_stats - target_stat 
    
    # Compute the Mahalanobis distance D = sqrt( diffs^T * inv_cov * diffs )
    distances = np.sqrt(np.sum(np.dot(diffs, inv_cov_matrix) * diffs, axis=1))
    
    # Find the distance threshold corresponding to the top 0.1%
    threshold = np.quantile(distances, QUANTILES_TOLERANCE)
    
    # Find the single best simulation to initialize the MCMC chain (zero burn-in!)
    best_idx = np.argmin(distances)
    theta_current = np.array([data["betas"][best_idx], data["gammas"][best_idx], data["rhos"][best_idx]])
    
    # Define the proposal covariance (how big of a step the random walk takes).
    # We use the covariance of the accepted Basic ABC parameters, but scaled down
    # to 1/5th (0.1) to force the MCMC to take much smaller steps and improve mixing.
    accepted_mask = distances <= threshold
    acc_params = np.column_stack([data["betas"][accepted_mask], data["gammas"][accepted_mask], data["rhos"][accepted_mask]])
    proposal_cov = np.cov(acc_params, rowvar=False) * 0.1
    
    del data # Free up memory
    print(f"Target \u03b5 Threshold set to: {threshold:.4f}")
    print(f"Chain starting at optimal \u03b8: Beta={theta_current[0]:.3f}, Gamma={theta_current[1]:.3f}, Rho={theta_current[2]:.3f}\n")

    # -----------------------------------------------------------------
    # 3. ABC-MCMC Execution
    # -----------------------------------------------------------------
    prior_bounds = [(0.05, 0.50), (0.02, 0.20), (0.0, 0.8)]
    chain = np.zeros((NUM_MCMC_STEPS, 3))
    chain_distances = np.zeros(NUM_MCMC_STEPS)
    accepted_moves = 0
    
    # Setup random generator
    rng = np.random.default_rng(42)
    current_dist = distances[best_idx]
    
    print(f"Starting ABC-MCMC for {NUM_MCMC_STEPS:,} steps. This requires on-the-fly simulation...")
    
    for i in range(NUM_MCMC_STEPS):
        # A. Propose a new state via Multivariate Gaussian Random Walk
        theta_prop = rng.multivariate_normal(theta_current, proposal_cov)
        
        # B. Check Prior Bounds. If outside, immediately reject and stay in place.
        out_of_bounds = False
        for p_idx in range(3):
            if theta_prop[p_idx] < prior_bounds[p_idx][0] or theta_prop[p_idx] > prior_bounds[p_idx][1]:
                out_of_bounds = True
                break
                
        if out_of_bounds:
            chain[i] = theta_current
            chain_distances[i] = current_dist
            continue
            
        # C. Run Simulator with proposed parameters
        inf, rew, deg = simulate(beta=theta_prop[0], gamma=theta_prop[1], rho=theta_prop[2], rng=rng)
        
        # D. Format output for the stats function (requires 2D arrays)
        inf_2d = inf.reshape(1, -1)
        rew_2d = rew.reshape(1, -1)
        deg_2d = deg.reshape(1, -1)
        
        # E. Calculate Stats and Distance
        stat_prop = compute_summary_statistics(inf_2d, rew_2d, deg_2d)[0]
        diff_prop = stat_prop - target_stat
        dist_prop = np.sqrt(np.dot(diff_prop, np.dot(inv_cov_matrix, diff_prop)))
        
        # F. Accept or Reject based on threshold
        if dist_prop <= threshold:
            theta_current = theta_prop
            current_dist = dist_prop
            accepted_moves += 1
            
        # Record the state of the chain
        chain[i] = theta_current
        chain_distances[i] = current_dist
        
        if (i + 1) % 5000 == 0:
            print(f"  Step {i+1:,}/{NUM_MCMC_STEPS:,} | Current Acceptance Rate: {(accepted_moves/(i+1))*100:.2f}%")

    print(f"\nMCMC Complete. Final Acceptance Rate (moves): {(accepted_moves/NUM_MCMC_STEPS)*100:.2f}%\n")
    
    # -----------------------------------------------------------------
    # 4. Plotting MCMC Diagnostics (Trace Plots)
    # -----------------------------------------------------------------
    print("Plotting MCMC Trace Diagnostics...")
    os.makedirs("./diagrams", exist_ok=True)
    param_labels =[r"$\beta$ (Infection)", r"$\gamma$ (Recovery)", r"$\rho$ (Rewiring)"]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for p_idx in range(3):
        # We plot the full chain here to visualize "stickiness"
        axes[p_idx].plot(chain[:, p_idx], color='steelblue', alpha=0.8, linewidth=0.5)
        axes[p_idx].set_ylabel(param_labels[p_idx], fontsize=12, fontweight='bold')
        axes[p_idx].grid(True, linestyle=':', alpha=0.6)
        # Visual guide for prior bounds
        bound_low, bound_high = prior_bounds[p_idx]
        axes[p_idx].axhspan(bound_low, bound_high, color='gray', alpha=0.1)
        axes[p_idx].set_ylim(prior_bounds[p_idx])
        
    axes[2].set_xlabel("MCMC Step (Iteration)", fontsize=12, fontweight='bold')
    # Using comma formatting for larger numbers in the title
    fig.suptitle(f"ABC-MCMC Trace Plots ({NUM_MCMC_STEPS:,} Steps)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("./diagrams/5a-traceplots-mcmc.png", dpi=300, bbox_inches='tight')
    plt.show()

    # -----------------------------------------------------------------
    # 5. Plotting the Approximate Posterior
    # -----------------------------------------------------------------
    print("Plotting MCMC Posterior Distributions...")
    
    df_posterior = pd.DataFrame({
        param_labels[0]: chain[:, 0],
        param_labels[1]: chain[:, 1],
        param_labels[2]: chain[:, 2]
    })
    
    post_params = [chain[:, 0], chain[:, 1], chain[:, 2]]
    post_means = [np.mean(p) for p in post_params]
    post_cis = [(np.percentile(p, 2.5), np.percentile(p, 97.5)) for p in post_params]

    sns.set_theme(style="white") 
    g = sns.PairGrid(df_posterior, corner=True, diag_sharey=False)
    
    g.map_diag(sns.histplot, kde=True, color="mediumseagreen", bins=30, edgecolor='white', alpha=0.7)
    g.map_lower(sns.kdeplot, fill=True, cmap="Greens", alpha=0.7, levels=6)
    
    # We thin the scatter plot so 100,000 points don't create a massive black blob
    thin_idx = np.arange(0, NUM_MCMC_STEPS, 10) 
    g.map_lower(lambda x, y, **kwargs: plt.scatter(x.iloc[thin_idx], y.iloc[thin_idx], s=10, color=".2", alpha=0.05, linewidth=0))
    
    for i in range(len(param_labels)):
        for j in range(i + 1):  
            ax = g.axes[i, j]
            if ax is None:
                continue
            
            ax.tick_params(labelbottom=True, labelleft=True, direction='out', length=4, width=1)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            
            ax.set_xlabel(param_labels[j], fontsize=11, fontweight='bold')
            if i == j:
                ax.set_ylabel("Density", fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel(param_labels[i], fontsize=11, fontweight='bold')
            
            ax.grid(True, linestyle=':', alpha=0.6)
            
            if i == j:
                mean_val = post_means[i]
                ax.axvline(mean_val, color='crimson', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
                ci_low, ci_high = post_cis[i]
                ax.axvline(ci_low, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label=f'95% CI: ({ci_low:.3f}, {ci_high:.3f})')
                ax.axvline(ci_high, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
                
                bound_low, bound_high = prior_bounds[i]
                ax.axvspan(bound_low, bound_high, color='gray', alpha=0.08, label='Prior')
                ax.legend(loc='upper right', fontsize=7, frameon=True, facecolor='white')
                
            else:
                r = np.corrcoef(post_params[j], post_params[i])[0, 1]
                ax.annotate(f"$r = {r:.2f}$", xy=(0.1, 0.9), xycoords='axes fraction', 
                            ha='left', va='top', fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="forestgreen", alpha=0.7))

    g.fig.suptitle(f"ABC-MCMC: Approximate Posteriors\n({NUM_MCMC_STEPS:,} Steps / Distance Threshold $\epsilon \leq {threshold:.3f}$)", 
                   fontsize=16, fontweight='bold')
    g.fig.subplots_adjust(top=0.88, hspace=0.3, wspace=0.3) 
    
    plt.savefig("./diagrams/5b-posterior-pairplot-mcmc.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()