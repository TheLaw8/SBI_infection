"""
Sequential Monte Carlo Approximate Bayesian Computation (SMC-ABC)

This program implements the Adaptive SMC-ABC algorithm (Beaumont et al. 2009). It:
1. Loads observed data and calculates the target statistics.
2. Extracts the final target epsilon from the reference table.
3. Initializes a Population of N=1000 particles at a very relaxed initial tolerance.
4. Iteratively shrinks the tolerance, sampling and perturbing particles from the 
   previous generation until the entire population reaches the final target epsilon.
5. Computes weights to correct for the proposal bias.
6. Plots the final approximate posterior.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import multivariate_normal

# Import your simulator
from simulator import simulate

# =====================================================================
# Hyperparameters
# =====================================================================
REFERENCE_TABLE_PATH = "full_reference_table.npz"
QUANTILES_TOLERANCE = 0.001  # Our final goal (Top 0.1% of 1M = 1000 particles)
INITIAL_TOLERANCE = 0.05     # Start wide (Top 5%) so the population can evolve
NUM_PARTICLES = 1000         # How many particles in our SMC population


def load_observed_data():
    """Reads the observed data CSVs and pivots them into 2D NumPy matrices."""
    print("Loading observed data...")
    df_inf = pd.read_csv("data/infected_timeseries.csv")
    df_rew = pd.read_csv("data/rewiring_timeseries.csv")
    df_deg = pd.read_csv("data/final_degree_histograms.csv")

    obs_inf = df_inf.pivot(index="replicate_id", columns="time", values="infected_fraction").values
    obs_rew = df_rew.pivot(index="replicate_id", columns="time", values="rewire_count").values
    obs_deg = df_deg.pivot(index="replicate_id", columns="degree", values="count").values
    
    return obs_inf, obs_rew, obs_deg


def compute_summary_statistics(infected, rewires, degrees):
    """Computes the optimized 4-dimensional vector of summary statistics.
       See other python files for a more detailed description of these summary stats. """
    max_inf = np.max(infected, axis=1)
    sum_inf = np.sum(infected, axis=1)
    max_rew = np.max(rewires, axis=1)
    
    n_nodes = 200
    bins = np.arange(31)
    total_k = np.sum(degrees * bins, axis=1)
    mean_k = total_k / n_nodes
    
    bin_diffs = np.abs(bins[:, None] - bins[None, :])
    count_prod_diff = np.sum(np.dot(degrees, bin_diffs) * degrees, axis=1)
    gini_deg = count_prod_diff / (2 * (n_nodes**2) * mean_k + 1e-6)

    return np.column_stack([max_inf, sum_inf, max_rew, gini_deg])


def get_weighted_covariance(particles, weights):
    """Calculates the weighted covariance matrix of a population."""
    mean = np.average(particles, axis=0, weights=weights)
    centered = particles - mean
    return np.cov(centered, rowvar=False, aweights=weights)


def main():
    # -----------------------------------------------------------------
    # 1. Prepare Observed Target & Baseline Metrics
    # -----------------------------------------------------------------
    obs_inf, obs_rew, obs_deg = load_observed_data()
    obs_stats_all = compute_summary_statistics(obs_inf, obs_rew, obs_deg)
    target_stat = np.mean(obs_stats_all, axis=0)
    
    print(f"Loading reference table to calculate baseline Mahalanobis metrics...")
    data = np.load(REFERENCE_TABLE_PATH)
    sim_stats = compute_summary_statistics(data["infected"], data["rewires"], data["degrees"])
    
    mahala_cov = np.cov(sim_stats, rowvar=False)
    inv_mahala_cov = np.linalg.inv(mahala_cov)
    
    diffs = sim_stats - target_stat 
    all_distances = np.sqrt(np.sum(np.dot(diffs, inv_mahala_cov) * diffs, axis=1))
    
    # This is the exact finish line (value of epsilon) we are trying to cross (0.1%)
    # We've seen this result, epsilon=0.323, in our previous algorithms :)
    FINAL_EPSILON = np.quantile(all_distances, QUANTILES_TOLERANCE)
    print(f"Target Final \u03b5 Threshold set to: {FINAL_EPSILON:.4f}\n")
    
    # -----------------------------------------------------------------
    # 2. SMC-ABC Generation 0 (Initialization)
    # -----------------------------------------------------------------
    print(f"--- Starting SMC-ABC Generation 0 (Top {INITIAL_TOLERANCE*100}% Prior) ---")
    prior_bounds = [(0.05, 0.50), (0.02, 0.20), (0.0, 0.8)]
    rng = np.random.default_rng(42)
    
    # We initialize the population using a much wider, looser tolerance (e.g., 5%)
    # This gives the SMC algorithm room to actually evolve and shrink the threshold.
    init_threshold = np.quantile(all_distances, INITIAL_TOLERANCE)
    valid_init_mask = all_distances <= init_threshold
    valid_init_indices = np.where(valid_init_mask)[0]
    
    # Randomly pick 1000 particles from this wider pool
    best_indices_gen0 = rng.choice(valid_init_indices, size=NUM_PARTICLES, replace=False)
    
    current_particles = np.column_stack([
        data["betas"][best_indices_gen0], 
        data["gammas"][best_indices_gen0], 
        data["rhos"][best_indices_gen0]
    ])
    current_distances = all_distances[best_indices_gen0]
    current_weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES  # Equal weights initially
    
    del data # Free memory
    
    # Set the tolerance for the NEXT generation (median of current pop)
    eps_t = np.median(current_distances)
    t = 1
    
    # -----------------------------------------------------------------
    # 3. SMC-ABC Main Loop (Evolving the Populations)
    # -----------------------------------------------------------------
    while True:
        # Cap the tolerance drop so we don't accidentally skip past the final finish line
        if eps_t < FINAL_EPSILON:
            eps_t = FINAL_EPSILON
            is_final_step = True
        else:
            is_final_step = False
            
        print(f"\n--- Generation {t} | Target \u03b5 = {eps_t:.4f} ---")
        
        # Calculate the proposal covariance from the previous population.
        # Beaumont (2009) recommends scaling the empirical covariance by 2 to ensure good mixing.
        prev_cov = get_weighted_covariance(current_particles, current_weights)
        proposal_cov = 2.0 * prev_cov
        
        new_particles = np.zeros((NUM_PARTICLES, 3))
        new_distances = np.zeros(NUM_PARTICLES)
        new_weights = np.zeros(NUM_PARTICLES)
        
        accepted_count = 0
        total_simulations_this_step = 0
        
        # Keep generating until we have N=1000 accepted particles
        while accepted_count < NUM_PARTICLES:
            # 1. Sample a "parent" particle based on the previous generation's weights
            parent_idx = rng.choice(NUM_PARTICLES, p=current_weights)
            parent_theta = current_particles[parent_idx]
            
            # 2. Perturb it via a Gaussian Random Walk
            theta_star = rng.multivariate_normal(parent_theta, proposal_cov)
            
            # 3. Check Prior Bounds
            out_of_bounds = False
            for p_idx in range(3):
                if theta_star[p_idx] < prior_bounds[p_idx][0] or theta_star[p_idx] > prior_bounds[p_idx][1]:
                    out_of_bounds = True
                    break
            if out_of_bounds:
                continue
                
            # 4. Simulate and Calculate Distance
            total_simulations_this_step += 1
            inf, rew, deg = simulate(beta=theta_star[0], gamma=theta_star[1], rho=theta_star[2], rng=rng)
            
            stat_star = compute_summary_statistics(inf.reshape(1, -1), rew.reshape(1, -1), deg.reshape(1, -1))[0]
            diff_star = stat_star - target_stat
            dist_star = np.sqrt(np.dot(diff_star, np.dot(inv_mahala_cov, diff_star)))
            
            # 5. Accept if distance is strictly less than current tolerance
            if dist_star <= eps_t:
                new_particles[accepted_count] = theta_star
                new_distances[accepted_count] = dist_star
                
                # Calculate weight: 1 / sum( W_prev * NormalPDF(theta_star | theta_prev, cov) )
                # We exploit the symmetry of the Gaussian ( N(A|B) == N(B|A) ) so SciPy vectorises it properly
                kernel_probs = multivariate_normal.pdf(current_particles, mean=theta_star, cov=proposal_cov)
                denominator = np.sum(current_weights * kernel_probs)
                new_weights[accepted_count] = 1.0 / denominator
                
                accepted_count += 1
                
                if accepted_count % 250 == 0:
                    print(f"  Accepted {accepted_count}/{NUM_PARTICLES} ...")
        
        # Normalize the weights so they sum to 1
        new_weights = new_weights / np.sum(new_weights)
        
        # Update state for the next generation
        current_particles = new_particles
        current_distances = new_distances
        current_weights = new_weights
        
        acceptance_rate = (NUM_PARTICLES / total_simulations_this_step) * 100
        print(f"Generation {t} Complete! Acceptance Rate: {acceptance_rate:.2f}%")
        
        if is_final_step:
            break
            
        # Drop the tolerance for the next round (Median of the current accepted pop)
        eps_t = np.median(current_distances)
        t += 1

    print("\nSMC-ABC Successfully Reached Final Tolerance!")

    # -----------------------------------------------------------------
    # 4. Plotting the Approximate Posterior (Weighted)
    # -----------------------------------------------------------------
    print("Plotting SMC-ABC Posterior Distributions...")
    os.makedirs("./diagrams", exist_ok=True)
    param_labels =[r"$\beta$ (Infection)", r"$\gamma$ (Recovery)", r"$\rho$ (Rewiring)"]
    
    # We must resample the final particles according to their SMC weights to plot them correctly
    resampled_indices = rng.choice(NUM_PARTICLES, size=NUM_PARTICLES, p=current_weights, replace=True)
    final_posteriors = current_particles[resampled_indices]
    
    df_posterior = pd.DataFrame({
        param_labels[0]: final_posteriors[:, 0],
        param_labels[1]: final_posteriors[:, 1],
        param_labels[2]: final_posteriors[:, 2]
    })
    
    post_params = [final_posteriors[:, 0], final_posteriors[:, 1], final_posteriors[:, 2]]
    post_means = [np.mean(p) for p in post_params]
    post_cis = [(np.percentile(p, 2.5), np.percentile(p, 97.5)) for p in post_params]

    # =================================================================
    # FIGURE 1: PairPlot
    # =================================================================
    sns.set_theme(style="white") 
    g = sns.PairGrid(df_posterior, corner=True, diag_sharey=False)
    
    g.map_diag(sns.histplot, kde=True, color="mediumpurple", bins=30, edgecolor='white', alpha=0.7)
    g.map_lower(sns.kdeplot, fill=True, cmap="Purples", alpha=0.7, levels=6)
    g.map_lower(sns.scatterplot, s=12, color=".2", alpha=0.15, linewidth=0)
    
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
                mean_val = post_means[i]
                ax.axvline(mean_val, color='crimson', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
                ci_low, ci_high = post_cis[i]
                ax.axvline(ci_low, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label=f'95% CI: ({ci_low:.3f}, {ci_high:.3f})')
                ax.axvline(ci_high, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
                
                bound_low, bound_high = prior_bounds[i]
                ax.axvspan(bound_low, bound_high, color='gray', alpha=0.08, label='Prior')
                ax.legend(loc='upper right', fontsize=7, frameon=True, facecolor='white')
                
            else:
                ax.set_ylabel(param_labels[i], fontsize=11, fontweight='bold')
                r = np.corrcoef(post_params[j], post_params[i])[0, 1]
                ax.annotate(f"$r = {r:.2f}$", xy=(0.1, 0.9), xycoords='axes fraction', 
                            ha='left', va='top', fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="indigo", alpha=0.7))

    g.fig.suptitle(f"SMC-ABC: Approximate Posteriors\n(Pop Size: {NUM_PARTICLES} / Distance Threshold $\epsilon \leq {FINAL_EPSILON:.3f}$)", 
                   fontsize=16, fontweight='bold')
    g.fig.subplots_adjust(top=0.88, hspace=0.3, wspace=0.3) 
    
    plt.savefig("./diagrams/posterior_pairplot_smc.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()