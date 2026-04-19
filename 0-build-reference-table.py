"""
Prior Simulation and Reference Table Generation for ABC.

What does this script do?
-------------------------
To perform Simulation-Based Inference (ABC), we need to compare our real, observed 
data against data simulated from random parameters. Simulating 1,000,000 times 
takes time. Instead of simulating "on-the-fly" during the ABC step, this script 
pre-computes 1,000,000 simulations upfront and saves them into a massive "Reference Table". 

Why use chunks and multiprocessing?
-----------------------------------
1. Multiprocessing runs simulations on all your CPU cores at the same time, making 
   it more efficient
2. Chunking (saving every 10,000 simulations to a temporary file) ensures that if 
   your computer crashes or goes to sleep, you don't lose your progress.
3. At the very end, this script cleanly merges all chunks into one single file 
   (`full_reference_table.npz`). 

Data Format for Summary Statistics:
-----------------------------------
The final file will contain 2D NumPy arrays (matrices). 
For example, the `infected` array will have 1,000,000 rows (simulations) and 
201 columns (time steps). This grid format makes computing summary statistics 
later much faster using vectorized NumPy functions.
"""

import os
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

# We import the simulator function you provided
from simulator import simulate

# =====================================================================
# Settings & Hyperparameters
# =====================================================================
TOTAL_SAMPLES = 1_000_000 # Total number of simulations to run
CHUNK_SIZE = 10_000       # How many simulations to do before saving a safe backup
TEMP_DIR = "temp_chunks"  # Folder to hold the temporary chunk files
FINAL_FILE = "full_reference_table.npz" # The single file we want at the end
BASE_SEED = 42            # Base random seed to ensure exact reproducibility


def sample_priors(num_samples, rng):
    """
    Sample random parameters (beta, gamma, rho) from their prior distributions.
    
    Parameters
    ----------
    num_samples : int
        The number of parameter sets to generate.
    rng : numpy.random.Generator
        The random number generator to use.
        
    Returns
    -------
    betas, gammas, rhos : np.ndarray (1D)
        Arrays of size (num_samples,) containing the random parameters.
    """
    betas = rng.uniform(0.05, 0.50, size=num_samples)
    gammas = rng.uniform(0.02, 0.20, size=num_samples)
    rhos = rng.uniform(0.0, 0.8, size=num_samples)
    
    return betas, gammas, rhos


def process_chunk(args):
    """
    Worker function that simulates one "chunk" (e.g., 10,000) of epidemics.
    
    This function is designed to be sent to a separate CPU core. It samples 
    its own parameters, runs the simulations, and saves the results to disk.
    
    Parameters
    ----------
    args : tuple
        Contains (chunk_id, num_samples, temp_dir, base_seed).
        We package these in a tuple so it plays nicely with mp.Pool().
        
    Returns
    -------
    bool
        True when finished.
    """
    # Unpack the arguments
    chunk_id, num_samples, temp_dir, base_seed = args
    
    # This is the file name where this specific chunk will be safely saved
    out_path = os.path.join(temp_dir, f"chunk_{chunk_id:04d}.npz")
    
    # Check if this chunk is already done (useful if resuming a stopped script)
    if os.path.exists(out_path):
        return True

    # 1. Setup a unique but reproducible random generator for this specific chunk
    chunk_rng = np.random.default_rng(base_seed + chunk_id)
    
    # 2. Draw the random parameters for this chunk
    betas, gammas, rhos = sample_priors(num_samples, chunk_rng)
    
    # 3. Create empty lists to hold the simulation results
    infected_list =[]
    rewires_list = []
    degrees_list =[]
    
    # 4. Run the simulations!
    for i in range(num_samples):
        inf, rew, deg = simulate(
            beta=betas[i], 
            gamma=gammas[i], 
            rho=rhos[i], 
            rng=chunk_rng
        )
        infected_list.append(inf)
        rewires_list.append(rew)
        degrees_list.append(deg)
        
    # 5. Save the input parameters AND the output data together in one file
    # Converting lists to NumPy arrays here makes them neat grids.
    np.savez_compressed(
        out_path,
        betas=betas,
        gammas=gammas,
        rhos=rhos,
        infected=np.array(infected_list, dtype=np.float32), # Shape: (10000, 201)
        rewires=np.array(rewires_list, dtype=np.int32),     # Shape: (10000, 201)
        degrees=np.array(degrees_list, dtype=np.int32)      # Shape: (10000, 31)
    )
    
    return True


def combine_chunks(temp_dir, num_chunks, final_file):
    """
    Reads all the small chunk files and stacks them into one massive final file.
    
    This ensures that when we move on to ABC, we only have to load a single 
    file, and all our data is in continuous 2D grids.
    """
    print(f"\nMerging all {num_chunks} chunks into a single file: {final_file}")
    
    all_betas, all_gammas, all_rhos = [], [], []
    all_infected, all_rewires, all_degrees = [], [], []
    
    # Loop through chunks with a progress bar
    for chunk_id in tqdm(range(num_chunks), desc="Merging Chunks"):
        out_path = os.path.join(temp_dir, f"chunk_{chunk_id:04d}.npz")
        data = np.load(out_path)
        
        # Append data blocks to lists
        all_betas.append(data["betas"])
        all_gammas.append(data["gammas"])
        all_rhos.append(data["rhos"])
        all_infected.append(data["infected"])
        all_rewires.append(data["rewires"])
        all_degrees.append(data["degrees"])
        
    # Stack everything vertically (np.concatenate) and save the final master file
    np.savez_compressed(
        final_file,
        betas=np.concatenate(all_betas),
        gammas=np.concatenate(all_gammas),
        rhos=np.concatenate(all_rhos),
        infected=np.concatenate(all_infected),
        rewires=np.concatenate(all_rewires),
        degrees=np.concatenate(all_degrees)
    )
    print("\nMerge complete! You can now safely delete the temporary folder if you wish.")


if __name__ == "__main__":
    # 1. Create a temporary folder if it doesn't exist yet
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 2. Calculate how many chunks we need to process
    num_chunks = int(np.ceil(TOTAL_SAMPLES / CHUNK_SIZE))
    
    # 3. Create the task instructions for each chunk
    tasks =[]
    samples_left = TOTAL_SAMPLES
    for chunk_id in range(num_chunks):
        samples_in_chunk = min(CHUNK_SIZE, samples_left)
        tasks.append((chunk_id, samples_in_chunk, TEMP_DIR, BASE_SEED))
        samples_left -= samples_in_chunk

    # 4. Determine optimal number of CPU workers
    # Use available cores minus 1 so your computer doesn't completely freeze up
    num_cores = max(1, mp.cpu_count() - 1)
    
    print(f"Starting simulation of {TOTAL_SAMPLES:,} samples...")
    print(f"Divided into {num_chunks} chunks of ~{CHUNK_SIZE:,} samples each.")
    print(f"Using {num_cores} CPU cores for multiprocessing.\n")
    
    # 5. Execute Multiprocessing Pool with a Progress Bar!
    # imap_unordered allows us to update the progress bar as soon as any chunk finishes
    with mp.Pool(processes=num_cores) as pool:
        # Wrap the multiprocessing execution in tqdm for a beautiful progress bar
        list(tqdm(pool.imap_unordered(process_chunk, tasks), total=num_chunks, desc="Simulating Chunks"))
            
    print("\nAll simulations complete!")
    
    # 6. Combine all the small chunk files into one massive reference table
    combine_chunks(TEMP_DIR, num_chunks, FINAL_FILE)