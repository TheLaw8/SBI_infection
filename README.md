# SBI for Epidemic Parameter Inference on Adaptive Networks

Code and data for the simulation-based inference (SBI) class assignment.
Given a stochastic SIR epidemic model on an adaptive network, infer the unknown parameters $(\beta, \gamma, \rho)$ from observed data using SBI methods.

Assignment by Lawrence Hider 

## READ THIS: How to run the code

First, you must run `0-build-reference-table.py` to build a reference table so we won't need to re-simulate every single time we run a new python program. This file builds the reference table in chunks, stored in a newly created folder called `temp_chunks`. These chunks will then be combined into a full reference table (used in subsequent programs) called `full_reference_table.npz`. 

Then, simply run all of the python programs in order, starting from `1-BasicABC-Euclidean-Original.py` to `6-SMC-ABC.py`. All diagrams will be saved into the `diagrams` folder. Note that I have already pre-generated these diagrams and they already exist within this folder. 

Next, note that the `papers` folder contains the papers I used when researching the advanced methodlogies. These papers have also been annotated with some of my own notes. 

Finally, the `data` folder contains data from the original observations, and `simulator.py` is used to run the actual simulation of the model.


## AI Disclaimer 
AI (Gemini) was used in this repository to help me draft and debug the code. I am responsible for the content and quality of the submitted work. The ideas behind the code are all my own and based on my own research, and I fully understand all the code submitted.

## The model

A population of 200 agents interact on an undirected contact network.
Each agent is **Susceptible (S)**, **Infected (I)**, or **Recovered (R)**.
The initial network is an Erdos-Renyi graph $G(N, p)$ with $p = 0.05$, giving an expected degree of about 10.
At time 0, five agents chosen uniformly at random are infected.

Three parameters govern the dynamics:

| Parameter | Meaning | Prior |
|-----------|---------|-------|
| $\beta$ | Infection probability per S--I edge per step | Uniform(0.05, 0.50) |
| $\gamma$ | Recovery probability per infected agent per step | Uniform(0.02, 0.20) |
| $\rho$ | Rewiring probability per S--I edge per step | Uniform(0.0, 0.8) |

At each of the 200 time steps, three phases are applied synchronously:

1. **Infection.** Each susceptible neighbor of an infected agent becomes infected with probability $\beta$.
2. **Recovery.** Each infected agent recovers with probability $\gamma$.
3. **Rewiring.** For each S-I edge, with probability $\rho$ the susceptible agent breaks the link and connects to a random non-neighbor. This models behavioral avoidance of infected contacts.

## Repository contents

```
simulator.py                      # Python implementation of the model
data/
  infected_timeseries.csv         # Fraction infected over time (40 replicates)
  rewiring_timeseries.csv         # Rewiring counts over time (40 replicates)
  final_degree_histograms.csv     # Degree distribution at t=200 (40 replicates)
```

## Simulator usage

```python
import numpy as np
from simulator import simulate

# Run one replicate with specific parameters
rng = np.random.default_rng(42)
infected, rewires, degrees = simulate(beta=0.3, gamma=0.15, rho=0.7, rng=rng)
```

See `simulator.py` for full parameter documentation.

## Observed data

The data files contain 40 independent realizations, all generated with the **same** unknown $(\beta, \gamma, \rho)$.
The contact network is never observed.

| File | Columns |
|------|---------|
| `infected_timeseries.csv` | `replicate_id`, `time`, `infected_fraction` |
| `rewiring_timeseries.csv` | `replicate_id`, `time`, `rewire_count` |
| `final_degree_histograms.csv` | `replicate_id`, `degree` (0-30, clipped), `count` |

## Requirements

- Python 3.8+
- NumPy
