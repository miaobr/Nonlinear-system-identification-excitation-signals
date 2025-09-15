#!/usr/bin/env python
# coding: utf-8

# In[2]:


#IDF_FID
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Parameters
N = 200  # total length of signal
n_sub = 20  # length of each subsequence
u_min = 0  # min input
u_max = 1  # max input
lambda_param = 0.7  # higher -> more transient

# Initialize signal
signal = np.array([])

# Iterative signal generation
while len(signal) < N:
    # Generate candidate subsequences
    n_candidates = 50
    candidates = np.random.rand(n_candidates, n_sub) * (u_max - u_min) + u_min

    if len(signal) == 0:
        # first iteration, pick random candidate
        idx = np.random.randint(n_candidates)
    else:
        # Compute distances from candidate subsequences to existing signal
        distances = np.zeros(n_candidates)
        for i in range(n_candidates):
            distances[i] = np.min(cdist(candidates[i, :].reshape(-1, 1), signal.reshape(-1, 1)))

        # Apply lambda: favor transient (higher lambda)
        scores = distances * (lambda_param + (1 - lambda_param))
        idx = np.argmax(scores)

    # Append chosen subsequence
    signal = np.concatenate([signal, candidates[idx, :]])

# Trim to exact length N
signal = signal[:N]

# Plot the excitation signal
plt.figure(figsize=(10, 4))
plt.plot(signal, marker='o')
plt.xlabel('Time step')
plt.ylabel('Excitation input')
plt.title('IDS-FID-like Excitation Signal')
plt.grid(True)
plt.show()


# In[4]:


#Reciding Horizon Control plus Space Filling Criterion
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def generate_rhc_excitation(N=200, H=5, levels=None, n_candidates=100):
    """
    Generate excitation signal using RHC + space-filling criterion.

    N : int
        Total signal length
    H : int
        Receding horizon (look-ahead window length)
    levels : list or None
        Predefined discrete input levels
    n_candidates : int
        Number of candidate sequences evaluated at each step
    """
    if levels is None:
        levels = np.linspace(0, 1, 5)  # default: 5 evenly spaced levels

    signal = []

    while len(signal) < N:
        # Generate candidate subsequences from allowed levels
        candidates = np.random.choice(levels, size=(n_candidates, H))

        if len(signal) == 0:
            # Start with a random candidate
            idx = np.random.randint(n_candidates)
        else:
            # Evaluate each candidate: maximize distance from existing signal
            distances = np.zeros(n_candidates)
            for i in range(n_candidates):
                candidate = candidates[i, :].reshape(-1, 1)
                existing = np.array(signal).reshape(-1, 1)
                # Minimum distance between new and existing points
                distances[i] = np.min(cdist(candidate, existing))

            # Pick candidate with best space-filling score
            idx = np.argmax(distances)

        # Append best subsequence
        signal.extend(candidates[idx, :].tolist())

    # Trim to exact length
    return np.array(signal[:N])


# Example usage
N = 200
signal = generate_rhc_excitation(N=N, H=5, levels=[0, 0.25, 0.5, 0.75, 1])

# Plot
plt.figure(figsize=(10, 4))
plt.step(range(N), signal, where='post')
plt.xlabel('Time step')
plt.ylabel('Excitation input')
plt.title('RHC + Space-Filling Excitation Signal')
plt.grid(True)
plt.show()


# In[5]:



#GOATS((Genetic Optimization of Amplitude and Time Steps)
import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------------
# Parameters
# ------------------------------
N_steps = 20  # number of steps in signal
pop_size = 30  # GA population size
generations = 50  # number of generations
amp_bounds = (-1, 1)  # amplitude limits
dur_bounds = (5, 30)  # duration (samples per step)
signal_length = 500  # total samples per signal


# ------------------------------
# Fitness function
# ------------------------------
def evaluate_signal(amps, durs):
    """Return a fitness score: flatter spectrum + good amplitude spread."""
    # Build signal from amplitudes + durations
    signal = []
    for a, d in zip(amps, durs):
        signal.extend([a] * d)
        if len(signal) >= signal_length:
            break
    signal = np.array(signal[:signal_length])

    # 1. Frequency coverage (use FFT flatness)
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum += 1e-6  # avoid log(0)
    flatness = np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)

    # 2. Amplitude variance (spread in input space)
    amp_spread = np.std(signal)

    # Weighted fitness
    return 0.7 * flatness + 0.3 * amp_spread


# ------------------------------
# GA functions
# ------------------------------
def random_individual():
    amps = [random.uniform(*amp_bounds) for _ in range(N_steps)]
    durs = [random.randint(*dur_bounds) for _ in range(N_steps)]
    return amps, durs


def mutate(amps, durs, rate=0.2):
    amps = amps[:]
    durs = durs[:]
    for i in range(N_steps):
        if random.random() < rate:
            amps[i] = random.uniform(*amp_bounds)
        if random.random() < rate:
            durs[i] = random.randint(*dur_bounds)
    return amps, durs


def crossover(parent1, parent2):
    cut = random.randint(1, N_steps - 1)
    child1 = (parent1[0][:cut] + parent2[0][cut:], parent1[1][:cut] + parent2[1][cut:])
    child2 = (parent2[0][:cut] + parent1[0][cut:], parent2[1][:cut] + parent1[1][cut:])
    return child1, child2


# ------------------------------
# GA main loop
# ------------------------------
population = [random_individual() for _ in range(pop_size)]

for gen in range(generations):
    # Evaluate
    scores = [evaluate_signal(*ind) for ind in population]

    # Select top half
    ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    survivors = [p for _, p in ranked[:pop_size // 2]]

    # Reproduce
    next_pop = survivors[:]
    while len(next_pop) < pop_size:
        p1, p2 = random.sample(survivors, 2)
        c1, c2 = crossover(p1, p2)
        next_pop.append(mutate(*c1))
        if len(next_pop) < pop_size:
            next_pop.append(mutate(*c2))
    population = next_pop

    if gen % 10 == 0:
        print(f"Gen {gen}, best fitness = {ranked[0][0]:.4f}")

# ------------------------------
# Plot best signal
# ------------------------------
best_amps, best_durs = ranked[0][1]
signal = []
for a, d in zip(best_amps, best_durs):
    signal.extend([a] * d)
    if len(signal) >= signal_length:
        break
signal = np.array(signal[:signal_length])

plt.figure(figsize=(10, 4))
plt.plot(signal, lw=1)
plt.title("GA-Optimized Step Signal (GOATS-style)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


# In[ ]:




