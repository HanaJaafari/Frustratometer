import random
import numpy as np
import frustratometer
import pandas as pd  # Import pandas for data manipulation
import numba

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def index_to_sequence(seq_index):
    """Converts sequence index array back to sequence string."""
    return ''.join([_AA[index] for index in seq_index])

def sequence_to_index(sequence):
    """Converts sequence string to sequence index array."""
    return np.array([_AA.find(aa) for aa in sequence])

#@numba.jit
def sequence_swap(seq_index, potts_model,mask):
    seq_index = seq_index.copy()
    res1, res2 = random.sample(range(len(seq_index)), 2)
    
    het_difference = 0
    energy_difference = compute_swap_energy(seq_index, potts_model, mask, res1, res2)

    seq_index[res1], seq_index[res2] = seq_index[res2], seq_index[res1]

    return seq_index, het_difference, energy_difference

#@numba.jit
def compute_swap_energy(seq_index, potts_model, mask, pos1, pos2):
    aa2 , aa1 = seq_index[pos1],seq_index[pos2]
    
    #Compute fields
    energy_difference = 0
    energy_difference -= (potts_model['h'][pos1, aa1] - potts_model['h'][pos1, seq_index[pos1]])  # h correction aa1
    energy_difference -= (potts_model['h'][pos2, aa2] - potts_model['h'][pos2, seq_index[pos2]])  # h correction aa2
    
    #Compute couplings
    j_correction = 0
    for pos, aa in enumerate(seq_index):
        # J correction interactions with other aminoacids
        reduced_j = potts_model['J'][pos, :, aa, :].astype(np.float32)
        j_correction += reduced_j[pos1, seq_index[pos1]] * mask[pos, pos1]
        j_correction -= reduced_j[pos1, aa1] * mask[pos, pos1]
        j_correction += reduced_j[pos2, seq_index[pos2]] * mask[pos, pos2]
        j_correction -= reduced_j[pos2, aa2] * mask[pos, pos2]
    
    # J correction, interaction with self aminoacids
    j_correction -= potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
    j_correction += potts_model['J'][pos1, pos2, aa1, seq_index[pos2]] * mask[pos1, pos2]  # Added mistakenly
    j_correction += potts_model['J'][pos1, pos2, seq_index[pos1], aa2] * mask[pos1, pos2]  # Added mistakenly
    j_correction -= potts_model['J'][pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
    energy_difference += j_correction
    return energy_difference

#@numba.jit
def sequence_mutation(seq_index, potts_model,mask):
    seq_index = seq_index.copy()
    res = random.randint(0, len(seq_index) - 1)
    aa_new = random.choice(range(1, 21))

    aa_old_count = np.sum(seq_index == seq_index[res])
    aa_new_count = np.sum(seq_index == aa_new)
    
    het_difference = np.log(aa_old_count/ (aa_new_count+1))
    energy_difference = compute_mutation_energy(seq_index, potts_model, mask, res, aa_new)

    seq_index[res] = aa_new
    
    return seq_index, het_difference, energy_difference

#@numba.jit
def compute_mutation_energy(seq_index, potts_model, mask, pos, aa_new):
    aa_old=seq_index[pos]
    energy_difference = -potts_model['h'][pos,aa_new] + potts_model['h'][pos,aa_old]

    reduced_j = potts_model['J'][range(len(seq_index)), :, seq_index, :]
    j_correction = reduced_j[:, pos, aa_old] * mask[pos]
    j_correction -= reduced_j[:, pos, aa_new] * mask[pos]
    
    # J correction, interaction with self aminoacids
    energy_difference += j_correction.sum(axis=0)
    return energy_difference

#@numba.jit
def native_energy(seq_index: np.array,
                  potts_model: dict,
                  mask: np.array) -> float:
    seq_len = len(seq_index)

    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)

    h = -potts_model['h'][range(seq_len), seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * mask        

    energy = h.sum() + j_prime.sum() / 2
    return energy

def heterogeneity(seq_index):
    N = len(seq_index)
    _, counts = np.unique(seq_index, return_counts=True)
    denominator = np.prod(np.array([np.math.factorial(count) for count in counts]))
    het = np.math.factorial(N) / denominator
    return np.log(het)

def stirling_log(n):
    if n < 40:
        return np.log(np.math.factorial(n))
    else:
        return n * np.log(n / np.e) + 0.5 * np.log(2 * np.pi * n) + 1.0 / (12 * n)

def heterogeneity_approximation(seq_index):
    """
    Uses Stirling's approximation to calculate the heterogeneity of a sequence
    """
    N = len(seq_index)
    _, counts = np.unique(seq_index, return_counts=True)

    
    log_n_factorial = stirling_log(N)
    log_denominator = sum([stirling_log(count) for count in counts])
    het = log_n_factorial - log_denominator
    return het

#@numba.njit
def montecarlo_steps(temperature, potts_model, mask, seq_index, Ep=100, n_steps = 1000, kb = 0.001987) -> np.array:
    for _ in range(n_steps):
        new_sequence, het_difference, energy_difference = sequence_swap(seq_index, potts_model, mask) if random.random() > 0.5 else sequence_mutation(seq_index, potts_model, mask)
        exponent=(-energy_difference + Ep * het_difference) / (kb * temperature)
        acceptance_probability = np.exp(min(0, exponent))
        if random.random() < acceptance_probability:
            seq_index = new_sequence
    return seq_index

def annealing():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")
    
    simulation_data = []
    for temp in range(800, 1, -1):
        seq_index= montecarlo_steps(temp, model.potts_model, model.mask, seq_index, Ep=10, n_steps=1000)
        energy = native_energy(seq_index, model.potts_model, model.mask)
        het = heterogeneity_approximation(seq_index)
        simulation_data.append({'Temperature': temp, 'Sequence': index_to_sequence(seq_index), 'Energy': energy, 'Heterogeneity': het})
        print(temp, index_to_sequence(seq_index), energy, het)
    simulation_df = pd.DataFrame(simulation_data)
    simulation_df.to_csv("mcso_simulation_results.csv", index=False)





def benchmark_montecarlo_steps(n_repeats=100, n_steps=1000):
    import time
    # Initialize the model for 1r69
    native_pdb = "tests/data/1r69.pdb"  # Ensure this path is correct
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    
    seq_len = len(model.sequence)
    times = []

    for _ in range(n_repeats):  # Run benchmark 10 times
        # Generate a new random sequence for each run
        seq_index = np.random.randint(1, 21, size=seq_len)
        start_time = time.time()
        
        montecarlo_steps(temperature=500, potts_model=model.potts_model, mask=model.mask, seq_index=seq_index, Ep=100, n_steps=n_steps)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    average_time = sum(times) / len(times) / n_steps * 1000
    average_time_per_step_s = average_time / 1000
    steps_per_hour = 3600 / average_time_per_step_s
    hours_needed = 1E10 / steps_per_hour / 8  # 8 processes in parallel

    print(f"Number of hours needed to explore 10^10 sequences with 8 process in parallel: {hours_needed:.2e} hours")
    print(f"Number of sequences explored per hour: {steps_per_hour:.2e}")
    print(f"Average execution time per step: {average_time:.5f} miliseconds")

if __name__ == '__main__':
    benchmark_montecarlo_steps()

