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

@numba.njit
def sequence_swap(seq_index, model_h, model_J, mask):
    seq_index = seq_index.copy()
    n=len(seq_index)
    res1 = np.random.randint(0,n-1)
    res2 = np.random.randint(0,n-2)
    res2 += (res2 >= res1)
    
    het_difference = 0
    energy_difference = compute_swap_energy(seq_index, model_h, model_J, mask, res1, res2)

    seq_index[res1], seq_index[res2] = seq_index[res2], seq_index[res1]

    return seq_index, het_difference, energy_difference

@numba.njit
def compute_swap_energy(seq_index, model_h, model_J, mask, pos1, pos2):
    aa2 , aa1 = seq_index[pos1],seq_index[pos2]
    
    #Compute fields
    energy_difference = 0
    energy_difference -= (model_h[pos1, aa1] - model_h[pos1, seq_index[pos1]])  # h correction aa1
    energy_difference -= (model_h[pos2, aa2] - model_h[pos2, seq_index[pos2]])  # h correction aa2
    
    #Compute couplings
    j_correction = 0.0
    for pos in range(len(seq_index)):
        aa = seq_index[pos]
        # Corrections for interactions with pos1 and pos2
        j_correction += model_J[pos, pos1, aa, seq_index[pos1]] * mask[pos, pos1]
        j_correction -= model_J[pos, pos1, aa, aa1] * mask[pos, pos1]
        j_correction += model_J[pos, pos2, aa, seq_index[pos2]] * mask[pos, pos2]
        j_correction -= model_J[pos, pos2, aa, aa2] * mask[pos, pos2]

    # J correction, interaction with self aminoacids
    j_correction -= model_J[pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
    j_correction += model_J[pos1, pos2, aa1, seq_index[pos2]] * mask[pos1, pos2]  # Added mistakenly
    j_correction += model_J[pos1, pos2, seq_index[pos1], aa2] * mask[pos1, pos2]  # Added mistakenly
    j_correction -= model_J[pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
    energy_difference += j_correction
    return energy_difference


@numba.njit
def sequence_mutation(seq_index, model_h, model_J, mask):
    seq_index = seq_index.copy()
    r = np.random.randint(0, len(seq_index)*20-1)  # Select a random index    
    res = r // 20
    aa_new = r % 20 + 1

    aa_old_count = np.sum(seq_index == seq_index[res])
    aa_new_count = np.sum(seq_index == aa_new)
    
    het_difference = np.log(aa_old_count/ (aa_new_count+1))
    energy_difference = compute_mutation_energy(seq_index, model_h, model_J, mask, res, aa_new)

    seq_index[res] = aa_new
    
    return seq_index, het_difference, energy_difference

@numba.njit
def compute_mutation_energy(seq_index: np.ndarray, model_h: np.ndarray, model_J: np.ndarray, mask: np.ndarray, pos: int, aa_new: int) -> float:
    aa_old=seq_index[pos]
    energy_difference = -model_h[pos,aa_new] + model_h[pos,aa_old]

    energy_difference = -model_h[pos, aa_new] + model_h[pos, aa_old]

    # Initialize j_correction to 0
    j_correction = 0.0

    # Manually iterate over the sequence indices
    for idx in range(len(seq_index)):
        aa_idx = seq_index[idx]  # The amino acid at the current position
        # Accumulate corrections for positions other than the mutated one
        j_correction += model_J[idx, pos, aa_idx, aa_old] * mask[idx, pos]
        j_correction -= model_J[idx, pos, aa_idx, aa_new] * mask[idx, pos]

    # For self-interaction, subtract the old interaction and add the new one
    j_correction -= model_J[pos, pos, aa_old, aa_old] * mask[pos, pos]
    j_correction += model_J[pos, pos, aa_new, aa_new] * mask[pos, pos]

    energy_difference += j_correction

    return energy_difference

@numba.njit
def model_energy(seq_index: np.array,
                  model_h: np.ndarray, model_J: np.ndarray,
                  mask: np.array) -> float:
    seq_len = len(seq_index)
    energy_h = 0.0
    energy_J = 0.0

    for i in range(seq_len):
        energy_h -= model_h[i, seq_index[i]]
    
    for i in range(seq_len):
        for j in range(seq_len):
            aa_i = seq_index[i]
            aa_j = seq_index[j]
            energy_J -= model_J[i, j, aa_i, aa_j] * mask[i, j]
    
    total_energy = energy_h + energy_J / 2
    return total_energy

def heterogeneity(seq_index):
    N = len(seq_index)
    _, counts = np.unique(seq_index, return_counts=True)
    denominator = np.prod(np.array([np.math.factorial(count) for count in counts]))
    het = np.math.factorial(N) / denominator
    return np.log(het)

log_factorial_table=np.log(np.array([np.math.factorial(i) for i in range(40)],dtype=np.float64))

@numba.njit
def stirling_log(n):
    if n < 40:
        return log_factorial_table[n]
    else:
        return n * np.log(n / np.e) + 0.5 * np.log(2 * np.pi * n) + 1.0 / (12 * n)

@numba.njit
def heterogeneity_approximation(seq_index):
    """
    Uses Stirling's approximation to calculate the heterogeneity of a sequence
    """
    N = len(seq_index)
    counts = np.zeros(21, dtype=np.int32)
    
    for val in seq_index:
        counts[val] += 1
        
    log_n_factorial = stirling_log(N)
    log_denominator = sum([stirling_log(count) for count in counts])
    het = log_n_factorial - log_denominator
    return het

@numba.njit
def montecarlo_steps(temperature, model_h, model_J, mask, seq_index, Ep=100, n_steps = 1000, kb = 0.001987) -> np.array:
    for _ in range(n_steps):
        new_sequence, het_difference, energy_difference = sequence_swap(seq_index, model_h, model_J, mask) if random.random() > 0.5 else sequence_mutation(seq_index, model_h, model_J, mask)
        exponent=(-energy_difference + Ep * het_difference) / (kb * temperature + 1E-10)
        acceptance_probability = np.exp(min(0, exponent)) 
        if random.random() < acceptance_probability:
            seq_index = new_sequence
    return seq_index

@numba.njit
def replica_exchanges(energies, seq_indices, temperatures, kb=0.001987):
    """
    Attempt to exchange configurations between pairs of replicas.
    """
    n_replicas = len(temperatures)
    for i in range(n_replicas - 1):
        energy1, energy2 = energies[i], energies[i + 1]
        temp1, temp2 = temperatures[i], temperatures[i + 1]
        delta = (1/temp2 - 1/temp1) * (energy2 - energy1)
            
        # Calculate exchange probability
        exponent = -delta / kb
        prob= np.exp(min(0, exponent)) 

        # Decide whether to exchange
        if np.random.rand() < prob:
            # Exchange sequences
            seq_indices[i], seq_indices[i + 1] = seq_indices[i + 1].copy(), seq_indices[i].copy()

@numba.njit(parallel=True)
def parallel_tempering_step(model_h, model_J, mask, seq_indices, temperatures, n_steps_per_cycle, Ep):
    n_replicas = len(temperatures)
    energies = np.zeros(n_replicas)
    heterogeneities = np.zeros(n_replicas)
    total_energies = np.zeros(n_replicas)
    for i in numba.prange(n_replicas):
        temp_seq_index = seq_indices[i]
        seq_indices[i] = montecarlo_steps(temperatures[i], model_h, model_J, mask, seq_index=temp_seq_index, Ep=Ep, n_steps=n_steps_per_cycle)
        energy = model_energy(seq_indices[i], model_h, model_J, mask)
        het = heterogeneity_approximation(seq_indices[i])
        # Compute energy for the new sequence
        total_energies[i] = energy - Ep * het # Placeholder for actual energy calculation
        energies[i] = energy
        heterogeneities[i] = het

    # Perform replica exchanges
    replica_exchanges(energies, seq_indices, temperatures)
    return seq_indices, energies, heterogeneities

def parallel_tempering(model_h, model_J, mask, seq_indices, temperatures, n_steps, n_steps_per_cycle, Ep):
    simulation_data=[]
    for s in range(n_steps//n_steps_per_cycle):
        seq_indices, energy, het = parallel_tempering_step(model_h, model_J, mask, seq_indices, temperatures, n_steps_per_cycle, Ep)
        #Save data every 100 exchanges
        if s%100==99:
            for i,temp in enumerate(temperatures):
                simulation_data.append({'Step':(s+1)*n_steps_per_cycle,'Temperature': temp, 'Sequence': index_to_sequence(seq_indices[i]), 'Energy': energy[i], 'Heterogeneity': het[i], 'Total Energy': energy[i] + Ep * het[i]})
            print(*simulation_data[-1].values())
        if s%10000==0:
            simulation_df = pd.DataFrame(simulation_data)
            simulation_df.to_csv("parallel_tempering_results.csv", index=False)
    simulation_df = pd.DataFrame(simulation_data)
    simulation_df.to_csv("parallel_tempering_results.csv", index=False)


def annealing(temp_max=500, temp_min=0, n_steps=1E8, Ep=10):
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")
    
    simulation_data = []
    n_steps_per_cycle=n_steps//(temp_max-temp_min)
    for temp in range(temp_max, temp_min, -1):
        seq_index= montecarlo_steps(temp, model.potts_model['h'], model.potts_model['J'], model.mask, seq_index, Ep=Ep, n_steps=n_steps_per_cycle)
        energy = model_energy(seq_index, model.potts_model, model.mask)
        het = heterogeneity_approximation(seq_index)
        simulation_data.append({'Temperature': temp, 'Sequence': index_to_sequence(seq_index), 'Energy': energy, 'Heterogeneity': het, 'Total Energy': energy + Ep * het})
        print(temp, index_to_sequence(seq_index), energy + Ep * het, energy, het)
    simulation_df = pd.DataFrame(simulation_data)
    simulation_df.to_csv("mcso_simulation_results.csv", index=False)

def benchmark_montecarlo_steps(n_repeats=100, n_steps=20000):
    import time
    # Initialize the model for 1r69
    native_pdb = "tests/data/1r69.pdb"  # Ensure this path is correct
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    
    seq_len = len(model.sequence)
    times = []

    #Adds one step for numba compilation time
    montecarlo_steps(temperature=500, model_h=model.potts_model['h'], model_J=model.potts_model['J'], mask=model.mask, seq_index=sequence_to_index(model.sequence), Ep=100, n_steps=1)

    for _ in range(n_repeats):  # Run benchmark 10 times
        # Generate a new random sequence for each run
        seq_index = np.random.randint(1, 21, size=seq_len)
        start_time = time.time()
        
        montecarlo_steps(temperature=500, model_h=model.potts_model['h'], model_J=model.potts_model['J'], mask=model.mask, seq_index=seq_index, Ep=100, n_steps=n_steps)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    average_time_per_step_s = sum(times) / len(times) / n_steps
    average_time_per_step_us = average_time_per_step_s * 1000000
    
    steps_per_hour = 3600 / average_time_per_step_s
    minutes_needed = 1E10 / steps_per_hour / 8 * 60  # 8 processes in parallel

    print(f"Time needed to explore 10^10 sequences with 8 process in parallel: {minutes_needed:.2e} minutes")
    print(f"Number of sequences explored per hour: {steps_per_hour:.2e}")
    print(f"Average execution time per step: {average_time_per_step_us:.5f} microseconds")

if __name__ == '__main__':
    benchmark_montecarlo_steps()
    #annealing(n_steps=1E8)
    import warnings
    import numpy as np

    # Convert overflow warnings to exceptions
    warnings.filterwarnings('error', 'overflow encountered in power', category=RuntimeWarning)


    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")

    temperatures=np.logspace(0,5,11)
    seq_indices=np.random.randint(1, 21, size=(len(temperatures),len(model.sequence)))
    print(len(seq_indices))
    parallel_tempering(model.potts_model['h'], model.potts_model['J'], model.mask, seq_indices, temperatures, n_steps=int(1E10), n_steps_per_cycle=int(1E3), Ep=10)
