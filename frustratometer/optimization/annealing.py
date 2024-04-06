import random
import numpy as np
import frustratometer
import pandas as pd  # Import pandas for data manipulation

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def index_to_sequence(seq_index):
    """Converts sequence index array back to sequence string."""
    return ''.join([_AA[index] for index in seq_index])

def sequence_to_index(sequence):
    """Converts sequence string to sequence index array."""
    return np.array([_AA.find(aa) for aa in sequence])

def sequence_swap(seq_index, potts_model,mask):
    seq_index = seq_index.copy()
    res1, res2 = random.sample(range(len(seq_index)), 2)
    
    het_difference = 0
    energy_difference = compute_swap_energy(potts_model, mask, seq_index, res1, res2)

    seq_index[res1], seq_index[res2] = seq_index[res2], seq_index[res1]

    return seq_index, het_difference, energy_difference

def compute_swap_energy(potts_model, mask, seq_index, pos1, pos2):
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

def sequence_mutation(seq_index, potts_model,mask):
    seq_index = seq_index.copy()
    res = random.randint(0, len(seq_index) - 1)
    aa_new = random.choice(range(1, 21))

    aa_old_count = np.sum(seq_index == seq_index[res])
    aa_new_count = np.sum(seq_index == aa_new)
    
    het_difference = np.log(aa_old_count/ (aa_new_count+1))
    energy_difference = compute_mutation_energy(potts_model, mask, seq_index, res, aa_new)

    seq_index[res] = aa_new
    
    return seq_index, het_difference, energy_difference

def compute_mutation_energy(potts_model, mask, seq_index, pos, aa_new):
    aa_old=seq_index[pos]
    energy_difference = -potts_model['h'][pos,aa_new] + potts_model['h'][pos,aa_old]

    reduced_j = potts_model['J'][range(len(seq_index)), :, seq_index, :]
    j_correction = reduced_j[:, pos, aa_old] * mask[pos]
    j_correction -= reduced_j[:, pos, aa_new] * mask[pos]
    
    # J correction, interaction with self aminoacids
    energy_difference += j_correction.sum(axis=0)
    return energy_difference

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

def heterogeneity_approximation(seq_index):
    """
    Uses Stirling's approximation to calculate the heterogeneity of a sequence
    """
    N = len(seq_index)
    _, counts = np.unique(seq_index, return_counts=True)
    def stirling_log(n):
        if n < 40:
            return np.log(np.math.factorial(n))
        else:
            return n * np.log(n / np.e) + 0.5 * np.log(2 * np.pi * n) + 1.0 / (12 * n)
    
    log_n_factorial = stirling_log(N)
    log_denominator = sum(stirling_log(count) for count in counts)
    het = log_n_factorial - log_denominator
    return het

def montecarlo_steps(temperature, potts_model, mask, seq_index, Ep=100, n_steps = 1000):
    kb = 0.001987
    energy = native_energy(seq_index, potts_model,mask)
    het = heterogeneity_approximation(seq_index)
    for _ in range(n_steps):
        new_sequence, het_difference, energy_difference = sequence_swap(seq_index, potts_model, mask) if random.random() > 0.5 else sequence_mutation(seq_index, potts_model, mask)
        exponent=(-energy_difference + Ep * het_difference) / (kb * temperature)
        acceptance_probability = np.exp(min(0, exponent))
        if random.random() < acceptance_probability:
            seq_index = new_sequence
            energy += energy_difference
            het += het_difference
    return seq_index, energy, het

def annealing():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")
    
    simulation_data = []
    for temp in range(800, 1, -1):
        seq_index, energy, het = montecarlo_steps(temp, model.potts_model, model.mask, seq_index, Ep=10, n_steps=1000)
        simulation_data.append({'Temperature': temp, 'Sequence': index_to_sequence(seq_index), 'Energy': energy, 'Heterogeneity': het})
        print(temp, index_to_sequence(seq_index), energy, het)
    simulation_df = pd.DataFrame(simulation_data)
    simulation_df.to_csv("mcso_simulation_results.csv", index=False)

def test_heterogeneity_approximation():
    sequence = list("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLP")
    het = heterogeneity(sequence)
    het_approx = heterogeneity_approximation(sequence)
    assert np.isclose(het, het_approx), f"Heterogeneity: {het}, Approximation: {het_approx}"

def test_heterogeneity_difference_permutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_swap(seq_index, model.potts_model,model.mask)
    het = heterogeneity_approximation(seq_index)
    new_het = heterogeneity_approximation(new_sequence)
    het_difference2 = new_het - het
    assert np.isclose(het_difference, het_difference2), f"Heterogeneity difference: {het_difference}, {het_difference2}"

def test_heterogeneity_difference_mutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_mutation(seq_index, model.potts_model,model.mask)
    print(model.sequence)
    print(index_to_sequence(new_sequence))
    het = heterogeneity_approximation(seq_index)
    new_het = heterogeneity_approximation(new_sequence)
    print(het, new_het)
    het_difference2 = new_het - het
    assert np.isclose(het_difference, het_difference2), f"Heterogeneity difference: {het_difference}, {het_difference2}"

def test_energy_difference_permutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_swap(seq_index, model.potts_model,model.mask)
    energy = native_energy(seq_index, model.potts_model,model.mask)
    new_energy = native_energy(new_sequence, model.potts_model,model.mask)
    energy_difference2 = new_energy - energy
    assert np.isclose(energy_difference, energy_difference2), f"Energy difference: {energy_difference}, {energy_difference2}"

def test_energy_difference_mutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_mutation(seq_index, model.potts_model,model.mask)
    energy = native_energy(seq_index, model.potts_model,model.mask)
    new_energy = native_energy(new_sequence, model.potts_model,model.mask)
    energy_difference2 = new_energy - energy
    assert np.isclose(energy_difference, energy_difference2), f"Energy difference: {energy_difference}, {energy_difference2}"



if __name__ == '__main__':
    test_heterogeneity_approximation()
    test_heterogeneity_difference_permutation()
    test_heterogeneity_difference_mutation()
    test_energy_difference_permutation()
    test_energy_difference_mutation()
    annealing()

