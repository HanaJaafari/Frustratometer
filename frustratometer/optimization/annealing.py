import random
import numpy as np
import frustratometer
import pandas as pd  # Import pandas for data manipulation

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def sequence_permutation(sequence, model):
    sequence = sequence.copy()
    res1, res2 = random.sample(range(len(sequence)), 2)
    sequence[res1], sequence[res2] = sequence[res2], sequence[res1]
    het_difference = 0
    
    
    
    
    
    energy_difference = 0
    return sequence, het_difference, energy_difference

def sequence_mutation(sequence, model):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = sequence.copy()
    res = random.randint(0, len(sequence) - 1)
    aa_new = random.choice(amino_acids)
    aa_old=sequence[res]
    
    het_difference = np.log(sequence.count(aa_old)/ (sequence.count(aa_new)+1))
    energy_difference = -model.potts_model['h'][res,_AA.find(aa_new)] + model.potts_model['h'][res,_AA.find(aa_old)]

    reduced_j = model.potts_model['J'][range(len(sequence)), :, np.array([_AA.find(aa) for aa in sequence]), :]
    print(reduced_j.shape)
    j_correction = reduced_j[:, res, _AA.find(aa_old)] * model.mask[res]
    j_correction -= reduced_j[:, res, _AA.find(aa_new)] * model.mask[res]
    
    # J correction, interaction with self aminoacids
    energy_difference += j_correction.sum(axis=0)

    sequence[res] = aa_new
    
    return sequence, het_difference, energy_difference

def heterogeneity(sequence):
    N = len(sequence)
    _, counts = np.unique(sequence, return_counts=True)
    denominator = np.prod(np.array([np.math.factorial(count) for count in counts]))
    het = np.math.factorial(N) / denominator
    return np.log(het)

def heterogeneity_approximation(sequence):
    """
    Uses Stirling's approximation to calculate the heterogeneity of a sequence
    """
    N = len(sequence)
    _, counts = np.unique(sequence, return_counts=True)
    def stirling_log(n):
        if n < 40:
            return np.log(np.math.factorial(n))
        else:
            return n * np.log(n / np.e) + 0.5 * np.log(2 * np.pi * n) + 1.0 / (12 * n)
    
    log_n_factorial = stirling_log(N)
    log_denominator = sum(stirling_log(count) for count in counts)
    het = log_n_factorial - log_denominator
    return het

def montecarlo_steps(temperature, model, sequence, Ep=100, n_steps = 1000):
    kb = 0.001987
    energy = model.native_energy(sequence)
    het = heterogeneity_approximation(sequence)
    for _ in range(n_steps):
        new_sequence, het_difference, energy_difference = sequence_permutation(sequence, model) if random.random() > 0.5 else sequence_mutation(sequence, model)
        new_energy = model.native_energy(new_sequence)
        new_het = heterogeneity_approximation(new_sequence)
        energy_difference = new_energy - energy
        het_difference = new_het - het
        exponent=(-energy_difference + Ep * het_difference) / (kb * temperature)
        acceptance_probability = np.exp(min(0, exponent))
        if random.random() < acceptance_probability:
            sequence = new_sequence
            energy = new_energy
            het = new_het
    return sequence, energy, het

def test_heterogeneity_approximation():
    sequence = list("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLP")
    het = heterogeneity(sequence)
    het_approx = heterogeneity_approximation(sequence)
    assert np.isclose(het, het_approx), f"Heterogeneity: {het}, Approximation: {het_approx}"

def test_heterogeneity_difference_permutation():
    sequence = list("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLP")
    new_sequence, het_difference, energy_difference=sequence_permutation(sequence)
    het = heterogeneity_approximation(sequence)
    new_het = heterogeneity_approximation(new_sequence)
    het_difference2 = new_het - het
    assert np.isclose(het_difference, het_difference2), f"Heterogeneity difference: {het_difference}, {het_difference2}"

def test_heterogeneity_difference_mutation():
    sequence = list("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLP")
    new_sequence, het_difference, energy_difference=sequence_mutation(sequence)
    het = heterogeneity_approximation(sequence)
    new_het = heterogeneity_approximation(new_sequence)
    het_difference2 = new_het - het
    assert np.isclose(het_difference, het_difference2), f"Heterogeneity difference: {het_difference}, {het_difference2}"

def test_energy_difference_permutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    sequence = list(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_permutation(sequence, model)
    energy = model.native_energy(sequence)
    new_energy = model.native_energy(new_sequence)
    energy_difference2 = new_energy - energy
    assert np.isclose(energy_difference, energy_difference2), f"Energy difference: {energy_difference}, {energy_difference2}"

def test_energy_difference_mutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    sequence = list(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_mutation(sequence, model)
    energy = model.native_energy(sequence)
    new_energy = model.native_energy(new_sequence)
    energy_difference2 = new_energy - energy
    assert np.isclose(energy_difference, energy_difference2), f"Energy difference: {energy_difference}, {energy_difference2}"

if __name__ == '__main__':
    test_heterogeneity_approximation()
    #test_heterogeneity_difference_permutation()
    #test_heterogeneity_difference_mutation()
    test_energy_difference_permutation()
    test_energy_difference_mutation()

    # native_pdb = "tests/data/1r69.pdb"
    # structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    # model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    # sequence = list("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")
    
    # simulation_data = []
    # for temp in range(800, 1, -1):
    #     sequence, energy, het = montecarlo_steps(temp, model, sequence, Ep=10, n_steps=1000)
    #     simulation_data.append({'Temperature': temp, 'Sequence': ''.join(sequence), 'Energy': energy, 'Heterogeneity': het})
    #     print(temp, ''.join(sequence), energy, het)
    # simulation_df = pd.DataFrame(simulation_data)
    # simulation_df.to_csv("mcso_simulation_results.csv", index=False)