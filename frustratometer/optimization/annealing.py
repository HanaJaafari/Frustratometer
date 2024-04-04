import random
import numpy as np
import frustratometer

mcso_seq_output_file = "mcso_sequence.txt"
mcso_energy_output_file = "mcso_energy.txt"

def sequence_permutation(sequence):
    sequence = sequence.copy()
    res1, res2 = random.sample(range(len(sequence)), 2)
    sequence[res1], sequence[res2] = sequence[res2], sequence[res1]
    return sequence

def sequence_mutation(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = sequence.copy()
    res = random.randint(0, len(sequence) - 1)
    aa = random.choice(amino_acids)
    sequence[res] = aa
    return sequence

def heterogeneity(sequence):
    N = len(sequence)
    _, counts = np.unique(sequence, return_counts=True)
    denominator = np.prod(np.array([np.math.factorial(count) for count in counts]))
    het = np.math.factorial(N) / denominator
    return np.log(het)

def mutation_process(temperature, model, sequence, Ep=100):
    kb = 0.001987
    energy = model.native_energy(sequence)
    het = heterogeneity(sequence)
    for _ in range(1000):
        new_sequence = sequence_permutation(sequence) if random.random() > 0.5 else sequence_mutation(sequence)
        new_energy = model.native_energy(new_sequence)
        new_het = heterogeneity(new_sequence)
        energy_difference = new_energy - energy
        het_difference = new_het - het
        acceptance_probability = min(1, np.exp((-energy_difference + Ep * het_difference) / (kb * temperature)))
        if random.random() < acceptance_probability:
            sequence = new_sequence
            energy = new_energy
            het = new_het
    return sequence

if __name__ == '__main__':
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    sequence = list("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")
    
    with open(mcso_seq_output_file, 'w') as seq_file, open(mcso_energy_output_file, 'w') as energy_file:
        for temp in range(800, 1, -1):
            sequence = mutation_process(temp, model, sequence)
            seq_file.write(f'Temperature: {temp} Sequence: {" ".join(sequence)}\n')
            energy_file.write(f'Temperature: {temp} Energy: {model.native_energy(sequence)}\n')
            print(temp, ''.join(sequence), model.native_energy(sequence), heterogeneity(sequence))