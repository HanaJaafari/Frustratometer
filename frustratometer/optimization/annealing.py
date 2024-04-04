import random
import math
import frustratometer
from math import exp, log, factorial
from collections import Counter

mcso_seq_output_file = "mcso_sequence.txt"
mcso_energy_output_file = "mcso_energy.txt"

def frust(model, seq):
    energy = model.native_energy(seq)
    return energy

def permut(sequence):
    rand_res_1, rand_res_2 = random.sample(range(len(sequence)), 2)
    sequence[rand_res_1], sequence[rand_res_2] = sequence[rand_res_2], sequence[rand_res_1]
    return sequence

def point_mut(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    posi2mut = random.randint(0, len(sequence) - 1)
    aa2mut = random.choice(amino_acids)
    sequence = sequence[:posi2mut] + [aa2mut] + sequence[posi2mut + 1:]
    return sequence

def np(seq):
    N = len(seq)
    amino_acid_counts = Counter(seq)
    np_numerator = factorial(N)
    np_denominator = 1
    for count in amino_acid_counts.values():
        np_denominator *= factorial(count)
    return np_numerator / np_denominator

def mutation_process(temp,model,seq):
    k_b = 0.001987  # Boltzmann constant in kcal/mol*K
    Ep = 100 # Scale factor for heterogeneity in pacc, set 1 for testing
    mcso_seq = seq
    for _ in range(1000): # Maybe 1000 for formal testing
                
        # Perform permutation or point mutation
        x = random.random()
        if x > 0.5:
            mut_seq = permut(seq[:])
        else:
            mut_seq = point_mut(seq[:])
        
        native_energy = frust(model, seq)  
        mutated_energy = frust(model, mut_seq)
        energy_difference = mutated_energy - native_energy #mut1 nat2  #-

        np_old = np(seq)
        np_new = np(mut_seq)
        pacc = min(1, exp((-energy_difference + Ep * log(np_new / np_old)) / (k_b * temp))) #- -> <0
        random_probability = random.random()
        # if random_probability > math.exp(-energy_difference / (k_b * temp)):# Will generate homopolymer?
        if random_probability < pacc:
            mcso_seq = mut_seq
        


        seq=mcso_seq
    return seq

if __name__=='__main__':
    
    native_pdb = "tests/data/1r69.pdb"  
    structure = frustratometer.Structure.full_pdb(native_pdb,"A")
    model = frustratometer.AWSEM(structure,distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq = list("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")
    with open(mcso_seq_output_file, 'w') as seq_file, open(mcso_energy_output_file, 'w') as energy_file:
        for temp in range(800, 199, -1):# Sample temperatures For simulated annealing
            print(temp)
            seq = mutation_process(temp,model,seq)
            seq_file.write(f'Temperature: {temp} Sequence: {" ".join(seq)}\n')
            energy_file.write(f'Temperature: {temp} Energy: {frust(model, seq)}\n')
