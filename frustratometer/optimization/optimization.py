import numpy as np
from frustratometer.classes import Frustratometer
from frustratometer.classes import Structure
from frustratometer.classes import AWSEM
import pandas as pd  # Import pandas for data manipulation
import numba
from pathlib import Path
from frustratometer.optimization.EnergyTerm import EnergyTerm
import math
from frustratometer.optimization.inner_product import compute_all_region_means
from frustratometer.optimization.inner_product import create_region_masks_1_by_1
from frustratometer.optimization.inner_product import create_region_masks_1_by_2
from frustratometer.optimization.inner_product import create_region_masks_2_by_2
from frustratometer.optimization.inner_product import build_mean_inner_product_matrix
from frustratometer.optimization.inner_product import diff_mean_inner_product_matrix
import itertools

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def index_to_sequence(seq_index, alphabet):
    """Converts sequence index array back to sequence string."""
    return ''.join([alphabet[index] for index in seq_index])

def sequence_to_index(sequence, alphabet):
    """Converts sequence string to sequence index array."""
    return np.array([alphabet.find(aa) for aa in sequence])

@numba.njit
def random_seed(seed):
    np.random.seed(seed)

class Zero(EnergyTerm):
    @staticmethod
    def compute_energy(seq_index:np.array):
        return 0.
    
    @staticmethod
    def compute_denergy_mutation(seq_index:np.array, pos:int, aa):
        return 0.
    
    @staticmethod
    def compute_denergy_swap(seq_index:np.array, pos1:int, pos2:int):
        return 0.

class Heterogeneity(EnergyTerm):
    def __init__(self,use_numba=True, exact=False, alphabet=_AA):
        self.use_numba=use_numba
        self.exact=exact
        self.alphabet=alphabet
        self.alphabet_size=len(alphabet)
        self.initialize_functions()

    def initialize_functions(self):
        log_factorial_table=np.log(np.array([math.factorial(i) for i in range(20)],dtype=np.float64))
        alphabet_size=self.alphabet_size
        def stirling_log(n):
            if n < 20:
                return log_factorial_table[n]
            else:
                return n * np.log(n / np.e) + 0.5 * np.log(2 * np.pi * n) + 1.0 / (12 * n)

        stirling_log=self.numbify(stirling_log)

        def heterogeneity_exact(seq_index):
            n = len(seq_index)
            counts = np.bincount(seq_index, minlength=alphabet_size)
            denominator = np.prod(np.array([math.gamma(count+1) for count in counts]))
            het = math.gamma(n+1) / denominator
            return np.log(het)

        def heterogeneity_approximation(seq_index):
            """
            Uses Stirling's approximation to calculate the sequence entropy
            """
            N = len(seq_index)
            counts = np.zeros(21, dtype=np.int32)
            
            for val in seq_index:
                counts[val] += 1
                
            log_n_factorial = stirling_log(N)
            log_denominator = sum([stirling_log(count) for count in counts])   
            het = log_n_factorial - log_denominator
            return het
        
        def dheterogeneity(seq_index, pos, aa):
            aa_old_count = np.sum(seq_index == seq_index[pos])
            aa_new_count = np.sum(seq_index == aa)
            return np.log(aa_old_count / (aa_new_count+(seq_index[pos]!=aa)))
        
        if self.exact:
            self.compute_energy=heterogeneity_exact
        else:
            self.compute_energy=heterogeneity_approximation

        self.compute_denergy_mutation=dheterogeneity

class AWSEM_dE(EnergyTerm):
    def __init__(self, use_numba=True, model=Frustratometer, alphabet=_AA):
        self.use_numba=use_numba
        self.model_h = model.potts_model['h']
        self.model_J = model.potts_model['J']
        self.mask = model.mask

        if alphabet!=_AA:
            self.reindex_dca=[_AA.index(aa) for aa in alphabet]
            self.model_h=self.model_h[:,self.reindex_dca]
            self.model_J=self.model_J[:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.initialize_functions()
    
    def initialize_functions(self):
        mask=self.mask
        model_h=self.model_h
        model_J=self.model_J
        
        def compute_energy(seq_index: np.array) -> float:
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

        def compute_denergy_mutation(seq_index: np.ndarray, pos: int, aa_new: int) -> float:
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

        def compute_denergy_swap(seq_index, pos1, pos2):
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
        
        self.compute_energy=compute_energy
        self.compute_denergy_mutation=compute_denergy_mutation
        self.compute_denergy_swap=compute_denergy_swap

class AWSEM_dE2(EnergyTerm):      
    def __init__(self, use_numba=True, model=Frustratometer, gamma=None, alphabet=_AA):
        self.use_numba=use_numba
        self.model_h = model.potts_model['h']
        self.model_J = model.potts_model['J']
        self.mask = model.mask
        assert "indicators" in model.__dict__.keys(), "Indicator functions were not exposed. Initialize AWSEM function with `expose_indicator_functions=True` first."
        self.indicators = model.indicators
        self.alphabet_size=len(alphabet)
        assert gamma is not None, "Gamma matrix was not provided, please provide a gamma matrix as an argument"
        self.gamma=gamma
        self.initialize_functions()
    
    def initialize_functions(self):
        indicators1D=np.array(self.indicators[0:3])
        indicators2D=np.array(self.indicators[3:6])
        print([a.shape for a in indicators1D])
        print([a.shape for a in indicators2D])
        len_alphabet=self.alphabet_size
        gamma=self.gamma
        
        mask1x1=create_region_masks_1_by_1(len_alphabet)
        mask1x2=create_region_masks_1_by_2(len_alphabet)
        mask2x2=create_region_masks_2_by_2(len_alphabet)
        region_means=compute_all_region_means(indicators1D,indicators2D)
       
        #1/0
        if use_numba:
            #1/0                
            pass
            
        def compute_energy(seq_index):
            aa_count = np.bincount(seq_index, minlength=len_alphabet)
            print(aa_count.shape)
            freq_i=aa_count
            freq_ij=np.outer(freq_i,freq_i)
            alpha = np.diag(freq_i)
            beta = freq_ij.copy()
            np.fill_diagonal(beta, freq_i*(freq_i-1))

            #phi_len=sum([len_alphabet**len(ind.shape) for ind in indicators])
            phi_len= indicators1D.shape[0]*len_alphabet + indicators2D.shape[0]*len_alphabet**2
            phi_mean = np.zeros(phi_len)
            offset=0
            for indicator in indicators1D:
                phi_mean[offset:offset+len_alphabet]=np.mean(indicator)*freq_i
                offset += len_alphabet
            for indicator in indicators2D:
                temp_indicator=indicator.copy()
                mean_diagonal_indicator = np.diag(temp_indicator).mean()
                np.fill_diagonal(temp_indicator, 0)
                mean_offdiagonal_indicator = temp_indicator.mean()
                
                phi_mean[offset:offset+len_alphabet**2]=alpha.ravel()*mean_diagonal_indicator + beta.ravel()*mean_offdiagonal_indicator
                offset += len_alphabet**2
            
            B = build_mean_inner_product_matrix(freq_i.copy(),indicators1D.copy(),indicators2D.copy()) - np.outer(phi_mean,phi_mean)
            return gamma @ B @ gamma
        
        compute_energy_numba=self.numbify(compute_energy)
        
        #TODO: Code this function using diff_mean_inner_product_matrix
        def denergy_mutation(seq_index, pos, aa):
            seq_index_new = seq_index.copy()
            seq_index_new[pos] = aa
            return compute_energy_numba(seq_index_new) - compute_energy_numba(seq_index)
        
        self.compute_energy = compute_energy
        self.compute_denergy_mutation = denergy_mutation

@numba.njit
def sequence_swap(seq_index, model_h, model_J, mask):
    seq_index = seq_index.copy()
    n=len(seq_index)
    res1 = np.random.randint(0,n)
    res2 = np.random.randint(0,n-1)
    res2 += (res2 >= res1)
    
    het_difference = 0
    de2_difference = 0
    energy_difference = compute_swap_energy(seq_index, model_h, model_J, mask, res1, res2)

    seq_index[res1], seq_index[res2] = seq_index[res2], seq_index[res1]

    return seq_index, het_difference, de2_difference, energy_difference

@numba.njit
def sequence_mutation(seq_index, model_h, model_J, mask, indicators, gamma, valid_indices=np.arange(len(_AA))):
    seq_index_new = seq_index.copy()
    r = np.random.randint(0, len(valid_indices)*len(seq_index)) # Select a random index
    res = r // len(valid_indices)
    aa_new = valid_indices[r % len(valid_indices)] 

    aa_old_count = np.sum(seq_index == seq_index[res])
    aa_new_count = np.sum(seq_index == aa_new)

    seq_index_new[res] = aa_new
    
    het_difference = np.log(aa_old_count / (aa_new_count+1))
    de2_difference = compute_dE2(gamma, seq_index_new, indicators, len_alphabet=len(valid_indices)) - compute_dE2(gamma, seq_index, indicators, len_alphabet=len(valid_indices))
    energy_difference = compute_mutation_energy(seq_index, model_h, model_J, mask, res, aa_new)

    return seq_index_new, het_difference, de2_difference, energy_difference



@numba.njit
def montecarlo_steps(temperature, model_h, model_J, mask, indicators, gamma, seq_index, Ep=10, Ee=10, n_steps = 1000, kb = 0.008314,valid_indices=np.arange(len(_AA))) -> np.array:
    for _ in range(n_steps):
        new_sequence, het_difference, de2_difference, energy_difference = sequence_swap(seq_index, model_h, model_J, mask) if np.random.random() > 0.5 else sequence_mutation(seq_index, model_h, model_J, mask, indicators, gamma, valid_indices)
        exponent=(-energy_difference + Ep * het_difference + Ee * de2_difference ) / (kb * temperature + 1E-10)
        acceptance_probability = np.exp(min(0, exponent)) 
        if np.random.random() < acceptance_probability:
            seq_index = new_sequence
    return seq_index

@numba.njit
def replica_exchanges(energies, temperatures, kb=0.008314, exchange_id=0):
    """
    Determine pairs of configurations between replicas for exchange.
    Returns a list of tuples with the indices of replicas to be exchanged.
    """
    n_replicas = len(temperatures)
    start_index = exchange_id % 2
    order = np.arange(len(temperatures), dtype=np.int64)
    
    for i in np.arange(start_index, n_replicas - 1, 2):
        energy1, energy2 = energies[i], energies[i + 1]
        temp1, temp2 = temperatures[i], temperatures[i + 1]
        delta = (1/temp2 - 1/temp1) * (energy2 - energy1)
            
        exponent = delta / kb # Sign is correct, as we want to swap when the system with higher temperature has lower energy
        prob = np.exp(min(0., exponent)) 

        if 1 <= prob:
            order[i]=i+1
            order[i+1]=i
    return order

@numba.njit(parallel=True)
def parallel_montecarlo_step(model_h, model_J, mask, indicators, gamma, seq_indices, temperatures, n_steps_per_cycle, Ep,valid_indices=np.arange(len(_AA))):
    n_replicas = len(temperatures)
    energies = np.zeros(n_replicas)
    heterogeneities = np.zeros(n_replicas)
    total_energies = np.zeros(n_replicas)
    delta_e2 = np.zeros(n_replicas)
    for i in numba.prange(n_replicas):
        temp_seq_index = seq_indices[i]
        seq_indices[i] = montecarlo_steps(temperatures[i], model_h, model_J, mask, indicators, gamma, seq_index=temp_seq_index, Ep=Ep, n_steps=n_steps_per_cycle,valid_indices=valid_indices)
        energy = model_energy(seq_indices[i], model_h, model_J, mask)
        het = heterogeneity_approximation(seq_indices[i])
        de2 = compute_dE2(gamma,seq_indices[i],indicators,len(valid_indices))
        # Compute energy for the new sequence
        total_energies[i] = energy - Ep * het # Placeholder for actual energy calculation
        energies[i] = energy
        heterogeneities[i] = het
        delta_e2[i] = de2

    
    return seq_indices, energies, heterogeneities, delta_e2, total_energies

@numba.njit
def parallel_tempering_numba(model_h, model_J, mask, indicators, gamma, seq_indices, temperatures, n_steps, n_steps_per_cycle, Ep,valid_indices=np.arange(len(_AA))):
    for s in range(n_steps//n_steps_per_cycle):
        seq_indices, energy, het, de2, total_energies = parallel_montecarlo_step(model_h, model_J, mask, indicators, gamma, seq_indices, temperatures, n_steps_per_cycle, Ep,valid_indices=valid_indices)

        # Yield data every 10 exchanges
        if s % 10 == 9:
            yield s, seq_indices, energy, het, de2, total_energies

        # Perform replica exchanges
        order = replica_exchanges(total_energies, temperatures, exchange_id=s)
        seq_indices = seq_indices[order]
        


def parallel_tempering(model_h, model_J, mask, indicators, gamma, seq_indices, temperatures, n_steps, n_steps_per_cycle, Ep, filename="parallel_tempering_resultsv3.csv",valid_indices=np.arange(len(_AA)),alphabet=_AA):
    columns=['Step', 'Temperature', 'Sequence', 'Energy', 'Heterogeneity', 'Total Energy']
    df_headers = pd.DataFrame(columns=columns)
    df_headers.to_csv(filename, index=False)
    print(*columns, sep='\t')

    # Run the simulation and append data periodically
    for s, updated_seq_indices, energy, het, de2, total_energy in parallel_tempering_numba(model_h, model_J, mask, indicators, gamma, seq_indices, temperatures, n_steps, n_steps_per_cycle, Ep,valid_indices=valid_indices):
        # Prepare data for this chunk
        data_chunk = []
        for i, temp in enumerate(temperatures):
            sequence_str = index_to_sequence(updated_seq_indices[i],alphabet=alphabet)  # Convert sequence index back to string
            #total_energy = energy[i] - Ep * het[i]
            data_chunk.append({'Step': (s+1) * n_steps_per_cycle, 'Temperature': temp, 'Sequence': sequence_str, 'Energy': energy[i], 'Heterogeneity': het[i], 'DE2': de2[i], 'Total Energy': total_energy[i]})
        
        # Convert the chunk to a DataFrame and append it to the CSV
        df_chunk = pd.DataFrame(data_chunk)
        print(*df_chunk.iloc[-1].values, sep='\t')
        df_chunk.to_csv(filename, mode='a', header=False, index=False)


def annealing(temp_max=500, temp_min=0, n_steps=1E8, Ep=10,valid_indices=np.arange(len(_AA))):
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")
    
    simulation_data = []
    n_steps_per_cycle=n_steps//(temp_max-temp_min)
    for temp in range(temp_max, temp_min, -1):
        seq_index= montecarlo_steps(temp, model.potts_model['h'], model.potts_model['J'], model.mask, seq_index, Ep=Ep, n_steps=n_steps_per_cycle,valid_indices=valid_indices)
        energy = model_energy(seq_index, model.potts_model['h'],model.potts_model['J'], model.mask)
        het = heterogeneity_approximation(seq_index)
        simulation_data.append({'Temperature': temp, 'Sequence': index_to_sequence(seq_index), 'Energy': energy, 'Heterogeneity': het, 'Total Energy': energy - Ep * het})
        print(temp, index_to_sequence(seq_index), energy - Ep * het, energy, het)
    simulation_df = pd.DataFrame(simulation_data)
    simulation_df.to_csv("mcso_simulation_results.csv", index=False)

def benchmark_montecarlo_steps(n_repeats=100, n_steps=20000,valid_indices=np.arange(len(_AA))):
    import time
    # Initialize the model for 1r69
    native_pdb = "tests/data/1r69.pdb"  # Ensure this path is correct
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    
    seq_len = len(model.sequence)
    times = []

    #Adds one step for numba compilation time
    montecarlo_steps(temperature=500, model_h=model.potts_model['h'], model_J=model.potts_model['J'], mask=model.mask, seq_index=sequence_to_index(model.sequence), Ep=100, n_steps=1,valid_indices=valid_indices)

    for _ in range(n_repeats):  # Run benchmark 10 times
        # Generate a new random sequence for each run
        seq_index = np.random.randint(1, 21, size=seq_len)
        start_time = time.time()
        
        montecarlo_steps(temperature=500, model_h=model.potts_model['h'], model_J=model.potts_model['J'], mask=model.mask, seq_index=seq_index, Ep=100, n_steps=n_steps,valid_indices=valid_indices)
        
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
    
    native_pdb = "tests/data/1r69.pdb"
    structure = Structure.full_pdb(native_pdb, "A")
    reduced_alphabet = 'ADEFGHIKLMNQRSTVWY'
    
    model = AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2,expose_indicator_functions=True)
    reindex_dca=[_AA.index(aa) for aa in reduced_alphabet]
    gamma = np.concatenate([g for g in model.gamma['Burial'][:,model.aa_map_awsem_list][:,reindex_dca]] +\
                           [model.gamma['Direct'][0, model.aa_map_awsem_x, model.aa_map_awsem_y][reindex_dca][:,reindex_dca].ravel()] +\
                           [model.gamma['Water'][0, model.aa_map_awsem_x, model.aa_map_awsem_y][reindex_dca][:,reindex_dca].ravel()] +\
                           [model.gamma['Protein'][0, model.aa_map_awsem_x, model.aa_map_awsem_y][reindex_dca][:,reindex_dca].ravel()])
    
    seq_indices = np.random.randint(0, len(reduced_alphabet), size=(1,len(structure.sequence)))
    #Tests
    # for exact,use_numba in [(True,False),(False,False),(True,True),(False,True)]:
    #     het=Heterogeneity(exact=exact,use_numba=use_numba)
    #     for i in range(1):
    #         het.test(seq_indices[i])
    
    # for use_numba in [False, True]:
    #     awsem_energy = AWSEM_dE(use_numba=use_numba, model=model, alphabet=reduced_alphabet)
    #     for i in range(1):
    #         awsem_energy.test(seq_indices[i])

    for use_numba in [False, True]:
        awsem_de2 = AWSEM_dE2(use_numba=use_numba, model=model, alphabet=reduced_alphabet, gamma=gamma)
        for i in range(1):
            awsem_de2.test(seq_indices[0])
    
    # import warnings
    # import numpy as np

    # AA_DCA = '-ACDEFGHIKLMNPQRSTVWY'
    # reduced_alphabet='ADEFGHIKLMNQRSTVWY'
    # reindex_dca=[AA_DCA.index(aa) for aa in reduced_alphabet]
    
    # #Reformat the potts models and indicator functions to account for the excluded amino acids
    # native_pdb = "tests/data/1r69.pdb"
    # structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    # model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2,expose_indicator_functions=True)
    # gamma=np.concatenate([g for g in model.gamma['Burial'][:,model.aa_map_awsem_list][:,reindex_dca]] +\
    #                       [model.gamma['Direct'][0, model.aa_map_awsem_x, model.aa_map_awsem_y][reindex_dca][:,reindex_dca].ravel()] +\
    #                       [model.gamma['Water'][0, model.aa_map_awsem_x, model.aa_map_awsem_y][reindex_dca][:,reindex_dca].ravel()] +\
    #                       [model.gamma['Protein'][0, model.aa_map_awsem_x, model.aa_map_awsem_y][reindex_dca][:,reindex_dca].ravel()])


    # potts_h = model.potts_model['h'][:,reindex_dca]
    # potts_J = model.potts_model['J'][:,:,reindex_dca][:,:,:,reindex_dca]

    # temperatures=np.logspace(0,6,7)
    # seq_indices=np.random.randint(0, len(reduced_alphabet), size=(len(temperatures),len(model.sequence)))
    # valid_indices=np.arange(len(reduced_alphabet))

    # parallel_tempering(potts_h, potts_J, model.mask, model.indicators, gamma, seq_indices, temperatures, n_steps=int(1E3), n_steps_per_cycle=int(1E1), Ep=10, alphabet=reduced_alphabet, valid_indices=valid_indices, )
    pass

#Scratch
        # def mean_inner_product_2_by_2(repetitions,i0,i1):
        #     n_elements= len(repetitions)

        #     # Create the mean_inner_product array
        #     mean_inner_product = np.zeros((n_elements, n_elements, n_elements, n_elements)).flatten()
            
        #     # Create arrays of indices for elements
        #     #n_i, n_j, n_k, n_l = np.meshgrid(*[repetitions]*4, indexing='ij', sparse=False)
        #     n_i = np.repeat(repetitions, n_elements**3).reshape(n_elements, n_elements, n_elements, n_elements)
        #     n_j = n_i.copy().transpose(3, 0, 1, 2).flatten()
        #     n_k = n_i.copy().transpose(2, 3, 0, 1).flatten()
        #     n_l = n_i.copy().transpose(1, 2, 3, 0).flatten()
        #     n_i=n_i.flatten()

        #     mask_names = ['ijkl', 'iikl', 'ijil', 'ijjl', 'ijki', 'ijkj', 'ijkk',
        #                   'iiil', 'iiki', 'iikk', 'ijii', 'ijij', 'ijji', 'ijjj', 'iiii']
            
        #     nm={mask_name:mask_name+'_'+str(i0)+'_'+str(i1) for mask_name in mask_names}
        #     #mask={mask_name:mask for mask_name,mask in zip(mask2x2_names,mask2x2_values)}
        #     mask=mask2x2_array
        #     #region_means=dict(a:bregion_means_global)


        #     m = mask['iiii']
        #     mean_inner_product[m] = (
        #         n_i[m] * region_means[nm['iiii']] +
        #         n_i[m] * (n_i[m]-1) * (region_means[nm['iikk']] + region_means[nm['ijij']] + region_means[nm['ijji']] + region_means[nm['iiil']] + region_means[nm['iiki']] + region_means[nm['ijjj']] + region_means[nm['ijii']]) +
        #         n_i[m] * (n_i[m]-1) * (n_i[m]-2) * (region_means[nm['ijil']] + region_means[nm['ijjl']] + region_means[nm['ijki']] + region_means[nm['ijkj']] + region_means[nm['iikl']] + region_means[nm['ijkk']]) +
        #         n_i[m] * (n_i[m]-1) * (n_i[m]-2) * (n_i[m]-3) * region_means[nm['ijkl']]
        #     )
            
        #     # m = masks['iiii']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * region_means[i0,i1,'iiii'] +
        #     #     n_i[m] * (n_i[m]-1) * (region_means[i0,i1,'iikk'] + region_means[i0,i1,'ijij'] + region_means[i0,i1,'ijji'] + region_means[i0,i1,'iiil'] + region_means[i0,i1,'iiki'] + region_means[i0,i1,'ijjj'] + region_means[i0,i1,'ijii']) +
        #     #     n_i[m] * (n_i[m]-1) * (n_i[m]-2) * (region_means[i0,i1,'ijil'] + region_means[i0,i1,'ijjl'] + region_means[i0,i1,'ijki'] + region_means[i0,i1,'ijkj'] + region_means[i0,i1,'iikl'] + region_means[i0,i1,'ijkk']) +
        #     #     n_i[m] * (n_i[m]-1) * (n_i[m]-2) * (n_i[m]-3) * region_means[i0,i1,'ijkl']
        #     # )
            
        #     # m = masks['iikk']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_k[m] * region_means[i0,i1,'iikk'] +
        #     #     n_i[m] * n_k[m] * (n_i[m]-1) * region_means[i0,i1,'ijkk'] +
        #     #     n_i[m] * n_k[m] * (n_k[m]-1) * region_means[i0,i1,'iikl'] +
        #     #     n_i[m] * n_k[m] * (n_i[m]-1) * (n_k[m]-1) * region_means[i0,i1,'ijkl']
        #     # )
            
        #     # m = masks['ijij']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * region_means[i0,i1,'ijij'] +
        #     #     n_i[m] * n_j[m] * (n_i[m]-1) * region_means[i0,i1,'ijkj'] +
        #     #     n_i[m] * n_j[m] * (n_j[m]-1) * region_means[i0,i1,'ijil'] +
        #     #     n_i[m] * n_j[m] * (n_i[m]-1) * (n_j[m]-1) * region_means[i0,i1,'ijkl']
        #     # )

        #     # m = masks['ijji']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * region_means[i0,i1,'ijji'] +
        #     #     n_i[m] * n_j[m] * (n_i[m]-1) * region_means[i0,i1,'ijki'] +
        #     #     n_i[m] * n_j[m] * (n_j[m]-1) * region_means[i0,i1,'ijjl'] +
        #     #     n_i[m] * n_j[m] * (n_i[m]-1) * (n_j[m]-1) * region_means[i0,i1,'ijkl']
        #     # )
            
        #     # m = masks['iiil']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_l[m] * region_means[i0,i1,'iiil'] +
        #     #     n_i[m] * n_l[m] * (n_i[m]-1) * (region_means[i0,i1,'ijjl'] + region_means[i0,i1,'ijil'] + region_means[i0,i1,'iikl']) +
        #     #     n_i[m] * n_l[m] * (n_i[m]-1) * (n_i[m]-2) * region_means[i0,i1,'ijkl']
        #     # )
            
        #     # m = masks['iiki']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_k[m] * region_means[i0,i1,'iiki'] +
        #     #     n_i[m] * n_k[m] * (n_i[m]-1) * (region_means[i0,i1,'ijki'] + region_means[i0,i1,'ijkj'] + region_means[i0,i1,'iikl']) +
        #     #     n_i[m] * n_k[m] * (n_i[m]-1) * (n_i[m]-2) * region_means[i0,i1,'ijkl']
        #     # )

        #     # m = masks['ijjj']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * region_means[i0,i1,'ijjj'] +
        #     #     n_i[m] * n_j[m] * (n_j[m]-1) * (region_means[i0,i1,'ijjl'] + region_means[i0,i1,'ijkj'] + region_means[i0,i1,'ijkk']) +
        #     #     n_i[m] * n_j[m] * (n_j[m]-1) * (n_j[m]-2) * region_means[i0,i1,'ijkl']
        #     # )

        #     # m = masks['ijii']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * region_means[i0,i1,'ijii'] +
        #     #     n_i[m] * n_j[m] * (n_i[m]-1) * (region_means[i0,i1,'ijki'] + region_means[i0,i1,'ijil'] + region_means[i0,i1,'ijkk']) +
        #     #     n_i[m] * n_j[m] * (n_i[m]-1) * (n_i[m]-2) * region_means[i0,i1,'ijkl']
        #     # )
            
        #     # m = masks['ijil']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * n_l[m] * region_means[i0,i1,'ijil'] +
        #     #     n_i[m] * n_j[m] * n_l[m] * (n_i[m]-1) * region_means[i0,i1,'ijkl']
        #     # )

        #     # m = masks['ijjl']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * n_l[m] * region_means[i0,i1,'ijjl'] +
        #     #     n_i[m] * n_j[m] * n_l[m] * (n_j[m]-1) * region_means[i0,i1,'ijkl']
        #     # )

        #     # m = masks['ijki']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * n_k[m] * region_means[i0,i1,'ijki'] +
        #     #     n_i[m] * n_j[m] * n_k[m] * (n_i[m]-1) * region_means[i0,i1,'ijkl']
        #     # )

        #     # m = masks['ijkj']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * n_k[m] * region_means[i0,i1,'ijkj'] +
        #     #     n_i[m] * n_j[m] * n_k[m] * (n_j[m]-1) * region_means[i0,i1,'ijkl']
        #     # )

        #     # m = masks['ijkk']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * n_k[m] * region_means[i0,i1,'ijkk'] +
        #     #     n_i[m] * n_j[m] * n_k[m] * (n_k[m]-1) * region_means[i0,i1,'ijkl']
        #     # )
            
        #     # m = masks['iikl']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_k[m] * n_l[m] * region_means[i0,i1,'iikl'] +
        #     #     n_i[m] * n_k[m] * n_l[m] * (n_i[m]-1) * region_means[i0,i1,'ijkl']
        #     # )
            
        #     # m = masks['ijkl']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * n_k[m] * n_l[m] * region_means[i0,i1,'ijkl']
        #     # )
            
        #     # Flatten the mean_inner array and expand each equation
        #     return mean_inner_product.reshape(n_elements**2, n_elements**2)

        # def mean_inner_product_1_by_2(repetitions,i0,i1):
        #     n_elements= len(repetitions)

        #     # Create the mean_inner_product array
        #     mean_inner_product = np.zeros((n_elements, n_elements, n_elements)).flatten()
            
        #     # Create arrays of indices for elements
        #     n_i = np.repeat(repetitions, n_elements**2).reshape(n_elements, n_elements, n_elements)
        #     n_j = n_i.copy().transpose(2, 0, 1).flatten()
        #     n_k = n_i.copy().transpose(1, 2, 0).flatten()
        #     n_i=n_i.flatten()

        #     # m = masks['iii']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * region_means[i0,i1,'iii'] +
        #     #     n_i[m] * (n_i[m]-1) * (region_means[i0,i1,'iik'] + region_means[i0,i1,'iji'] + region_means[i0,i1,'ijj']) +
        #     #     n_i[m] * (n_i[m]-1) * (n_i[m]-2) * (region_means[i0,i1,'ijk']) 
        #     # )
            
        #     # m = masks['ijj']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * region_means[i0,i1,'ijj'] +
        #     #     n_i[m] * n_j[m] * (n_j[m]-1) * region_means[i0,i1,'ijk']
        #     # )
            
        #     # m = masks['iji']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * region_means[i0,i1,'iji'] +
        #     #     n_i[m] * n_j[m] * (n_i[m]-1) * region_means[i0,i1,'ijk']
        #     # )

        #     # m = masks['iik']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_k[m] * region_means[i0,i1,'iik'] +
        #     #     n_i[m] * n_k[m] * (n_i[m]-1) * region_means[i0,i1,'ijk']
        #     # )
            
        #     # m = masks['ijk']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * n_k[m] * region_means[i0,i1,'ijk']
        #     # )
        #     # Flatten the mean_inner array and expand each equation
        #     return mean_inner_product.reshape(n_elements, n_elements**2)

        # def mean_inner_product_1_by_1(repetitions,i0,i1):
            
        #     n_elements= len(repetitions)

        #     # Create the mean_inner_product array
        #     mean_inner_product = np.zeros((n_elements, n_elements)).flatten()
            
        #     # Create arrays of indices for elements
        #     n_i = np.repeat(repetitions, n_elements).reshape(n_elements, n_elements)
        #     n_j = n_i.copy().transpose(1, 0).flatten()
        #     n_i=n_i.flatten()

        #     # Define masks for elements
            
        #     # m = masks['ii']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * region_means[i0,i1,'ii'] +
        #     #     n_i[m] * (n_i[m]-1) * region_means[i0,i1,'ij']
        #     # )
            
        #     # m = masks['ij']
        #     # mean_inner_product[m] = (
        #     #     n_i[m] * n_j[m] * region_means[i0,i1,'ij']
        #     # )
        
        #     return mean_inner_product.reshape(n_elements, n_elements)
        
        # mean_inner_product_2_by_2=self.numbify(mean_inner_product_2_by_2)
        # mean_inner_product_1_by_2=self.numbify(mean_inner_product_1_by_2)
        # mean_inner_product_1_by_1=self.numbify(mean_inner_product_1_by_1)
        
        # def build_mean_inner_product_matrix(repetitions,indicators1d,indicators2d):
        #     num_matrices1d = len(indicators1d)
        #     num_matrices2d = len(indicators2d)
        #     num_matrices = num_matrices1d + num_matrices2d
        #     n_elements=len(repetitions)
            
        #     # Compute the size of each block and the total size
        #     block_sizes = [n_elements for ind in indicators1d] + [n_elements**2 for ind in indicators2d]
        #     total_size = sum(block_sizes)
            
        #     # Create the resulting matrix filled with zeros
        #     R = np.zeros((total_size, total_size))
                
        #     # Compute the starting indices for each matrix
        #     #start_indices = np.cumsum([0] + block_sizes[:-1])
        #     start_indices=np.zeros(len(block_sizes),dtype=np.int64)
        #     start=0
        #     for i in range(1,len(block_sizes)):
        #         start=start+block_sizes[i-1]
        #         start_indices[i] = start
            
        #     for i in range(num_matrices):
        #         for j in range(i, num_matrices):  # Use symmetry, compute only half
        #             if i<num_matrices1d and j<num_matrices1d:
        #                 result_block = mean_inner_product_1_by_1(repetitions,i, j)
        #             elif i<num_matrices1d and j>=num_matrices1d:
        #                 result_block = mean_inner_product_1_by_2(repetitions,i,j)
        #             elif j<num_matrices1d and i>=num_matrices1d:
        #                 result_block = mean_inner_product_1_by_2(repetitions,j,i).T
        #             elif i>=num_matrices1d and j>=num_matrices1d:
        #                 result_block = mean_inner_product_2_by_2(repetitions,i,j)
                
        #             si, sj = start_indices[i], start_indices[j]
        #             ei, ej = si + result_block.shape[0], sj + result_block.shape[1]
        #             R[si:ei, sj:ej] = result_block
        #             if i != j:
        #                 R[sj:ej, si:ei] = result_block.T  # Leverage symmetry
                    
        #     return R
        
        # build_mean_inner_product_matrix=self.numbify(build_mean_inner_product_matrix)