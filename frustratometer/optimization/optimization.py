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
from frustratometer.optimization.inner_product import build_mean_inner_product_matrix
from frustratometer.optimization.inner_product import diff_mean_inner_product_matrix
import itertools

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def index_to_sequence(seq_index, alphabet):
    """Converts sequence index array back to a sequence string."""
    seq = ''.join([alphabet[index] for index in seq_index])
    return seq

def sequence_to_index(sequence, alphabet):
    """Converts sequence string to sequence index array."""
    index = np.array([alphabet.find(aa) for aa in sequence])
    if np.any(index == -1):
        raise ValueError("Invalid amino acid in sequence")
    return index

@numba.njit
def random_seed(seed):
    """Sets the random seed for the numpy random number generator.
    This function is needed to set the seed in a numba compatible way."""
    np.random.seed(seed)

class Zero(EnergyTerm):
    """ Zero energy term for testing purposes. """
    @staticmethod
    def compute_energy(seq_index:np.ndarray):
        return 0.
    
    @staticmethod
    def compute_denergy_mutation(seq_index:np.ndarray, pos:int, aa):
        return 0.
    
    @staticmethod
    def compute_denergy_swap(seq_index:np.ndarray, pos1:int, pos2:int):
        return 0.

class Heterogeneity(EnergyTerm):
    """ Heterogeneity energy term.
        This term calculates the heterogeneity of a sequence using the Shannon entropy.
        The heterogeneity is calculated as the ratio of the number of possible sequences with the same amino acid composition to the total number of sequences.
        The energy is the negative logarithm of the heterogeneity.
     """
    def __init__(self, exact=False, alphabet=_AA, use_numba=True):
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

class AwsemEnergy(EnergyTerm):
    """ AWSEM energy term.
        This term calculates the energy of a sequence using the AWSEM model.
        The energy is calculated from the fields and couplings of the Potts model.
        """
    def __init__(self, model:Frustratometer, alphabet=_AA, use_numba=True):
        self.use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.model_h = model.potts_model['h']
        self.model_J = model.potts_model['J']
        self.mask = model.mask

        if alphabet!=_AA:
            self.reindex_dca=[_AA.index(aa) for aa in alphabet]
            self.model_h=self.model_h[:,self.reindex_dca]
            self.model_J=self.model_J[:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.initialize_functions()
    
    def initialize_functions(self):
        mask=self.mask.copy()
        model_h=self.model_h.copy()
        model_J=self.model_J.copy()
        
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
            j_correction += model_J[pos1, pos2, aa1, seq_index[pos2]] * mask[pos1, pos2]  # Correction for incorrect addition in the for loop
            j_correction += model_J[pos1, pos2, seq_index[pos1], aa2] * mask[pos1, pos2]  # Correction for incorrect addition in the for loop
            j_correction -= model_J[pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
            energy_difference += j_correction
            return energy_difference
        
        self.compute_energy=compute_energy
        self.compute_denergy_mutation=compute_denergy_mutation
        self.compute_denergy_swap=compute_denergy_swap

    def regression_test(self):
        expected_energy=self.model.native_energy()
        seq_index=sequence_to_index(self.model.sequence,alphabet=self.alphabet)
        energy=self.compute_energy(seq_index)
        assert np.isclose(energy,expected_energy), f"Expected energy {expected_energy} but got {energy}"

class AwsemEnergyAverage(EnergyTerm):   
    def __init__(self, model:Frustratometer, use_numba=True, alphabet=_AA):
        self.use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.reindex_dca=[_AA.index(aa) for aa in alphabet]
        
        assert "indicators" in model.__dict__.keys(), "Indicator functions were not exposed. Initialize AWSEM function with `expose_indicator_functions=True` first."
        self.indicators = model.indicators
        self.alphabet_size=len(alphabet)
        self.model=model
        self.model_h=model.potts_model['h'][:,self.reindex_dca]
        self.model_J= model.potts_model['J'][:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.mask = model.mask
        self.indicators1D=np.array([ind for ind in self.indicators if len(ind.shape)==1])
        self.indicators2D=np.array([ind for ind in self.indicators if len(ind.shape)==2])
        #TODO: Fix the gamma matrix to account for elecrostatics
        self.gamma = np.concatenate([(a[self.reindex_dca].ravel() if len(a.shape)==1 else a[self.reindex_dca][:,self.reindex_dca].ravel()) for a in model.gamma_array])
        
        self.initialize_functions()
    
    def initialize_functions(self):
        indicators1D=self.indicators1D
        indicators2D=self.indicators2D
        len_alphabet=self.alphabet_size
        gamma=self.gamma
        phi_len= indicators1D.shape[0]*len_alphabet + indicators2D.shape[0]*len_alphabet**2
      
        def compute_energy(seq_index):
            aa_count = np.bincount(seq_index, minlength=len_alphabet)
            freq_i=aa_count
            freq_ij=np.outer(freq_i,freq_i)
            alpha = np.diag(freq_i)
            beta = freq_ij.copy()
            np.fill_diagonal(beta, freq_i*(freq_i-1))
            
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
            
            return gamma @ phi_mean
        
        compute_energy_numba=self.numbify(compute_energy, cache=True)
        
        def denergy_mutation(seq_index, pos, aa):
            seq_index_new = seq_index.copy()
            seq_index_new[pos] = aa
            return compute_energy_numba(seq_index_new) - compute_energy_numba(seq_index)
        
        self.compute_energy = compute_energy
        self.compute_denergy_mutation = denergy_mutation

class AwsemEnergyVariance(EnergyTerm):   
    def __init__(self, model:Frustratometer, use_numba=True, alphabet=_AA):
        self.use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.reindex_dca=[_AA.index(aa) for aa in alphabet]
        
        assert "indicators" in model.__dict__.keys(), "Indicator functions were not exposed. Initialize AWSEM function with `expose_indicator_functions=True` first."
        self.indicators = model.indicators
        self.alphabet_size=len(alphabet)
        self.model=model
        self.model_h = model.potts_model['h'][:,self.reindex_dca]
        self.model_J = model.potts_model['J'][:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.mask = model.mask
        self.indicators1D=np.array([ind for ind in self.indicators if len(ind.shape)==1])
        self.indicators2D=np.array([ind for ind in self.indicators if len(ind.shape)==2])
        #TODO: Fix the gamma matrix to account for elecrostatics
        self.gamma = np.concatenate([(a[self.reindex_dca].ravel() if len(a.shape)==1 else a[self.reindex_dca][:,self.reindex_dca].ravel()) for a in model.gamma_array])
        
        self.initialize_functions()
    
    def initialize_functions(self):
        indicators1D=self.indicators1D
        indicators2D=self.indicators2D
        len_alphabet=self.alphabet_size
        gamma=self.gamma
        phi_len= indicators1D.shape[0]*len_alphabet + indicators2D.shape[0]*len_alphabet**2
        
        region_means=compute_all_region_means(indicators1D,indicators2D)
        
        def compute_energy(seq_index):
            aa_count = np.bincount(seq_index, minlength=len_alphabet)
            freq_i=aa_count
            freq_ij=np.outer(freq_i,freq_i)
            np.fill_diagonal(freq_ij,freq_i*(freq_i-1))
            
            phi_mean = np.zeros(phi_len)
            offset=0
            for indicator in indicators1D:
                phi_mean[offset:offset+len_alphabet]=np.mean(indicator)*freq_i
                offset += len_alphabet
            for indicator in indicators2D:
                # Calculate the off-diagonal mean
                sum=0
                count=0
                for i in range(len(indicator)):
                    for j in range(len(indicator)):
                        if i!=j:
                            sum+=indicator[i,j]
                            count+=1
                mean_offdiagonal_indicator=sum/count
                
                
                phi_mean[offset:offset+len_alphabet**2]=freq_ij.ravel()*mean_offdiagonal_indicator
                offset += len_alphabet**2
            
            B = build_mean_inner_product_matrix(freq_i.copy(),indicators1D.copy(),indicators2D.copy(), region_means) - np.outer(phi_mean,phi_mean)
            return gamma @ B @ gamma
        
        compute_energy_numba=self.numbify(compute_energy, cache=True)
        
        def denergy_mutation(seq_index, pos, aa):
            seq_index_new = seq_index.copy()
            seq_index_new[pos] = aa
            return compute_energy_numba(seq_index_new) - compute_energy_numba(seq_index)
        
        self.compute_energy = compute_energy
        self.compute_denergy_mutation = denergy_mutation

        awsem_energy = AwsemEnergy(use_numba=self.use_numba, model=self.model, alphabet=self.alphabet).energy_function

        def compute_energy_sample(seq_index,n_decoys=100000):
            """ Function to compute the variance of the energy of permutations of a sequence using random shuffling.
                This function is much faster than compute_energy_permutation but is an approximation"""
            energies=np.zeros(n_decoys)
            shuffled_index=seq_index.copy()
            for i in numba.prange(n_decoys):
                energies[i]=awsem_energy(shuffled_index[np.random.permutation(len(shuffled_index))])
            return np.var(energies)

        def compute_energy_permutation(seq_index):
            """ Function to compute the variance of the energy of all permutations of a sequence 
                Caution: This function is very slow for normal sequences """
            from itertools import permutations
            decoy_sequences = np.array(list(permutations(seq_index)))
            energies=np.zeros(len(decoy_sequences))
            for i in numba.prange(len(decoy_sequences)):
                energies[i]=awsem_energy(decoy_sequences[i])
            return np.var(energies)
        
        self.compute_energy_sample=self.numbify(compute_energy_sample,parallel=True)
        self.compute_energy_permutation=compute_energy_permutation

    def regression_test(self, seq_index):
        expected_energy=self.compute_energy_permutation(seq_index)
        energy=self.compute_energy(seq_index)
        assert np.isclose(energy,expected_energy), f"Expected energy {expected_energy} but got {energy}"

class MonteCarlo:
    def __init__(self, sequence: str, energy: EnergyTerm, alphabet:str=_AA, use_numba:bool=True, evaluation_energies:list=[]):
        self.seq_len=len(sequence)
        self.seq_index=sequence_to_index(sequence,alphabet)
        self.energy = energy
        self.alphabet = alphabet
        self.use_numba = use_numba
        self.evaluation_energies = evaluation_energies
        self.initialize_functions()

    def generate_random_sequences(self,n):
        """ Generates n random sequences of the same length as the input sequence. """
        return np.random.randint(0, len(self.alphabet), size=(n,self.seq_len))
        
    @property
    def numbify(self):
        """ Returns the numba decorator if use_numba is True, otherwise returns a dummy decorator. """
        if self.use_numba:
            return numba.njit
        else:
            return self.dummy_decorator

    @staticmethod        
    def dummy_decorator(func, *args, **kwargs):
        """ Dummy decorator that returns the function unchanged. """
        return func

    def initialize_functions(self):
        """ Initializes the Monte Carlo functions for numba. """
        alphabet=self.alphabet
        alphabet_size=len(alphabet)
        sequence_size = self.seq_len
        energy=self.energy.energy_function
        energy_functions = (e.energy_function for e in self.evaluation_energies)
        mutation_energy=self.energy.denergy_mutation_function
        swap_energy=self.energy.denergy_swap_function

        def sequence_swap(seq_index):
            seq_index_new = seq_index[:]
            n=len(seq_index_new)
            res1 = np.random.randint(0,n)
            res2 = np.random.randint(0,n-1)
            res2 += (res2 >= res1)
            energy_difference = swap_energy(seq_index, res1, res2)
            seq_index_new[res1], seq_index_new[res2] = seq_index[res2], seq_index[res1]
            return seq_index_new, energy_difference
        
        sequence_swap=self.numbify(sequence_swap)

        def sequence_mutation(seq_index):
            seq_index_new = seq_index[:]
            r = np.random.randint(0, alphabet_size*sequence_size) # Select a random index
            res = r // alphabet_size
            aa_new = r % alphabet_size
            seq_index_new[res] = aa_new
            energy_difference = mutation_energy(seq_index, res, aa_new)
            return seq_index_new, energy_difference
        
        sequence_mutation=self.numbify(sequence_mutation)

        def montecarlo_steps(temperature, seq_index, n_steps = 1000, kb = 0.008314) -> np.array:
            for _ in range(n_steps):
                new_sequence, energy_difference = sequence_swap(seq_index) if np.random.random() > 0.5 else sequence_mutation(seq_index)
                exponent= (-energy_difference) / (kb * temperature + 1E-10)
                acceptance_probability = np.exp(min(0, exponent)) 
                if np.random.random() < acceptance_probability:
                    seq_index = new_sequence
            return seq_index
        
        montecarlo_steps=self.numbify(montecarlo_steps)

        def replica_exchange(energies, temperatures, kb=0.008314, exchange_id=0):
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

                if np.random.random() <= prob:
                    order[i]=i+1
                    order[i+1]=i
            return order
        
        replica_exchange=self.numbify(replica_exchange)

        def parallel_montecarlo_step(seq_indices, temperatures, n_steps_per_cycle):
            n_replicas = len(temperatures)
            energies = np.zeros(n_replicas)

            for i in numba.prange(n_replicas):
                temp_seq_index = seq_indices[i]
                seq_indices[i] = montecarlo_steps(temperatures[i], seq_index=temp_seq_index, n_steps=n_steps_per_cycle)
                energy = energy(seq_indices[i])
                # Compute energy for the new sequence
                energies[i] = energy

            return seq_indices, energies
        parallel_montecarlo_step=self.numbify(parallel_montecarlo_step, parallel=True)

        def parallel_tempering_steps(seq_indices, temperatures, n_steps, n_steps_per_cycle):
            for s in range(n_steps//n_steps_per_cycle):
                seq_indices, total_energies = parallel_montecarlo_step(seq_indices, temperatures, n_steps_per_cycle)

                # Yield data every 10 exchanges
                if s % 10 == 9:
                    yield s, seq_indices, total_energies

                # Perform replica exchanges
                order = replica_exchange(total_energies, temperatures, exchange_id=s)
                seq_indices = seq_indices[order]

        parallel_tempering_steps=self.numbify(parallel_tempering_steps)

        self.sequence_swap=sequence_swap
        self.sequence_mutation=sequence_mutation
        self.montecarlo_steps=montecarlo_steps
        self.replica_exchange=replica_exchange
        self.parallel_montecarlo_step=parallel_montecarlo_step
        self.parallel_tempering_steps=parallel_tempering_steps
            

    def parallel_tempering(self, seq_indices, temperatures, n_steps, n_steps_per_cycle, filename="parallel_tempering_results.csv", alphabet=_AA):
        columns=['Step', 'Temperature', 'Sequence', 'Energy', 'Heterogeneity', 'Total Energy']
        df_headers = pd.DataFrame(columns=columns)
        df_headers.to_csv(filename, index=False)
        print(*columns, sep='\t')

        # Run the simulation and append data periodically
        for s, updated_seq_indices, total_energy in self.parallel_tempering_steps(seq_indices, temperatures, n_steps, n_steps_per_cycle):
            # Prepare data for this chunk
            data_chunk = []
            for i, temp in enumerate(temperatures):
                sequence_str = index_to_sequence(updated_seq_indices[i],alphabet=alphabet)  # Convert sequence index back to string
                data_chunk.append({'Step': (s+1) * n_steps_per_cycle, 'Temperature': temp, 'Sequence': sequence_str, 'Total Energy': total_energy[i]})
            
            # Convert the chunk to a DataFrame and append it to the CSV
            df_chunk = pd.DataFrame(data_chunk)
            print(*df_chunk.iloc[-1].values, sep='\t')
            df_chunk.to_csv(filename, mode='a', header=False, index=False)


    def annealing(self,temp_max=500, temp_min=0, n_steps=1E8):
        simulation_data = []
        n_steps_per_cycle=n_steps//(temp_max-temp_min)
        for temp in range(temp_max, temp_min, -1):
            seq_index= self.montecarlo_steps(temp, seq_index, n_steps=n_steps_per_cycle)
            energy = energy.energy(seq_index)
            simulation_data.append({'Temperature': temp, 'Sequence': index_to_sequence(seq_index), 'Energy': energy})
            print(temp, index_to_sequence(seq_index, self.alphabet), energy)
        simulation_df = pd.DataFrame(simulation_data)
        simulation_df.to_csv("mcso_simulation_results.csv", index=False)

    def benchmark_montecarlo_steps(self, n_repeats=10, n_steps=20000):
        import time
        times = []

        #Adds one step for numba compilation time
        seq_index = np.random.randint(1, len(self.alphabet), size=self.seq_len)
        self.montecarlo_steps(temperature=500, seq_index=self.generate_random_sequences(1)[0], n_steps=1)

        for _ in range(n_repeats):  # Run benchmark 10 times
            # Generate a new random sequence for each run
            seq_index = np.random.randint(1, len(self.alphabet), size=self.seq_len)
            start_time = time.time()
            
            self.montecarlo_steps(temperature=500, seq_index=seq_index, n_steps=n_steps)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        average_time_per_step_s = sum(times) / len(times) / n_steps
        std_time_per_step_us = np.std(times) / len(times) / n_steps * 1000000
        average_time_per_step_us = average_time_per_step_s * 1000000
        
        steps_per_hour = 3600 / average_time_per_step_s
        minutes_needed = 1E10 / steps_per_hour / 8 * 60  # 8 processes in parallel

        print(f"Time needed to explore 10^10 sequences with 8 process in parallel: {minutes_needed:.2e} minutes")
        print(f"Number of sequences explored per hour: {steps_per_hour:.2e}")
        print(f"Average execution time per step: {average_time_per_step_us:.5f} +- {3*std_time_per_step_us:.5f} microseconds")


if __name__ == '__main__':
    
    native_pdb = "tests/data/1bfz.pdb"
    # native_pdb = "frustratometer/optimization/10.3_model_LinkerBack_partialEGFR.pdb"
    structure_bound = Structure.full_pdb(native_pdb, chain=None)
    structure_free = Structure.full_pdb(native_pdb, "A")
    model_bound = AWSEM(structure_bound, distance_cutoff_contact=10, min_sequence_separation_contact=2, expose_indicator_functions=True)
    model_free = AWSEM(structure_free, distance_cutoff_contact=10, min_sequence_separation_contact=2, expose_indicator_functions=True)
    reduced_alphabet = 'ADEFHIKLMNQRSTVWY'

    energy_bound = AwsemEnergy(model_bound, reduced_alphabet)
    energy_unbound = AwsemEnergy(model_free, reduced_alphabet)
    energy_variance = AwsemEnergyVariance(model_free, reduced_alphabet)
    heterogeneity = Heterogeneity(exact=False, use_numba=True)

    Ep = 10
    Ev = 10
    energy= (energy_bound - energy_unbound) + 10 * energy_variance + 10 * heterogeneity

    for key,value in {"energy_bound": energy_bound, "energy_unbound": energy_unbound, "heterogeneity": heterogeneity}.items():
        print (f"Energy term: {key}")
        monte_carlo = MonteCarlo(sequence = structure_free.sequence, energy=value, alphabet=reduced_alphabet, evaluation_energies=[energy_bound, energy_unbound, energy_variance, heterogeneity])
        monte_carlo.benchmark_montecarlo_steps(n_repeats=3,n_steps=200)
    #monte_carlo.annealing()
    #print(monte_carlo.sequences)




    
    
    # native_pdb = "tests/data/1bfz.pdb"
    # structure = Structure.full_pdb(native_pdb, "A")
    # #reduced_alphabet = 'ADEFGHIKLMNQRSTVWY'
    # reduced_alphabet = _AA
    
    # model = AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2, expose_indicator_functions=True, k_electrostatics=0.0)
    # reindex_dca=[_AA.index(aa) for aa in reduced_alphabet]
    # model_seq_index=sequence_to_index(model.sequence,alphabet=reduced_alphabet)
    # seq_indices = np.random.randint(0, len(reduced_alphabet), size=(1,len(structure.sequence)))
    
    # # Tests
    # for exact,use_numba in [(True,False),(False,False),(True,True),(False,True)]:
    #     het=Heterogeneity(exact=exact,use_numba=use_numba)
    #     for i in range(1):
    #         het.test(seq_indices[i])
    
    # for use_numba in [False, True]:
    #     awsem_energy = AwsemEnergy(use_numba=use_numba, model=model, alphabet=reduced_alphabet)
    #     for i in range(1):
    #         awsem_energy.test(seq_indices[i])
    #     awsem_energy.regression_test()

    # for use_numba in [False, True]:
    #     awsem_de2 = AwsemEnergyVariance(use_numba=use_numba, model=model, alphabet=reduced_alphabet)
    #     for i in range(1):
    #         awsem_de2.test(seq_indices[0])

    # seq_index=sequence_to_index(model.sequence,alphabet=_AA)
    
    # def compute_energy_variance_sample(seq_index,n_decoys=10000):
    #     energies=[]
    #     shuffled_index=seq_index.copy()
    #     for i in range(n_decoys):
    #         np.random.shuffle(shuffled_index)
    #         energies.append(model.native_energy(index_to_sequence(shuffled_index,alphabet=_AA)))
    #         #if i%(n_decoys//100)==0:
    #             #energies_array=np.array(energies)
    #             #print(i,energies_array.mean(),energies_array.var())
    #     #Split the energies into 10 groups and compute the variance of each group to get an error estimate
    #     energies_array=np.array(energies)
    #     energies_array=energies_array.reshape(10,-1)
    #     energy_variances=np.var(energies_array,axis=1)
    #     mean_variance=energy_variances.mean()
    #     error_variance=energy_variances.std()
    #     print(f"Decoy Variance: {mean_variance} +/- {3*error_variance}") #3 sigma error
    #     print(f"Expected variance: {awsem_de2.compute_energy(seq_index)}")
    #     return np.var(energies), awsem_de2.compute_energy(seq_index)
    
    # print(compute_energy_variance_sample(seq_index))

    # def compute_energy_variance_permutation(seq_index):
    #     from itertools import permutations
    #     decoy_sequences = np.array(list(permutations(seq_index)))
    #     energies=[]
    #     for seq in decoy_sequences:
    #         energies.append(model.native_energy(index_to_sequence(seq,alphabet=_AA)))

    #     print(f"Decoy Variance: {np.var(energies)}") # Exact variance
    #     print(f"Expected variance: {awsem_de2.compute_energy(seq_index)}")
    #     return np.var(energies), awsem_de2.compute_energy(seq_index)
    
    # print(compute_energy_variance_permutation(seq_index))

    # from itertools import permutations
    # decoy_sequences = np.array(list(permutations(seq_index)))
    # indicators1D=np.array(model.indicators[:3])
    # indicators2D=np.array(model.indicators[3:])
    # indicator_arrays=[]
    # energies=[]
    # for decoy_index in decoy_sequences:
    #     ind1D=np.zeros((len(indicators1D),21))
    #     for i in range(len(ind1D)):
    #         ind1D[i] = np.bincount(decoy_index, weights=indicators1D[i], minlength=21)

    #     decoy_index2D=decoy_index[np.newaxis,:]*21+decoy_index[:,np.newaxis]
    #     ind2D=np.zeros((len(indicators2D),21*21))
    #     for i in range(len(ind2D)):
    #         ind2D[i] =np.bincount(decoy_index2D.ravel(), weights=indicators2D[i].ravel(), minlength=21*21)

    #     indicator_array = np.concatenate([ind1D.ravel(),ind2D.ravel()])
    #     gamma_array = np.concatenate([a.ravel() for a in model.gamma_array])

    #     energy_i = gamma_array @ indicator_array
    #     assert np.isclose(model.native_energy(index_to_sequence(decoy_index,alphabet=_AA)),energy_i), f"Expected energy {model.native_energy(index_to_sequence(decoy_index,alphabet=_AA))} but got {energy_i}"
    #     energies.append(energy_i)
    #     indicator_arrays.append(indicator_array)

    # indicator_arrays = np.array(indicator_arrays)
    # energies = np.array(energies)
    # assert np.isclose(gamma_array@indicator_arrays.mean(axis=0),energies.mean()), f"Expected mean energy {gamma_array@indicator_arrays.mean(axis=0)} but got {np.mean(energies)}"

    # # I will code something like this using numpy einsums:
    # # np.array([[np.outer(indicator_arrays[:,i],indicator_arrays[:,j]).mean() - indicator_arrays[:,i].mean()*indicator_arrays[:,i].mean() for i in range(indicator_arrays.shape[1])] for j in range(indicator_arrays.shape[1])])
    # outer_product = np.einsum('ij,ik->ijk', indicator_arrays, indicator_arrays)
    # mean_outer_product = outer_product.mean(axis=0)
    # mean_outer_product -= np.outer(indicator_arrays.mean(axis=0), indicator_arrays.mean(axis=0))
    # assert np.allclose(gamma_array @ mean_outer_product @ gamma_array, energies.var()), "Covariance matrix is not correct"

    # # Indicator tests    
    # indicators1D=np.array(model.indicators[0:3])
    # indicators2D=np.array(model.indicators[3:])
    # gamma=model.gamma_array
    # true_indicator1D=np.array([indicators1D[:,model_seq_index==i].sum(axis=1) for i in range(21)]).T
    # true_indicator2D=np.array([indicators2D[:,model_seq_index==i][:,:, model_seq_index==j].sum(axis=(1,2)) for i in range(21) for j in range(21)]).reshape(21,21,3).T
    # true_indicator=np.concatenate([true_indicator1D.ravel(),true_indicator2D.ravel()])
    # burial_gamma=np.concatenate(model.gamma_array[:3])
    # burial_energy_predicted = (burial_gamma * np.concatenate(true_indicator1D)).sum()
    # burial_energy_expected = -model.potts_model['h'][range(len(model_seq_index)), model_seq_index].sum()
    # assert np.isclose(burial_energy_predicted,burial_energy_expected), f"Expected energy {burial_energy_expected} but got {burial_energy_predicted}"
    # contact_gamma=np.concatenate([a.ravel() for a in model.gamma_array[3:]])
    # contact_energy_predicted = (contact_gamma * np.concatenate([a.ravel() for a in true_indicator2D])).sum()
    # contact_energy_expected = model.couplings_energy()
    # assert np.isclose(contact_energy_predicted,contact_energy_expected), f"Expected energy {contact_energy_expected} but got {contact_energy_predicted}"


    
    # Combination test
    #energy = awsem_energy + 10 * het + 20 * awsem_de2

    # Monte Carlo old code
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