import frustratometer
import logging
from frustratometer.optimization import build_mean_inner_product_matrix
print(dir)
print("Module Structure\n")
for submodule in dir(frustratometer):
    #print(dca_frustratometer.__dict__)
    if "__" not in submodule:
        m=frustratometer.__dict__[submodule]
        print(f'{submodule}: {" ".join(a for a in dir(m) if "__" not in a)}')

alphabet_awsem = ''.join(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])

def index_to_sequence(seq_index, alphabet):
    """Converts sequence index array back to sequence string."""
    return ''.join([alphabet[index] for index in seq_index])

def sequence_to_index(sequence, alphabet):
    """Converts sequence string to sequence index array."""
    return np.array([alphabet.find(aa) for aa in sequence])

def indices_to_sequences(indices, alphabet):
    """Converts sequence indices to sequence strings."""
    np.empty(len(indices), dtype='U1')
    return [index_to_sequence(seq_index, alphabet) for seq_index in indices]

def sequences_to_indices(sequences, alphabet):
    """Converts sequence strings to sequence indices."""
    np.empty((len(sequences),len(sequences[0])), dtype='int')
    return np.array([sequence_to_index(sequence, alphabet) for sequence in sequences])

def compute_AB(seq_index, indicators, len_alphabet=len(alphabet_awsem)):
    """Computes A B """
    
    aa_count = np.bincount(seq_index, minlength=len_alphabet)
    freq_i=np.array(aa_count)
    freq_ij=np.outer(freq_i,freq_i)
    alpha = np.diag(freq_i)
    beta = freq_ij.copy()
    np.fill_diagonal(beta, freq_i*(freq_i-1))

    phi_len=sum([len_alphabet**len(ind.shape) for ind in indicators])
    phi_mean = np.zeros(phi_len)
    offset=0
    for indicator in indicators:
        if len(indicator.shape) == 1:  # 1D indicator
            phi_mean[offset:offset+len_alphabet]=np.mean(indicator)*freq_i
            offset += len_alphabet
        elif len(indicator.shape) == 2:  # 2D indicator
            
            temp_indicator=indicator.copy()
            mean_diagonal_indicator = np.diag(temp_indicator).mean()
            np.fill_diagonal(temp_indicator, 0)
            mean_offdiagonal_indicator = temp_indicator.mean()
            
            phi_mean[offset:offset+len_alphabet**2]=alpha.ravel()*mean_diagonal_indicator + beta.ravel()*mean_offdiagonal_indicator
            offset += len_alphabet**2
    
    phi_native=phi(seq_index=seq_index,indicators=indicators,len_alphabet=len_alphabet)
    A = phi_mean-phi_native
    B = build_mean_inner_product_matrix(freq_i,indicators) - np.outer(phi_mean,phi_mean)
    return A,B

def phi(seq_index, indicators, len_alphabet=len(alphabet_awsem)):
    """ Sums the indicators according to the type determined by the sequence"""


    phi_len=sum([len_alphabet**len(ind.shape) for ind in indicators])
    seq_pairs = (np.array(np.meshgrid(seq_index, seq_index)) * np.array([1, len_alphabet])[:, None, None]).sum(axis=0).ravel()

    phi_sum=np.zeros(phi_len)
    offset=0
    for indicator in indicators:
        if len(indicator.shape) == 1:  # 1D indicator
            np.add.at(phi_sum, seq_index + offset, indicator)
            offset += len_alphabet
        elif len(indicator.shape) == 2:  # 2D indicator
            np.add.at(phi_sum, seq_pairs + offset, indicator.ravel())
            offset += len_alphabet ** 2
    return phi_sum

if __name__ == "__main__":
    import numpy as np
    pdb_file = frustratometer._path.parent/'tests'/'data'/'1r69.pdb'
    pdb_structure = frustratometer.Structure.full_pdb(str(pdb_file))
    self = frustratometer.AWSEM(pdb_structure,expose_indicator_functions=True)
    indicators=[self.burial_indicator[:,0],self.burial_indicator[:,1],self.burial_indicator[:,2], 
            self.direct_indicator[:,:,0,0], self.water_indicator[:,:,0,0], self.protein_indicator[:,:,0,0]]
    
    sequence1='SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRARNDCQEGHILKMFPSTWYV'
    sequence2='SISSAVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRARNDCQEGHILKMFPSTWYV' # Mutation
    sequence3='SISSVRKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRARNDCQEGHILKMFPSTWYV' # Swap

    s1=sequence_to_index(sequence1, alphabet_awsem)
    s2=sequence_to_index(sequence2, alphabet_awsem)
    s3=sequence_to_index(sequence2, alphabet_awsem)

    aa_count1 = np.bincount(s1, minlength=len(alphabet_awsem))
    aa_count2 = np.bincount(s2, minlength=len(alphabet_awsem))
    aa_count3 = np.bincount(s2, minlength=len(alphabet_awsem))

    print(aa_count1)
    print(aa_count2)
    print(aa_count3)

    A1,B1=compute_AB(s1, indicators, len(alphabet_awsem))
    A2,B2=compute_AB(s2, indicators, len(alphabet_awsem))
    A3,B3=compute_AB(s3, indicators, len(alphabet_awsem))

    DeltaE2_mutation=self.gamma.gamma_array @ B2 @ self.gamma.gamma_array
    DeltaE2_swap=self.gamma.gamma_array @ B2 @ self.gamma.gamma_array

    print(DeltaE2_mutation,DeltaE2_swap)

    
    #self.sequence

    # #compute_AB(self,sequence)
    # self=awsem
    # sequence=awsem.sequence
    # temp_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    # aa_to_index = {aa: index for index, aa in enumerate(temp_aa)}
    # seq_index = np.array([aa_to_index[aa] for aa in self.sequence])
    # seq_index2=np.meshgrid(seq_index,seq_index)
    # seq_index2=np.array(seq_index2).T
    # seq_index=np.ravel_multi_index(seq_index2.reshape(-1,2).T,(20,20))

    # aa_count=dict(a for a in np.array(np.unique(seq_index, return_counts=True)).T)
    # freq_i=np.array([(aa_count[aa] if aa in aa_count.keys() else 0) for aa in range(20)])
    # freq_ij=np.outer(freq_i,freq_i)
    # freq_ij2=np.outer(freq_ij.ravel(),freq_ij.ravel())

    # if self.burial_indicator is None:
    #     logging.error("Indicator were not saved. Initialize AWSEM function with `expose_indicator_functions=True` first.")
    
    # for indicator_type,indicator in enumerate([self.burial_indicator[:,0],self.burial_indicator[:,1],self.burial_indicator[:,2],
    #                                             self.direct_indicator.ravel(), self.water_indicator.ravel(), self.protein_indicator.ravel()]):
    #     if indicator_type<3:
    #         size=20
    #         f_i=freq_i
    #         f_ij=freq_ij
    #         native_sequence=seq_index
    #     else:
    #         size=400
    #         f_i=freq_ij.ravel()
    #         f_ij=freq_ij2
    #         native_sequence=seq_index2
    #     phi_native=np.zeros(size)
    #     phi_mean=np.zeros(size)
    #     np.add.at(phi_native, native_sequence, indicator)
    #     np.add.at(phi_mean, native_sequence, indicator.mean())

    #     phi_outer_mean = np.outer(phi_mean, phi_mean)

    #     indicator_outer = np.outer(indicator,indicator)
    #     #f_ij=np.outer(f_i,f_i)
    #     mean_diagonal=indicator_outer.diagonal().sum()/len(indicator)
    #     mean_offdiagonal=(indicator_outer.sum()-indicator_outer.diagonal().sum())/len(indicator)/(len(indicator)-1)
    #     inner_product_diagonal=((aa_counts>0)*mean_diagonal+(aa_counts-1)*((aa_counts-1)>0)*mean_offdiagonal)*aa_counts
    #     phi_inner_mean=f_ij*mean_offdiagonal
    #     np.fill_diagonal(phi_inner_mean,inner_product_diagonal)

    #     Bij = phi_inner_mean-phi_outer_mean
    #     Ai = phi_mean-phi_native
    #     Ai,Bij
