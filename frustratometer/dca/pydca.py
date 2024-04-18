import numpy as np
import pydca

def plmdca(filtered_alignment_file,
            sequence_type='protein',
            seqid=0.8, 
            lambda_h=1.0,
            lambda_J=20.0,
            num_threads=10,
            max_iterations=500,
            verbose=False,
            msa_file_format='fasta'):
    plmdca_inst = pydca.plmdca.PlmDCA(msa_file=filtered_alignment_file,
                                        biomolecule=sequence_type, 
                                        seqid = seqid,
                                        lambda_h=lambda_h,
                                        lambda_J = lambda_J,
                                        max_iterations = max_iterations,
                                        num_threads = num_threads, 
                                        verbose = verbose,
                                        msa_file_format=msa_file_format)
            
    potts_model = plmdca_inst.get_potts_model()
    # Move gaps to the beginning
    qq = [20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    potts_model['h'] = potts_model['h'][:,qq]
    potts_model['J'] = potts_model['J'][:,:,qq,:][:,:,:,qq]
    potts_model['L'] = plmdca_inst.num_sequences
    return potts_model

def mfdca(filtered_alignment_file):
    mfdca_inst = pydca.meanfield_dca.MeanFieldDCA(filtered_alignment_file,'protein')
    N = mfdca_inst.sequences_len
    q = mfdca_inst.num_site_states

    reg_fi = mfdca_inst.get_reg_single_site_freqs()
    reg_fij = mfdca_inst.get_reg_pair_site_freqs()

    corr_mat = mfdca_inst.construct_corr_mat(reg_fi, reg_fij)

    couplings = mfdca_inst.compute_couplings(corr_mat)
    fields = mfdca_inst.compute_fields(couplings)

    couplings=couplings.reshape(N,q-1,N,q-1)
    fields = np.array([a for a in fields.values()])

    fields = np.concatenate([np.zeros([N,1]),fields],axis=1)
    couplings = np.concatenate([np.zeros([N,1,N,q-1]),couplings],axis=1)
    couplings = np.concatenate([np.zeros([N,q,N,1]),couplings],axis=3)
    couplings = couplings.transpose(0,2,1,3)

    potts_model = {'h':fields,'J':couplings,'N':N,'q':q}
    return potts_model

__all__ = ['plmdca', 'mfdca']