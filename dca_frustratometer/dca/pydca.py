import numpy as np

try:
    import pydca
except ImportError as e:
    print("Error: Could not import pydca. Please make sure it is installed and available in your system.",e)
    def plmdca():
        print("Function plmdca cannot be called as the module pydca is not available.")

    def mfdca():
        print("Function mfdca cannot be called as the module pydca is not available.")
else:
    def plmdca(filtered_alignment_file,
            sequence_type='protein',
            seqid=0.8, 
            lambda_h=1.0,
            lambda_J=20.0,
            num_threads=10,
            max_iterations=500):
        plmdca_inst = pydca.plmdca.PlmDCA(filtered_alignment_file,
                                        sequence_type,
                                        seqid,
                                        lambda_h,
                                        lambda_J,
                                        num_threads,
                                        max_iterations)
        potts_model = plmdca_inst.get_potts_model()
        # Move gaps to the beginning
        qq = [20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        potts_model['h'] =  potts_model['h'][:,qq]
        potts_model['J'] =  potts_model['J'][:,:,qq,:][:,:,:,qq]
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