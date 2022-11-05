import pydca

def run(filtered_alignment_file,
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
    return plmdca_inst.get_potts_model()