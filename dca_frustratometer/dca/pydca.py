import pydca

def run(fasta_sequence,
        sequence_type='protein',
        seqid=0.8, 
        lambda_h=1.0,
        lambda_J=20.0,
        num_threads=10,
        max_iterations=500):
    plmdca_inst = pydca.plmdca.PlmDCA(fasta_sequence,
                                      sequence_type,
                                      seqid,
                                      lambda_h,
                                      lambda_J,
                                      num_threads,
                                      max_iterations)
    return plmdca_inst.get_potts_model()