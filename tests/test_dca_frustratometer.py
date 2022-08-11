"""
Unit and regression test for the dca_frustratometer package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import dca_frustratometer
import numpy as np


def test_dca_frustratometer_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "dca_frustratometer" in sys.modules

def test_functional_compute_native_energy():
    seq = dca_frustratometer.get_protein_sequence_from_pdb('examples/data/1l63.pdb', 'A')
    distance_matrix = dca_frustratometer.get_distance_matrix_from_pdb('examples/data/1l63.pdb', 'A')
    potts_model = dca_frustratometer.load_potts_model('examples/data/PottsModel1l63A.mat')
    mask = dca_frustratometer.compute_mask(distance_matrix, distance_cutoff=4, sequence_distance_cutoff=0)
    e = dca_frustratometer.compute_native_energy(seq, potts_model, mask)
    assert np.round(e, 4) == -92.7688

def test_OOP_compute_native_energy():
    pdb_file = 'examples/data/1l63.pdb'
    chain = 'A'
    potts_model_file = 'examples/data/PottsModel1l63A.mat'
    model = dca_frustratometer.PottsModel(pdb_file, chain, potts_model_file, distance_cutoff=4, sequence_cutoff=0)
    e = model.native_energy()
    assert np.round(e, 4) == -92.7688

def test_compute_mutational_decoy_energy():
    from scipy.spatial import distance as sdist
    aa_x = 12
    pos_x = 126
    aa_y = 18
    pos_y = 47
    distance_matrix = dca_frustratometer.get_distance_matrix_from_pdb('examples/data/1l63.pdb', 'A')
    potts_model = dca_frustratometer.load_potts_model('examples/data/PottsModel1l63A.mat')
    seq = dca_frustratometer.get_protein_sequence_from_pdb('examples/data/1l63.pdb', 'A')
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    distance_cutoff=4
    seq_index = np.array([AA.find(aa) for aa in seq])
    seq_len = len(seq_index)
    seq_distance = sdist.squareform(sdist.pdist(np.arange(seq_len)[:, np.newaxis]))
    test_seq = seq_index.copy()
    test_seq[pos_x] = aa_x
    test_seq[pos_y] = aa_y
    test_h = -potts_model['h'][range(seq_len), test_seq]
    test_J = -potts_model['J'][range(seq_len), :, test_seq, :][:, range(seq_len), test_seq]
    test_J_prime = test_J * (seq_distance > 0) * (distance_matrix <= distance_cutoff)
    test_energy = test_h.sum() + test_J_prime.sum() / 2
    mask = dca_frustratometer.compute_mask(distance_matrix,distance_cutoff,0)
    decoy_energy=dca_frustratometer.compute_mutational_decoy_energy(seq,potts_model,mask)
    assert (decoy_energy[pos_x,pos_y,aa_x,aa_y]-test_energy)**2 < 1E-16

def test_compute_singleresidue_frustration():
    pass


