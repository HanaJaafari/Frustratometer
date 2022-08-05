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
    e = dca_frustratometer.compute_native_energy(seq, potts_model, distance_matrix,
                                                 distance_cutoff=4, sequence_distance_cutoff=0)
    assert np.round(e, 4) == -92.7688

def test_OOP_compute_native_energy():
    pdb_file = 'examples/data/1l63.pdb'
    chain = 'A'
    potts_model_file = 'examples/data/PottsModel1l63A.mat'
    model = dca_frustratometer.PottsModel(pdb_file, chain, potts_model_file, distance_cutoff=4, sequence_cutoff=0)
    e = model.native_energy()
    assert np.round(e, 4) == -92.7688
def test_compute_singleresidue_frustration():
    pass


