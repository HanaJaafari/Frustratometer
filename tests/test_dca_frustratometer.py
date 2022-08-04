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

def test_compute_native_energy():
    seq = dca_frustratometer.get_protein_sequence_from_pdb('examples/data/1l63.pdb', 'A')
    distance_matrix = dca_frustratometer.get_distance_matrix_from_pdb('examples/data/1l63.pdb', 'A')
    e = dca_frustratometer.compute_native_energy(seq, 'examples/data/PottsModel1l63A.mat', distance_matrix,
                                                 distance_cutoff=4, sequence_distance_cutoff=0)
    assert np.round(e,4) == -92.7688

