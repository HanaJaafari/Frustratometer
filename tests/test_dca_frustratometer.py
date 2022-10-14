"""
Unit and regression test for the dca_frustratometer package.
"""

# Import package, test suite, and other packages as needed
import sys
import dca_frustratometer
import numpy as np


# def test_create_pfam_database():
#     alignments_path = dca_frustratometer.create_pfam_database(url='https://ftp.ebi.ac.uk/pub/databases/Pfam'
#                                                                   '/current_release/Pfam-A.dead.gz',
#                                                               name='test')
#     assert (alignments_path / 'Unknown.sto').exists() is False

def test_get_alignment_from_database():
    pass

def test_get_alignment_from_interpro():
    output = dca_frustratometer.download_alignment_from_interpro('PF09696')
    assert output.exists()
    output_text = output.read_text()
    assert "#=GF AC   PF09696" in output_text

def test_filter_alignment():
    pass

def create_potts_model_from_aligment():
    pass

def create_potts_model_from_pdb():
    pass

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
    model = dca_frustratometer.PottsModel.from_potts_model_file(potts_model_file, pdb_file, chain, distance_cutoff=4,
                                                                sequence_cutoff=0)
    e = model.native_energy()
    assert np.round(e, 4) == -92.7688


def test_scores():
    pdb_file = 'examples/data/1l63.pdb'
    chain = 'A'
    potts_model_file = 'examples/data/PottsModel1l63A.mat'
    model = dca_frustratometer.PottsModel.from_potts_model_file(potts_model_file, pdb_file, chain, distance_cutoff=4,
                                                                sequence_cutoff=0)
    assert np.round(model.scores()[30, 40], 5) == -0.03384


def test_compute_singleresidue_decoy_energy():
    aa_x = 5
    pos_x = 30
    distance_cutoff = 4
    sequence_cutoff = 0
    distance_matrix = dca_frustratometer.get_distance_matrix_from_pdb('examples/data/1l63.pdb', 'A')
    potts_model = dca_frustratometer.load_potts_model('examples/data/PottsModel1l63A.mat')
    seq = dca_frustratometer.get_protein_sequence_from_pdb('examples/data/1l63.pdb', 'A')
    mask = dca_frustratometer.compute_mask(distance_matrix, distance_cutoff, sequence_cutoff)
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq = [aa for aa in seq]
    seq[pos_x] = AA[aa_x]
    seq = ''.join(seq)
    test_energy = dca_frustratometer.compute_native_energy(seq, potts_model, mask)
    decoy_energy = dca_frustratometer.compute_decoy_energy(seq, potts_model, mask, 'singleresidue')
    assert (decoy_energy[pos_x, aa_x] - test_energy) ** 2 < 1E-16


def test_compute_mutational_decoy_energy():
    aa_x = 5
    pos_x = 30
    aa_y = 7
    pos_y = 69
    distance_cutoff = 4
    sequence_cutoff = 0
    distance_matrix = dca_frustratometer.get_distance_matrix_from_pdb('examples/data/1l63.pdb', 'A')
    potts_model = dca_frustratometer.load_potts_model('examples/data/PottsModel1l63A.mat')
    seq = dca_frustratometer.get_protein_sequence_from_pdb('examples/data/1l63.pdb', 'A')
    mask = dca_frustratometer.compute_mask(distance_matrix, distance_cutoff, sequence_cutoff)
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq = [aa for aa in seq]
    seq[pos_x] = AA[aa_x]
    seq[pos_y] = AA[aa_y]
    seq = ''.join(seq)
    test_energy = dca_frustratometer.compute_native_energy(seq, potts_model, mask)
    decoy_energy = dca_frustratometer.compute_decoy_energy(seq, potts_model, mask, 'mutational')
    assert (decoy_energy[pos_x, pos_y, aa_x, aa_y] - test_energy) ** 2 < 1E-16
