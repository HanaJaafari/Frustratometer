"""
Unit and regression test for the dca_frustratometer package.
"""

# Import package, test suite, and other packages as needed
import sys
import dca_frustratometer
import numpy as np
from pathlib import Path
import tempfile
import pytest
from dca_frustratometer.utils import _path
import Bio.AlignIO

data_path = dca_frustratometer.utils.create_directory(_path/'..'/'tests'/'data')
#scratch_path = dca_frustratometer.utils.create_directory(_path/'..'/'tests'/'scratch')

def test_dca_frustratometer_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "dca_frustratometer" in sys.modules

def test_download_pfam_database():
    """ This test downloads a small database from pfam and splits the files in a folder"""
    name='pfam_dead'
    url='https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.dead.gz'
    with tempfile.TemporaryDirectory() as scratch_path:
        alignments_path = dca_frustratometer.pfam.download_database(scratch_path, url=url, name=name)
        print(alignments_path)
        assert (alignments_path / 'Unknown.sto').exists() is False
        assert (alignments_path / 'PF00065.sto').exists() is True


def test_get_alignment_from_database():
    alignment_file = dca_frustratometer.pfam.get_alignment('PF17182', data_path/'pfam_database')
    assert alignment_file.exists()
    alignment_text = alignment_file.read_text()
    assert "#=GF AC   PF17182" in alignment_text

    alignment_file = dca_frustratometer.pfam.get_alignment('PF09696', data_path/'pfam_database')
    assert alignment_file.exists()
    alignment_text = alignment_file.read_text()
    assert "#=GF AC   PF09696" in alignment_text


def test_download_pfam_alignment():
    with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_interpro.sto') as output_handle:
        output = Path(output_handle.name)
        assert output.exists()
        output_text = output.read_text()
        assert output_text==""
        output = dca_frustratometer.pfam.alignment('PF09696',output)
        assert output.exists()
        output_text = output.read_text()
        assert "#=GF AC   PF09696" in output_text
    assert not output.exists()


def test_filter_alignment_memory():
    alignment_file = data_path/'pfam_database'/'PF09696.12.sto'
    expected_filtered_sequence='-IQTPSGLALLELQGTINLPEDAVDSDGKAT-------------KSIPVGRIDFPDYHPDTQSTAWMKRVYLYVGPHQRLTGEVKKLPKAIAIVRKKDGASNG-----------------------------------------'
    with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_filtered_memory.sto') as output_handle:
        output_file = Path(output_handle.name)
        filtered_file=dca_frustratometer.filter.filter_alignment(alignment_file, output_file)
        assert filtered_file.exists()
        unfiltered_alignment = Bio.AlignIO.read(alignment_file, 'stockholm')
        filtered_alignment = Bio.AlignIO.read(filtered_file, 'fasta')
        assert len(unfiltered_alignment)==len(filtered_alignment)
        assert unfiltered_alignment.get_alignment_length() > filtered_alignment.get_alignment_length()
        assert filtered_alignment.get_alignment_length() == len(expected_filtered_sequence)
        print(filtered_alignment[0].seq)
        assert filtered_alignment[0].seq == expected_filtered_sequence
        

def test_filter_alignment_lowmem():
    alignment_file = data_path/'pfam_database'/'PF17182.6.sto'
    expected_filtered_sequence='QHDSMFTINSDYDAYLLDFPLLGDDFLLYLARMELRCRFKRTERVLQSGLCVSGQTISGARSRLHHLLVNKTQIIVNIGSVDIMRGRPIVQIQHDFRQLVK'\
                               'DMHNRGLVPILTTLAPLANYCHDKAMCDKVVKFNQFIWKECASYLKVIDIHSCLVNENGVVRFDCFQYSSRNVTGSKESYVFWNKIGRQRVLQMIEASLE'
    with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_filtered_disk.sto') as output_handle:
        output_file = Path(output_handle.name)
        filtered_file=dca_frustratometer.filter.filter_alignment_lowmem(alignment_file,output_file)
        assert filtered_file.exists()
        unfiltered_alignment = Bio.AlignIO.read(alignment_file, 'stockholm')
        filtered_alignment = Bio.AlignIO.read(filtered_file, 'fasta')
        assert len(unfiltered_alignment)==len(filtered_alignment)
        assert unfiltered_alignment.get_alignment_length() > filtered_alignment.get_alignment_length()
        assert filtered_alignment.get_alignment_length() == len(expected_filtered_sequence)
        assert filtered_alignment[0].seq == expected_filtered_sequence
   
def test_generate_and_filter_hmmer_alignment():
    #PDB ID:1ZR7_1
    sequence_database = data_path/'selected_sequences.fa'
    protein_sequence="GSWTEHKSPDGRTYYYNTETKQSTWEKPDD"
    with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_filtered_disk.sto') as output_handle:
        output_file = Path(output_handle.name)
        alignment_file=dca_frustratometer.align.jackhmmer(protein_sequence,output_file,sequence_database)
        assert alignment_file.exists()
        output_text = alignment_file.read_text()
        assert "# STOCKHOLM" in output_text
        
def test_create_potts_model_from_aligment():
    filtered_file=data_path/'PF09696.12_gaps_filtered.fasta'
    potts_model = dca_frustratometer.dca.pydca.run(str(filtered_file))
    assert 'h' in potts_model.keys()
    assert 'J' in potts_model.keys()

def test_identify_pfamID():
    pfamID = dca_frustratometer.map.get_pfamID("6U5E","A")
    assert pfamID=="PF00160"


def test_functional_compute_native_energy():
    seq = dca_frustratometer.pdb.get_sequence('examples/data/1cyo.pdb', 'A')
    distance_matrix = dca_frustratometer.pdb.get_distance_matrix('examples/data/1cyo.pdb', 'A')
    potts_model = dca_frustratometer.dca.matlab.load_potts_model('examples/data/PottsModel1cyoA.mat')
    mask = dca_frustratometer.frustration.compute_mask(distance_matrix, distance_cutoff=4, sequence_distance_cutoff=0)
    e = dca_frustratometer.frustration.compute_native_energy(seq, potts_model, mask)
    assert np.round(e, 4) == -61.5248


def test_OOP_compute_native_energy():
    pdb_file = 'examples/data/1cyo.pdb'
    chain = 'A'
    potts_model_file = 'examples/data/PottsModel1cyoA.mat'
    model = dca_frustratometer.PottsModel.from_potts_model_file(potts_model_file, pdb_file, chain, distance_cutoff=4,
                                                                sequence_cutoff=0)
    e = model.native_energy()
    assert np.round(e, 4) == -61.5248

def test_fields_couplings_energy():
    pdb_file = 'examples/data/1cyo.pdb'
    chain = 'A'
    potts_model_file = 'examples/data/PottsModel1cyoA.mat'
    model = dca_frustratometer.PottsModel.from_potts_model_file(potts_model_file, pdb_file, chain, distance_cutoff=4,
                                                                sequence_cutoff=0)
    assert model.fields_energy() + model.couplings_energy() - model.native_energy()  < 1E-6

def test_AWSEM_native_energy():
    model=dca_frustratometer.AWSEMFrustratometer('examples/data/1l63.pdb','A')
    e = model.native_energy()
    assert np.round(e, 4) == -914.9407



def test_scores():
    pdb_file = 'examples/data/1cyo.pdb'
    chain = 'A'
    potts_model_file = 'examples/data/PottsModel1cyoA.mat'
    model = dca_frustratometer.PottsModel.from_potts_model_file(potts_model_file, pdb_file, chain, distance_cutoff=4,
                                                                sequence_cutoff=0)
    assert np.round(model.scores()[30, 40], 5) == -0.02234


def test_compute_singleresidue_decoy_energy():
    aa_x = 5
    pos_x = 30
    distance_cutoff = 4
    sequence_cutoff = 0
    distance_matrix = dca_frustratometer.pdb.get_distance_matrix('examples/data/1cyo.pdb', 'A')
    potts_model = dca_frustratometer.dca.matlab.load_potts_model('examples/data/PottsModel1cyoA.mat')
    seq = dca_frustratometer.pdb.get_sequence('examples/data/1cyo.pdb', 'A')
    mask = dca_frustratometer.frustration.compute_mask(distance_matrix, distance_cutoff, sequence_cutoff)
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq = [aa for aa in seq]
    seq[pos_x] = AA[aa_x]
    seq = ''.join(seq)
    test_energy = dca_frustratometer.frustration.compute_native_energy(seq, potts_model, mask)
    decoy_energy = dca_frustratometer.frustration.compute_decoy_energy(seq, potts_model, mask, 'singleresidue')
    assert (decoy_energy[pos_x, aa_x] - test_energy) ** 2 < 1E-16


def test_compute_mutational_decoy_energy():
    aa_x = 5
    pos_x = 30
    aa_y = 7
    pos_y = 69
    distance_cutoff = 4
    sequence_cutoff = 0
    distance_matrix = dca_frustratometer.pdb.get_distance_matrix('examples/data/1cyo.pdb', 'A')
    potts_model = dca_frustratometer.dca.matlab.load_potts_model('examples/data/PottsModel1cyoA.mat')
    seq = dca_frustratometer.pdb.get_sequence('examples/data/1cyo.pdb', 'A')
    mask = dca_frustratometer.frustration.compute_mask(distance_matrix, distance_cutoff, sequence_cutoff)
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq = [aa for aa in seq]
    seq[pos_x] = AA[aa_x]
    seq[pos_y] = AA[aa_y]
    seq = ''.join(seq)
    test_energy = dca_frustratometer.frustration.compute_native_energy(seq, potts_model, mask)
    decoy_energy = dca_frustratometer.frustration.compute_decoy_energy(seq, potts_model, mask, 'mutational')
    assert (decoy_energy[pos_x, pos_y, aa_x, aa_y] - test_energy) ** 2 < 1E-16

@pytest.mark.xfail
def test_initialize_from_pdb():
    PFAM_id='PFxxxxx'
    pdb='file.pdb'
    expected_energy=10.0
    potts_model=dca_frustratometer.PottsModel.from_PFAM(PFAM_id)
    potts_model.set_structure(pdb)
    assert potts_model.compute_native_energy()==expected_energy


def test_from_potts_model_file():
    pass

def from_pfam_alignment():
    pass

def from_hmmer_alignment():
    pass

def from_alignment():
    pass