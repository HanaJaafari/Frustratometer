"""
Unit and regression test for the dca_frustratometer package.
"""

# Import package, test suite, and other packages as needed
import sys
import frustratometer
import numpy as np
from pathlib import Path
import tempfile
import pytest
import pandas as pd
from frustratometer.utils import _path
import Bio.AlignIO
import subprocess

data_path = frustratometer.utils.create_directory(_path/'..'/'tests'/'data')
#scratch_path = dca_frustratometer.utils.create_directory(_path/'..'/'tests'/'scratch')

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def test_frustratometer_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "frustratometer" in sys.modules

def test_download_pfam_database():
    """Downloads a small database from Pfam and tests that the files are splitted correctly."""

    # Define the input parameters and expected output file names
    name = 'pfam_dead'
    url = 'https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.dead.gz' #Small database
    non_existent_file = 'Unknown.sto'
    expected_file = 'PF00065.sto'

    # Use a temporary directory to store the downloaded and split files
    with tempfile.TemporaryDirectory() as scratch_path:
        scratch_path = Path(scratch_path)

        # Call the download_database function with the specified URL and name
        alignments_path = frustratometer.pfam.download_database(scratch_path, url=url, name=name)

        # Check that the non-existent file is not present in the downloaded files
        assert not (alignments_path / non_existent_file).exists(), f"{non_existent_file} should not exist in the downloaded files."

        # Check that the expected file is present in the downloaded files
        assert (alignments_path / expected_file).exists(), f"{expected_file} should exist in the downloaded files."

def test_get_alignment_from_database():
    """Test that the get_alignment function retrieves the correct alignment files."""
    # Test the first alignment (PF17182)
    alignment_file = frustratometer.pfam.get_alignment('PF17182', data_path / 'pfam_database')
    assert alignment_file.exists()
    alignment_text = alignment_file.read_text()
    assert "#=GF AC   PF17182" in alignment_text

    # Test the second alignment (PF09696)
    alignment_file = frustratometer.pfam.get_alignment('PF09696', data_path / 'pfam_database')
    assert alignment_file.exists()
    alignment_text = alignment_file.read_text()
    assert "#=GF AC   PF09696" in alignment_text

def test_download_pfam_alignment():
    """Test that the download_pfam_alignment function downloads the correct alignment."""
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_interpro.sto') as output_handle:
        output = Path(output_handle.name)
        assert output.exists()

        # Check that the output file is initially empty
        output_text = output.read_text()
        assert output_text == ""

        # Download the PF09696 alignment
        output = frustratometer.pfam.download_aligment('PF09696', output, alignment_type='seed')
        assert output.exists()

        # Check that the downloaded alignment is correct
        output_text = output.read_text()
        assert "#=GF AC   PF09696" in output_text

    # Check that the temporary output file is deleted after the context is closed
    assert not output.exists()

def test_filter_alignment_memory():
    """Test the filter_alignment function using the in-memory method."""
    alignment_file = data_path / 'pfam_database' / 'PF09696.12.sto'
    expected_filtered_sequence = '-IQTPSGLALLELQGTINLPEDAVDSDGKAT-------------KSIPVGRIDFPDYHPDTQSTAWMKRVYLYVGPHQRLTGEVKKLPKAIAIVRKKDGASNG-----------------------------------------'

    with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_filtered_memory.sto') as output_handle:
        output_file = Path(output_handle.name)

        # Filter the alignment using the in-memory method
        filtered_file = frustratometer.filter.filter_alignment(alignment_file, output_file)

        # Check the results of the filtering
        assert filtered_file.exists()
        unfiltered_alignment = Bio.AlignIO.read(alignment_file, 'stockholm')
        filtered_alignment = Bio.AlignIO.read(filtered_file, 'fasta')
        assert len(unfiltered_alignment) == len(filtered_alignment)
        assert unfiltered_alignment.get_alignment_length() > filtered_alignment.get_alignment_length()
        assert filtered_alignment.get_alignment_length() == len(expected_filtered_sequence)
        assert filtered_alignment[0].seq == expected_filtered_sequence

def test_filter_alignment_lowmem():
    """Test the filter_alignment function using the low-memory method."""
    alignment_file = data_path / 'pfam_database' / 'PF17182.6.sto'
    expected_filtered_sequence = 'QHDSMFTINSDYDAYLLDFPLLGDDFLLYLARMELRCRFKRTERVLQSGLCVSGQTISGARSRLHHLLVNKTQIIVNIGSVDIMRGRPIVQIQHDFRQLVK'\
                                 'DMHNRGLVPILTTLAPLANYCHDKAMCDKVVKFNQFIWKECASYLKVIDIHSCLVNENGVVRFDCFQYSSRNVTGSKESYVFWNKIGRQRVLQMIEASLE'

    with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_filtered_disk.sto') as output_handle:
        output_file = Path(output_handle.name)

        # Filter the alignment using the low-memory method
        filtered_file = frustratometer.filter.filter_alignment_lowmem(alignment_file, output_file)

        # Check the results of the filtering
        assert filtered_file.exists()
        unfiltered_alignment = Bio.AlignIO.read(alignment_file, 'stockholm')
        filtered_alignment = Bio.AlignIO.read(filtered_file, 'fasta')
        assert len(unfiltered_alignment) == len(filtered_alignment)
        assert unfiltered_alignment.get_alignment_length() > filtered_alignment.get_alignment_length()
        assert filtered_alignment.get_alignment_length() == len(expected_filtered_sequence)
        assert filtered_alignment[0].seq == expected_filtered_sequence
   
def test_generate_and_filter_hmmer_alignment():
    #PDB ID:1ZR7_1
    sequence_database = data_path/'selected_sequences.fa'
    protein_sequence="GSWTEHKSPDGRTYYYNTETKQSTWEKPDD"
    with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_filtered_disk.sto') as output_handle:
        output_file = Path(output_handle.name)
        alignment_file=frustratometer.align.jackhmmer(protein_sequence,output_file,sequence_database)
        assert alignment_file.exists()
        output_text = alignment_file.read_text()
        assert "# STOCKHOLM" in output_text
        
def test_create_potts_model_from_aligment():
    """
    Test to check if the Potts model is created from a filtered alignment file using pydca.
    
    The test only executes if the pydca module is installed and if not, it skips with a message.
    If the module is installed, the filtered alignment file is loaded and a Potts model is created using the plmdca function from the pydca module.
    The test then asserts that the keys 'h' and 'J' are present in the Potts model.
    """
    
    try:
        import pydca
    except ImportError:
        pytest.skip('pydca module not installed')
    
    filtered_file=data_path/'PF09696.12_gaps_filtered.fasta'
    potts_model = frustratometer.dca.pydca.plmdca(str(filtered_file))
    assert 'h' in potts_model.keys()
    assert 'J' in potts_model.keys()

def test_identify_pfamID():
    """Test the get_pfamID function to ensure it correctly identifies the Pfam ID."""
    pdb_id = "6U5E"
    chain_id = "A"
    expected_pfam_id = "PF11976"

    pfam_id = frustratometer.map.get_pfamID(pdb_id, chain_id)
    assert pfam_id == expected_pfam_id

def seq_index_mapping():
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq="AWYPQ"
    seq_index = list(np.array([_AA.find(aa) for aa in seq]))
    assert seq_index==[1,19,20,13,14]

#####
#Test masks applied in energy calculations
#####

def test_distance_matrix():
    pdb_path = f'{data_path}/6JXX_A.pdb'
    chain_id = 'A'

    distance_matrix = frustratometer.pdb.get_distance_matrix(pdb_path, chain_id, method='CB')
    original_distance_matrix=np.loadtxt(f"{data_path}/6JXX_A_CB_CB_Distance_Map.txt")
    assert (original_distance_matrix==distance_matrix).all()

def test_couplings_mask_with_sequence_threshold():
    pdb_path = f'{data_path}/6JXX_A.pdb'
    potts_model_path=f"{data_path}/PF11976_PFAM_27_dca_gap_threshold_0.2.mat"
    chain_id = 'A'
    filtered_aligned_sequence="INLKVAGQDGSVVQFKIKRHTPLSKLMKAYCERQGLSM-RQIRFRFDGQPINETDTPAQLEMEDEDTIDV--"
    structure=frustratometer.Structure.full_pdb(pdb_file=pdb_path,chain=chain_id,filtered_aligned_sequence=filtered_aligned_sequence)
    DCA_model=frustratometer.DCA.from_potts_model_file(structure,potts_model_file=potts_model_path,sequence_cutoff=1)
    assert all(p == False for p in np.diag(DCA_model.mask))

def test_couplings_mask_with_distance_threshold():
    pdb_path = f'{data_path}/6JXX_A.pdb'
    potts_model_path=f"{data_path}/PF11976_PFAM_27_dca_gap_threshold_0.2.mat"
    chain_id = 'A'
    filtered_aligned_sequence="INLKVAGQDGSVVQFKIKRHTPLSKLMKAYCERQGLSM-RQIRFRFDGQPINETDTPAQLEMEDEDTIDV--"
    structure=frustratometer.Structure.full_pdb(pdb_file=pdb_path,chain=chain_id,filtered_aligned_sequence=filtered_aligned_sequence)
    DCA_model=frustratometer.DCA.from_potts_model_file(structure,potts_model_file=potts_model_path,distance_cutoff=16)

    original_distance_matrix=np.loadtxt(f"{data_path}/6JXX_A_CB_CB_Distance_Map.txt")
    mask = np.ones([77, 77])
    mask *=original_distance_matrix<=16
    assert (DCA_model.mask==mask.astype(np.bool8)).all()

def test_couplings_mask_with_distance_and_sequence_threshold():
    pdb_path = f'{data_path}/6JXX_A.pdb'
    potts_model_path=f"{data_path}/PF11976_PFAM_27_dca_gap_threshold_0.2.mat"
    chain_id = 'A'
    filtered_aligned_sequence="INLKVAGQDGSVVQFKIKRHTPLSKLMKAYCERQGLSM-RQIRFRFDGQPINETDTPAQLEMEDEDTIDV--"
    structure=frustratometer.Structure.full_pdb(pdb_file=pdb_path,chain=chain_id,filtered_aligned_sequence=filtered_aligned_sequence)
    DCA_model=frustratometer.DCA.from_potts_model_file(structure,potts_model_file=potts_model_path,sequence_cutoff=1,distance_cutoff=16)

    original_distance_matrix=np.loadtxt(f"{data_path}/6JXX_A_CB_CB_Distance_Map.txt")
    mask = np.ones([77, 77])
    mask *=original_distance_matrix<=16
    np.fill_diagonal(mask, 0)
    assert (DCA_model.mask==(mask)).all()

#####
#Test DCA Native Energy Calculations
#####

def test_functional_compute_DCA_native_energy():
    """Test the functional approach to compute the native energy of a protein."""
    pdb_path = 'examples/data/1cyo.pdb'
    chain_id = 'A'
    potts_model_path = 'examples/data/PottsModel1cyoA.mat'
    expected_energy = -61.5248

    sequence = frustratometer.pdb.get_sequence(pdb_path, chain_id)
    distance_matrix = frustratometer.pdb.get_distance_matrix(pdb_path, chain_id, method='minimum')
    potts_model = frustratometer.dca.matlab.load_potts_model(potts_model_path)
    mask = frustratometer.frustration.compute_mask(distance_matrix, distance_cutoff=4, sequence_distance_cutoff=0)
    energy = frustratometer.frustration.compute_native_energy(sequence, potts_model, mask)

    assert np.round(energy, 4) == expected_energy

def test_OOP_compute_DCA_native_energy():
    pdb_file = 'examples/data/1cyo.pdb'
    chain = 'A'
    distance_matrix_method='minimum'
    potts_model_file = 'examples/data/PottsModel1cyoA.mat'
    structure=frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method)
    model = frustratometer.DCA.from_potts_model_file(structure, potts_model_file, distance_cutoff=4,
                                                                sequence_cutoff=0)
                                                                
    e = model.native_energy()
    assert np.round(e, 4) == -61.5248

def test_OOP_compute_seq_DCA_energy_with_distance_threshold_without_gap_terms():
    pdb_file = f'{data_path}/6JXX_A.pdb'
    chain = 'A'
    distance_matrix_method='CB'
    potts_model_file = f"{data_path}/PF11976_PFAM_27_dca_gap_threshold_0.2.mat"
    filtered_aligned_sequence="INLKVAGQDGSVVQFKIKRHTPLSKLMKAYCERQGLSM-RQIRFRFDGQPINETDTPAQLEMEDEDTIDV--"
    aligned_sequence=subprocess.check_output(["sed","-n",""'/>%s$/,/>/p'"" % "6JXX_A",f'{data_path}/PF11976_all_pseudogene_parent_sequences_aligned_PFAM_27.fasta'])
    aligned_sequence="".join(aligned_sequence.decode().split("\n")[1:-2])

    structure=frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method,filtered_aligned_sequence=filtered_aligned_sequence,aligned_sequence=aligned_sequence)
    model = frustratometer.DCA.from_potts_model_file(structure, potts_model_file, distance_cutoff=16,
                                                                sequence_cutoff=1,reformat_potts_model=True)

    sample_sequence="-KLKVIGQDSSEIHFKVKMTTHLKKLKESYCQRQGVPM-NSLRFVFEDQRIAATHTIKELGMEE-DVIEVY-"                                                            
    e = model.native_energy(sequence=sample_sequence,ignore_couplings_of_gaps=True,ignore_fields_of_gaps=True)
    assert np.round(e, 4) == -408.8334

def test_OOP_compute_seq_DCA_energy_with_distance_threshold_with_gap_terms():
    pdb_file = f'{data_path}/6JXX_A.pdb'
    chain = 'A'
    distance_matrix_method='CB'
    potts_model_file = f"{data_path}/PF11976_PFAM_27_dca_gap_threshold_0.2.mat"
    filtered_aligned_sequence="INLKVAGQDGSVVQFKIKRHTPLSKLMKAYCERQGLSM-RQIRFRFDGQPINETDTPAQLEMEDEDTIDV--"
    aligned_sequence=subprocess.check_output(["sed","-n",""'/>%s$/,/>/p'"" % "6JXX_A",f'{data_path}/PF11976_all_pseudogene_parent_sequences_aligned_PFAM_27.fasta'])
    aligned_sequence="".join(aligned_sequence.decode().split("\n")[1:-2])

    structure=frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method,filtered_aligned_sequence=filtered_aligned_sequence,aligned_sequence=aligned_sequence)
    model = frustratometer.DCA.from_potts_model_file(structure, potts_model_file, distance_cutoff=16,
                                                                sequence_cutoff=1,reformat_potts_model=True)

    sample_sequence="-KLKVIGQDSSEIHFKVKMTTHLKKLKESYCQRQGVPM-NSLRFVFEDQRIAATHTIKELGMEE-DVIEVY-"                                                            
    e = model.native_energy(sequence=sample_sequence)
    assert np.round(e, 4) == -424.321

def test_OOP_compute_seq_DCA_energy_without_gap_terms():
    pdb_file = f'{data_path}/6JXX_A.pdb'
    chain = 'A'
    distance_matrix_method='CB'
    potts_model_file = f"{data_path}/PF11976_PFAM_27_dca_gap_threshold_0.2.mat"
    filtered_aligned_sequence="INLKVAGQDGSVVQFKIKRHTPLSKLMKAYCERQGLSM-RQIRFRFDGQPINETDTPAQLEMEDEDTIDV--"

    structure=frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method,filtered_aligned_sequence=filtered_aligned_sequence)
    model = frustratometer.DCA.from_potts_model_file(structure, potts_model_file, sequence_cutoff=1,reformat_potts_model=True)

    sample_sequence="-KLKVIGQDSSEIHFKVKMTTHLKKLKESYCQRQGVPM-NSLRFVFEDQRIAATHTIKELGMEE-DVIEVY-"                                                            
    e = model.native_energy(sequence=sample_sequence,ignore_couplings_of_gaps=True,ignore_fields_of_gaps=True)
    assert np.round(e, 4) == -612.0897

def test_OOP_compute_seq_DCA_energy_with_gap_terms():
    pdb_file = f'{data_path}/6JXX_A.pdb'
    chain = 'A'
    distance_matrix_method='CB'
    potts_model_file = f"{data_path}/PF11976_PFAM_27_dca_gap_threshold_0.2.mat"
    filtered_aligned_sequence="INLKVAGQDGSVVQFKIKRHTPLSKLMKAYCERQGLSM-RQIRFRFDGQPINETDTPAQLEMEDEDTIDV--"

    structure=frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method,filtered_aligned_sequence=filtered_aligned_sequence)
    model = frustratometer.DCA.from_potts_model_file(structure, potts_model_file, sequence_cutoff=1,reformat_potts_model=True)

    sample_sequence="-KLKVIGQDSSEIHFKVKMTTHLKKLKESYCQRQGVPM-NSLRFVFEDQRIAATHTIKELGMEE-DVIEVY-"                                                            
    e = model.native_energy(sequence=sample_sequence)
    assert np.round(e, 4) == -685.4002

def test_fields_couplings_DCA_energy():
    pdb_file = 'examples/data/1cyo.pdb'
    chain = 'A'
    distance_matrix_method='minimum'
    potts_model_file = 'examples/data/PottsModel1cyoA.mat'
    structure=frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method)
    model = frustratometer.DCA.from_potts_model_file(structure, potts_model_file, distance_cutoff=4,
                                                                sequence_cutoff=0)
    assert model.fields_energy() + model.couplings_energy() - model.native_energy()  < 1E-6

#####
#Test Full Protein Structure Object
#####
def test_structure_class():
    #PDB has cofactors and ions
    structure=frustratometer.Structure.full_pdb(f'{_path}/../tests/data/1rnb.pdb',"A")
    test_sequence="QVINTFDGVADYLQTYHKLPNDYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
    assert structure.sequence==test_sequence
    assert structure.distance_matrix.shape==(len(test_sequence),len(test_sequence))

@pytest.mark.skip
def test_structure_segment_class_original_indices():
    #PDB has cofactors and ions
    structure=frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/3ptn.pdb',"A",seq_selection="resnum `16to41`")
    test_sequence="IVGGYTCGANTVPYQVSLNSGYHF"
    selection_CB = structure.structure.select('name CB or (resname GLY IGL and name CA)')
    resid = selection_CB.getResindices()
    assert structure.pdb_init_index==16
    assert len(structure.select_gap_indices)==2
    assert structure.sequence==test_sequence
    assert structure.distance_matrix.shape == (len(structure.sequence),len(structure.sequence))
    assert len(resid)==len(structure.sequence)

def test_structure_segment_class_absolute_indices():
    #PDB has cofactors and ions
    structure=frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/3ptn.pdb',"A",seq_selection="resindex `0to23`")
    test_sequence="IVGGYTCGANTVPYQVSLNSGYHF"
    selection_CB = structure.structure.select('name CB or (resname GLY IGL and name CA)')
    resid = selection_CB.getResindices()
    assert structure.sequence==test_sequence
    assert structure.distance_matrix.shape == (len(structure.sequence),len(structure.sequence))
    assert len(resid)==len(structure.sequence)

#####
#Test Protein Segment Structure Object
#####

def test_structure_segment_class_original_indices_no_repair():
    structure=frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/1rnb.pdb',"A",seq_selection="resnum `2to21`",repair_pdb=False)
    test_sequence="QVINTFDGVADYLQTYHKLP"
    selection_CB = structure.structure.select('name CB or (resname GLY IGL and name CA)')
    resid = selection_CB.getResindices()
    assert structure.sequence==test_sequence
    assert structure.distance_matrix.shape == (len(structure.sequence),len(structure.sequence))
    assert len(resid)==len(structure.sequence)

def test_structure_segment_class_absolute_indices_no_repair():
    structure=frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/1rnb.pdb',"A",seq_selection="resindex `0to19`",repair_pdb=False)
    test_sequence="QVINTFDGVADYLQTYHKLP"
    selection_CB = structure.structure.select('name CB or (resname GLY IGL and name CA)')
    resid = selection_CB.getResindices()
    assert structure.sequence==test_sequence
    assert structure.distance_matrix.shape == (len(structure.sequence),len(structure.sequence))
    assert len(resid)==len(structure.sequence)

def test_scores():
    pdb_file = 'examples/data/1cyo.pdb'
    chain = 'A'
    structure=frustratometer.Structure.full_pdb(pdb_file,chain)
    potts_model_file = 'examples/data/PottsModel1cyoA.mat'
    model = frustratometer.DCA.from_potts_model_file(structure, potts_model_file, distance_cutoff=4,
                                                                sequence_cutoff=0)
    assert np.round(model.scores()[30, 40], 5) == -0.02234

#####
#Test DCA Decoy Energy Calculation
#####

def test_compute_singleresidue_DCA_decoy_energy():
    aa_x = 5
    pos_x = 30
    distance_cutoff = 4
    sequence_cutoff = 0
    distance_matrix = frustratometer.pdb.get_distance_matrix('examples/data/1cyo.pdb', 'A')
    potts_model = frustratometer.dca.matlab.load_potts_model('examples/data/PottsModel1cyoA.mat')
    seq = frustratometer.pdb.get_sequence('examples/data/1cyo.pdb', 'A')
    mask = frustratometer.frustration.compute_mask(distance_matrix, distance_cutoff, sequence_cutoff)
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq = [aa for aa in seq]
    seq[pos_x] = AA[aa_x]
    seq = ''.join(seq)
    test_energy = frustratometer.frustration.compute_native_energy(seq, potts_model, mask)
    decoy_energy = frustratometer.frustration.compute_decoy_energy(seq, potts_model, mask, 'singleresidue')
    assert (decoy_energy[pos_x, aa_x] - test_energy) ** 2 < 1E-16


def test_compute_mutational_DCA_decoy_energy():
    aa_x = 5
    pos_x = 30
    aa_y = 7
    pos_y = 69
    distance_cutoff = 4
    sequence_cutoff = 0
    distance_matrix = frustratometer.pdb.get_distance_matrix('examples/data/1cyo.pdb', 'A')
    potts_model = frustratometer.dca.matlab.load_potts_model('examples/data/PottsModel1cyoA.mat')
    seq = frustratometer.pdb.get_sequence('examples/data/1cyo.pdb', 'A')
    mask = frustratometer.frustration.compute_mask(distance_matrix, distance_cutoff, sequence_cutoff)
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq = [aa for aa in seq]
    seq[pos_x] = AA[aa_x]
    seq[pos_y] = AA[aa_y]
    seq = ''.join(seq)
    test_energy = frustratometer.frustration.compute_native_energy(seq, potts_model, mask)
    decoy_energy = frustratometer.frustration.compute_decoy_energy(seq, potts_model, mask, 'mutational')
    assert (decoy_energy[pos_x, pos_y, aa_x, aa_y] - test_energy) ** 2 < 1E-16





@pytest.mark.xfail
def test_initialize_from_pdb():
    PFAM_id='PFxxxxx'
    pdb='file.pdb'
    expected_energy=10.0
    potts_model=frustratometer.DCA.from_PFAM(PFAM_id)
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