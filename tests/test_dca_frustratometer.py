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
import pandas as pd
from dca_frustratometer.utils import _path
import Bio.AlignIO

data_path = dca_frustratometer.utils.create_directory(_path/'..'/'tests'/'data')
#scratch_path = dca_frustratometer.utils.create_directory(_path/'..'/'tests'/'scratch')

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def test_dca_frustratometer_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "dca_frustratometer" in sys.modules

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
        alignments_path = dca_frustratometer.pfam.download_database(scratch_path, url=url, name=name)

        # Check that the non-existent file is not present in the downloaded files
        assert not (alignments_path / non_existent_file).exists(), f"{non_existent_file} should not exist in the downloaded files."

        # Check that the expected file is present in the downloaded files
        assert (alignments_path / expected_file).exists(), f"{expected_file} should exist in the downloaded files."

def test_get_alignment_from_database():
    """Test that the get_alignment function retrieves the correct alignment files."""
    # Test the first alignment (PF17182)
    alignment_file = dca_frustratometer.pfam.get_alignment('PF17182', data_path / 'pfam_database')
    assert alignment_file.exists()
    alignment_text = alignment_file.read_text()
    assert "#=GF AC   PF17182" in alignment_text

    # Test the second alignment (PF09696)
    alignment_file = dca_frustratometer.pfam.get_alignment('PF09696', data_path / 'pfam_database')
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
        output = dca_frustratometer.pfam.download_aligment('PF09696', output, alignment_type='seed')
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
        filtered_file = dca_frustratometer.filter.filter_alignment(alignment_file, output_file)

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
        filtered_file = dca_frustratometer.filter.filter_alignment_lowmem(alignment_file, output_file)

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
        alignment_file=dca_frustratometer.align.jackhmmer(protein_sequence,output_file,sequence_database)
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
    potts_model = dca_frustratometer.dca.pydca.plmdca(str(filtered_file))
    assert 'h' in potts_model.keys()
    assert 'J' in potts_model.keys()

def test_identify_pfamID():
    """Test the get_pfamID function to ensure it correctly identifies the Pfam ID."""
    pdb_id = "6U5E"
    chain_id = "A"
    expected_pfam_id = "PF00160"

    pfam_id = dca_frustratometer.map.get_pfamID(pdb_id, chain_id)
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
    pdb_path = 'tests/data/6u5e.pdb'
    chain_id = 'A'

    distance_matrix = dca_frustratometer.pdb.get_distance_matrix(pdb_path, chain_id, method='CB')
    original_distance_matrix=np.loadtxt("examples/data/6U5E_A_CB_CB_Distance_Map.txt")
    assert (original_distance_matrix==distance_matrix).all()

def test_couplings_mask_with_sequence_threshold():
    pdb_path = 'examples/data/6U5E_A.pdb'
    potts_model_path="examples/data/PF00160_PFAM_27_dca_gap_threshold_0.2.mat"
    chain_id = 'A'
    filtered_aligned_sequence="-FDIAVDGLGRVSFELFADKVPKTAENFRALST-GGYKGSCFHRIIPGFMCQGGDFTRHNG--TGGSIYGEKFEDEN--FILKHGPGILSMANAG--PNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERGSRNGKTSKKITIADCG-"
    structure=dca_frustratometer.Structure.full_pdb(pdb_file=pdb_path,chain=chain_id,filtered_aligned_sequence=filtered_aligned_sequence)
    DCA_model=dca_frustratometer.PottsModel.from_potts_model_file(structure,potts_model_file=potts_model_path,sequence_cutoff=1)
    assert all(p == False for p in np.diag(DCA_model.mask))

def test_couplings_mask_with_distance_threshold():
    pdb_path = 'examples/data/6U5E_A.pdb'
    potts_model_path="examples/data/PF00160_PFAM_27_dca_gap_threshold_0.2.mat"
    chain_id = 'A'
    filtered_aligned_sequence="-FDIAVDGLGRVSFELFADKVPKTAENFRALST-GGYKGSCFHRIIPGFMCQGGDFTRHNG--TGGSIYGEKFEDEN--FILKHGPGILSMANAG--PNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERGSRNGKTSKKITIADCG-"
    structure=dca_frustratometer.Structure.full_pdb(pdb_file=pdb_path,chain=chain_id,filtered_aligned_sequence=filtered_aligned_sequence)
    DCA_model=dca_frustratometer.PottsModel.from_potts_model_file(structure,potts_model_file=potts_model_path,distance_cutoff=16)

    original_distance_matrix=np.loadtxt("examples/data/6U5E_A_CB_CB_Distance_Map.txt")
    mask = np.ones([163, 163])
    mask *=original_distance_matrix<=16
    assert (DCA_model.mask==mask.astype(np.bool8)).all()

def test_couplings_mask_with_distance_and_sequence_threshold():
    pdb_path = 'examples/data/6U5E_A.pdb'
    potts_model_path="examples/data/PF00160_PFAM_27_dca_gap_threshold_0.2.mat"
    chain_id = 'A'
    filtered_aligned_sequence="-FDIAVDGLGRVSFELFADKVPKTAENFRALST-GGYKGSCFHRIIPGFMCQGGDFTRHNG--TGGSIYGEKFEDEN--FILKHGPGILSMANAG--PNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERGSRNGKTSKKITIADCG-"
    structure=dca_frustratometer.Structure.full_pdb(pdb_file=pdb_path,chain=chain_id,filtered_aligned_sequence=filtered_aligned_sequence)
    DCA_model=dca_frustratometer.PottsModel.from_potts_model_file(structure,potts_model_file=potts_model_path,sequence_cutoff=1,distance_cutoff=16)

    original_distance_matrix=np.loadtxt("examples/data/6U5E_A_CB_CB_Distance_Map.txt")
    mask = np.ones([163, 163])
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

    sequence = dca_frustratometer.pdb.get_sequence(pdb_path, chain_id)
    distance_matrix = dca_frustratometer.pdb.get_distance_matrix(pdb_path, chain_id, method='minimum')
    potts_model = dca_frustratometer.dca.matlab.load_potts_model(potts_model_path)
    mask = dca_frustratometer.frustration.compute_mask(distance_matrix, distance_cutoff=4, sequence_distance_cutoff=0)
    energy = dca_frustratometer.frustration.compute_native_energy(sequence, potts_model, mask)

    assert np.round(energy, 4) == expected_energy

def test_OOP_compute_DCA_native_energy():
    pdb_file = 'examples/data/1cyo.pdb'
    chain = 'A'
    distance_matrix_method='minimum'
    potts_model_file = 'examples/data/PottsModel1cyoA.mat'
    structure=dca_frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method)
    model = dca_frustratometer.PottsModel.from_potts_model_file(structure, potts_model_file, distance_cutoff=4,
                                                                sequence_cutoff=0)
                                                                
    e = model.native_energy()
    assert np.round(e, 4) == -61.5248

def test_OOP_compute_seq_DCA_energy_with_distance_threshold_without_gap_terms():
    import subprocess
    pdb_file = 'examples/data/6U5E_A.pdb'
    chain = 'A'
    distance_matrix_method='CB'
    potts_model_file = "examples/data/PF00160_PFAM_27_dca_gap_threshold_0.2.mat"
    filtered_aligned_sequence="-FDIAVDGLGRVSFELFADKVPKTAENFRALST-GGYKGSCFHRIIPGFMCQGGDFTRHNG--TGGSIYGEKFEDEN--FILKHGPGILSMANAG--PNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERGSRNGKTSKKITIADCG-"
    aligned_sequence=subprocess.check_output(["sed","-n",""'/>%s$/,/>/p'"" % "6U5E_A",'examples/data/PF00160_all_pseudogene_parent_sequences_aligned_PFAM_27.fasta'])
    aligned_sequence="".join(aligned_sequence.decode().split("\n")[1:-2])

    structure=dca_frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method,filtered_aligned_sequence=filtered_aligned_sequence,aligned_sequence=aligned_sequence)
    model = dca_frustratometer.PottsModel.from_potts_model_file(structure, potts_model_file, distance_cutoff=16,
                                                                sequence_cutoff=1,reformat_potts_model=True)

    sample_sequence="--NIAINSLGHVSFELFADKFPKT-ENFRALST-GGYKGSCFHRIILGLLCQGGDFTCHNGTGGK-SVYREKFDDEN--FSMKHGPGILSMANAG--PNTNDSQIFICTAKTEWLDGKHVVSGRVKEGIKIVEAMKRGSKNGKSRKKITTADCG-"                                                            
    e = model.native_energy(sequence=sample_sequence,ignore_couplings_of_gaps=True,ignore_fields_of_gaps=True)
    assert np.round(e, 4) == -769.5400

def test_OOP_compute_seq_DCA_energy_with_distance_threshold_with_gap_terms():
    import subprocess
    pdb_file = 'examples/data/6U5E_A.pdb'
    chain = 'A'
    distance_matrix_method='CB'
    potts_model_file = "examples/data/PF00160_PFAM_27_dca_gap_threshold_0.2.mat"
    filtered_aligned_sequence="-FDIAVDGLGRVSFELFADKVPKTAENFRALST-GGYKGSCFHRIIPGFMCQGGDFTRHNG--TGGSIYGEKFEDEN--FILKHGPGILSMANAG--PNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERGSRNGKTSKKITIADCG-"
    aligned_sequence=subprocess.check_output(["sed","-n",""'/>%s$/,/>/p'"" % "6U5E_A",'examples/data/PF00160_all_pseudogene_parent_sequences_aligned_PFAM_27.fasta'])
    aligned_sequence="".join(aligned_sequence.decode().split("\n")[1:-2])

    structure=dca_frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method,filtered_aligned_sequence=filtered_aligned_sequence,aligned_sequence=aligned_sequence)
    model = dca_frustratometer.PottsModel.from_potts_model_file(structure, potts_model_file, distance_cutoff=16,
                                                                sequence_cutoff=1,reformat_potts_model=True)

    sample_sequence="--NIAINSLGHVSFELFADKFPKT-ENFRALST-GGYKGSCFHRIILGLLCQGGDFTCHNGTGGK-SVYREKFDDEN--FSMKHGPGILSMANAG--PNTNDSQIFICTAKTEWLDGKHVVSGRVKEGIKIVEAMKRGSKNGKSRKKITTADCG-"                                                            
    e = model.native_energy(sequence=sample_sequence)
    assert np.round(e, 4) == -801.9952

def test_OOP_compute_seq_DCA_energy_without_gap_terms():
    import subprocess
    pdb_file = 'examples/data/6U5E_A.pdb'
    chain = 'A'
    distance_matrix_method='CB'
    potts_model_file = "examples/data/PF00160_PFAM_27_dca_gap_threshold_0.2.mat"
    filtered_aligned_sequence="-FDIAVDGLGRVSFELFADKVPKTAENFRALST-GGYKGSCFHRIIPGFMCQGGDFTRHNG--TGGSIYGEKFEDEN--FILKHGPGILSMANAG--PNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERGSRNGKTSKKITIADCG-"

    structure=dca_frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method,filtered_aligned_sequence=filtered_aligned_sequence)
    model = dca_frustratometer.PottsModel.from_potts_model_file(structure, potts_model_file, sequence_cutoff=1,reformat_potts_model=True)

    sample_sequence="--NIAINSLGHVSFELFADKFPKT-ENFRALST-GGYKGSCFHRIILGLLCQGGDFTCHNGTGGK-SVYREKFDDEN--FSMKHGPGILSMANAG--PNTNDSQIFICTAKTEWLDGKHVVSGRVKEGIKIVEAMKRGSKNGKSRKKITTADCG-"                                                            
    e = model.native_energy(sequence=sample_sequence,ignore_couplings_of_gaps=True,ignore_fields_of_gaps=True)
    assert np.round(e, 4) == -1265.9532

def test_OOP_compute_seq_DCA_energy_with_gap_terms():
    import subprocess
    pdb_file = 'examples/data/6U5E_A.pdb'
    chain = 'A'
    distance_matrix_method='CB'
    potts_model_file = "examples/data/PF00160_PFAM_27_dca_gap_threshold_0.2.mat"
    filtered_aligned_sequence="-FDIAVDGLGRVSFELFADKVPKTAENFRALST-GGYKGSCFHRIIPGFMCQGGDFTRHNG--TGGSIYGEKFEDEN--FILKHGPGILSMANAG--PNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERGSRNGKTSKKITIADCG-"

    structure=dca_frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method,filtered_aligned_sequence=filtered_aligned_sequence)
    model = dca_frustratometer.PottsModel.from_potts_model_file(structure, potts_model_file, sequence_cutoff=1,reformat_potts_model=True)

    sample_sequence="--NIAINSLGHVSFELFADKFPKT-ENFRALST-GGYKGSCFHRIILGLLCQGGDFTCHNGTGGK-SVYREKFDDEN--FSMKHGPGILSMANAG--PNTNDSQIFICTAKTEWLDGKHVVSGRVKEGIKIVEAMKRGSKNGKSRKKITTADCG-"                                                            
    e = model.native_energy(sequence=sample_sequence)
    assert np.round(e, 4) == -1453.2369

def test_fields_couplings_DCA_energy():
    pdb_file = 'examples/data/1cyo.pdb'
    chain = 'A'
    distance_matrix_method='minimum'
    potts_model_file = 'examples/data/PottsModel1cyoA.mat'
    structure=dca_frustratometer.Structure.full_pdb(pdb_file,chain,distance_matrix_method=distance_matrix_method)
    model = dca_frustratometer.PottsModel.from_potts_model_file(structure, potts_model_file, distance_cutoff=4,
                                                                sequence_cutoff=0)
    assert model.fields_energy() + model.couplings_energy() - model.native_energy()  < 1E-6

#####
#Test AWSEM Native Energy Calculations
#####

def test_residue_density_calculation():
    #Import Lammps AWSEM Frustratometer single residue frustration values
    lammps_single_frustration_dataframe=pd.read_csv(f"{_path}/../tests/data/6U5E_A_tertiary_frustration_singleresidue_1E8decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    lammps_single_frustration_dataframe["i"]=lammps_single_frustration_dataframe["i"]-1
    expected_rho_values=lammps_single_frustration_dataframe["rho_i"]

    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../examples/data/6U5E_A.pdb',"A")
    model=dca_frustratometer.AWSEMFrustratometer(structure,distance_cutoff_contact=9.499,
                                                  min_sequence_separation_contact=2)
    check_rho_values=model.rho_r
    assert np.round(model.rho_r,2).all()==np.round(expected_rho_values,2).all()

def test_AWSEM_native_energy():
    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../examples/data/1l63.pdb',"A")
    model=dca_frustratometer.AWSEMFrustratometer(structure)
    e = model.native_energy()
    print(e)
    assert np.round(e, 0) == -915

def test_AWSEM_fields_energy():
    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../examples/data/6U5E_A.pdb',"A")
    model=dca_frustratometer.AWSEMFrustratometer(structure)
    e = model.fields_energy()
    print(e)
    assert np.round(e, 0) == -555

def test_AWSEM_couplings_energy():
    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../examples/data/6U5E_A.pdb',"A")
    model=dca_frustratometer.AWSEMFrustratometer(structure)
    e = model.couplings_energy()
    print(e)
    assert np.round(e, 0) == -362

def test_fields_couplings_AWSEM_energy():
    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../examples/data/6U5E_A.pdb',"A")
    model = dca_frustratometer.AWSEMFrustratometer(structure)
    assert model.fields_energy() + model.couplings_energy() - model.native_energy()  < 1E-6

def test_single_residue_AWSEM_energy():
    #Import Lammps AWSEM Frustratometer single residue frustration values
    lammps_single_frustration_dataframe=pd.read_csv(f"{_path}/../tests/data/6U5E_A_tertiary_frustration_singleresidue_1E8decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    ###
    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../examples/data/6U5E_A.pdb',"A")
    model=dca_frustratometer.AWSEMFrustratometer(structure,distance_cutoff_contact=9.499,
                                                  min_sequence_separation_contact=2)
    #Calculate fields
    seq_index = np.array([_AA.find(aa) for aa in structure.sequence])
    seq_len = len(seq_index)
    h = -model.potts_model['h'][range(seq_len), seq_index]

    #Calculate couplings
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    j = -model.potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * model.mask

    test_residue_total_energy=(h +j_prime.sum(axis=0))/4.184

    assert (abs(np.array(lammps_single_frustration_dataframe["native_energy"])-test_residue_total_energy) < 1E-1).all()

def test_contact_pair_AWSEM_energy():
    #Import Lammps AWSEM Frustratometer mutational frustration values
    lammps_mutational_frustration_dataframe=pd.read_csv(f"{_path}/../tests/data/6U5E_A_tertiary_frustration_mutational_1E6decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    lammps_mutational_frustration_dataframe["i"]=lammps_mutational_frustration_dataframe["i"]-1
    lammps_mutational_frustration_dataframe["j"]=lammps_mutational_frustration_dataframe["j"]-1
    ###
    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../examples/data/6U5E_A.pdb',"A")
    model=dca_frustratometer.AWSEMFrustratometer(structure,distance_cutoff_contact=9.499,
                                                  min_sequence_separation_contact=None)
    #Calculate fields
    seq_index = np.array([_AA.find(aa) for aa in structure.sequence])
    seq_len = len(seq_index)
    h = -model.potts_model['h'][range(seq_len), seq_index]

    #Calculate couplings
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    j = -model.potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * model.mask
    test_contact_energy_matrix=h[pos1]+h[pos2]+j_prime.sum(axis=0)[pos1]+j_prime.sum(axis=0)[pos2]-j_prime[pos1,pos2]
    ###
    lammps_mutational_frustration_dataframe["Test_Native_Energy"]=lammps_mutational_frustration_dataframe.apply(lambda x: test_contact_energy_matrix[x.i,x.j],axis=1)
    lammps_mutational_frustration_dataframe["Test_Native_Energy"]=lammps_mutational_frustration_dataframe["Test_Native_Energy"]/4.184

    assert (abs(np.array(lammps_mutational_frustration_dataframe["native_energy"])-np.array(lammps_mutational_frustration_dataframe["Test_Native_Energy"])) < 1E-1).all()

#####
#Test Full Protein Structure Object
#####
def test_structure_class():
    #PDB has cofactors and ions
    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../tests/data/1rnb.pdb',"A")
    test_sequence="QVINTFDGVADYLQTYHKLPNDYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
    assert structure.sequence==test_sequence
    assert structure.distance_matrix.shape==(len(test_sequence),len(test_sequence))

@pytest.mark.skip
def test_structure_segment_class_original_indices():
    #PDB has cofactors and ions
    structure=dca_frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/3ptn.pdb',"A",seq_selection="resnum `16to41`")
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
    structure=dca_frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/3ptn.pdb',"A",seq_selection="resindex `0to23`")
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
    structure=dca_frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/1rnb.pdb',"A",seq_selection="resnum `2to21`",repair_pdb=False)
    test_sequence="QVINTFDGVADYLQTYHKLP"
    selection_CB = structure.structure.select('name CB or (resname GLY IGL and name CA)')
    resid = selection_CB.getResindices()
    assert structure.sequence==test_sequence
    assert structure.distance_matrix.shape == (len(structure.sequence),len(structure.sequence))
    assert len(resid)==len(structure.sequence)

def test_structure_segment_class_absolute_indices_no_repair():
    structure=dca_frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/1rnb.pdb',"A",seq_selection="resindex `0to19`",repair_pdb=False)
    test_sequence="QVINTFDGVADYLQTYHKLP"
    selection_CB = structure.structure.select('name CB or (resname GLY IGL and name CA)')
    resid = selection_CB.getResindices()
    assert structure.sequence==test_sequence
    assert structure.distance_matrix.shape == (len(structure.sequence),len(structure.sequence))
    assert len(resid)==len(structure.sequence)

def test_selected_subsequence_AWSEM_contact_energy_matrix():
    structure=dca_frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/4wnc.pdb',"A",seq_selection="resnum 3to26")
    model=dca_frustratometer.AWSEMFrustratometer(structure)
    assert model.potts_model['h'].shape==(24,21)

def test_selected_subsequence_AWSEM_burial_energy_matrix():
    structure=dca_frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/4wnc.pdb',"A",seq_selection="resnum 150to315")
    model=dca_frustratometer.AWSEMFrustratometer(structure)
    assert model.potts_model['J'].shape==(166,166,21,21)

#####
#Test Protein Segment Native AWSEM Energy Calculation
#####

def test_selected_subsequence_AWSEM_burial_energy():
    structure=dca_frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/1MBA_A.pdb',"A",seq_selection="resnum 39to146")
    model=dca_frustratometer.AWSEMFrustratometer(structure)
    selected_region_burial=model.fields_energy()
    # Energy units are in kJ/mol
    assert np.round(selected_region_burial, 2) == -377.95

def test_selected_subsequence_AWSEM_contact_energy():
    structure=dca_frustratometer.Structure.spliced_pdb(f'{_path}/../tests/data/1MBA_A.pdb',"A",seq_selection="resnum 39to146")
    model=dca_frustratometer.AWSEMFrustratometer(structure, distance_cutoff_contact=None)
    selected_region_contact=model.couplings_energy()
    # Energy units are in kJ/mol
    assert np.round(selected_region_contact, 2) == -149.00

def test_scores():
    pdb_file = 'examples/data/1cyo.pdb'
    chain = 'A'
    structure=dca_frustratometer.Structure.full_pdb(pdb_file,chain)
    potts_model_file = 'examples/data/PottsModel1cyoA.mat'
    model = dca_frustratometer.PottsModel.from_potts_model_file(structure, potts_model_file, distance_cutoff=4,
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


def test_compute_mutational_DCA_decoy_energy():
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

def test_single_residue_decoy_AWSEM_energy_statistics():
    #Import Lammps AWSEM Frustratometer single residue frustration values
    lammps_single_frustration_dataframe=pd.read_csv(f"{_path}/../tests/data/6U5E_A_tertiary_frustration_singleresidue_1E8decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    ###
    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../examples/data/6U5E_A.pdb',"A")
    model=dca_frustratometer.AWSEMFrustratometer(structure,distance_cutoff_contact=9.499,
                                                  min_sequence_separation_contact=2)
    #Calculate fields
    seq_index = np.array([_AA.find(aa) for aa in structure.sequence])
    seq_len = len(seq_index)
    h = -model.potts_model['h'][range(seq_len), seq_index]

    #Calculate couplings
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    j = -model.potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * model.mask

    residue_total_energy=(h +j_prime.sum(axis=0))/4.184
    ###
    decoy_fluctuations=(model.decoy_fluctuation(kind='singleresidue'))/4.184
    weighted_decoy_fluctations=(model.aa_freq*decoy_fluctuations).sum(axis=1)/ model.aa_freq.sum()

    expected_mean_decoy_energy=(model.aa_freq*(residue_total_energy[:, np.newaxis]+decoy_fluctuations)).sum(axis=1)/ model.aa_freq.sum()
    expected_std_decoy_energy=np.sqrt(((model.aa_freq * (decoy_fluctuations - weighted_decoy_fluctations[:, np.newaxis]) ** 2) / model.aa_freq.sum()).sum(axis=1))
    
    assert (abs(np.array(lammps_single_frustration_dataframe["<decoy_energies>"])-(expected_mean_decoy_energy)) < 1.2E-1).all()
    assert (abs(np.array(lammps_single_frustration_dataframe["std(decoy_energies)"])-(expected_std_decoy_energy)) < 1.2E-1).all()

def test_contact_pair_decoy_AWSEM_energy_statistics():
    #Import Lammps AWSEM Frustratometer mutational frustration values
    lammps_mutational_frustration_dataframe=pd.read_csv(f"{_path}/../tests/data/6U5E_A_tertiary_frustration_mutational_1E6decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    lammps_mutational_frustration_dataframe["i"]=lammps_mutational_frustration_dataframe["i"]-1
    lammps_mutational_frustration_dataframe["j"]=lammps_mutational_frustration_dataframe["j"]-1
    ###
    structure=dca_frustratometer.Structure.full_pdb(f'{_path}/../examples/data/6U5E_A.pdb',"A")
    model=dca_frustratometer.AWSEMFrustratometer(structure,distance_cutoff_contact=9.5,
                                                  min_sequence_separation_contact=None)
    #Calculate fields
    seq_index = np.array([_AA.find(aa) for aa in structure.sequence])
    seq_len = len(seq_index)
    h = -model.potts_model['h'][range(seq_len), seq_index]

    #Calculate couplings
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    j = -model.potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * model.mask
    test_contact_energy_matrix=h[pos1]+h[pos2]+j_prime.sum(axis=0)[pos1]+j_prime.sum(axis=0)[pos2]-j_prime[pos1,pos2]
    ###
    calculated_mutational_frustration_dataframe=pd.DataFrame(data=test_contact_energy_matrix.ravel(),columns=["Test_Native_Energy"])
    calculated_mutational_frustration_dataframe["Test_Native_Energy"]=calculated_mutational_frustration_dataframe["Test_Native_Energy"]/4.184
    i,j=np.meshgrid(range(0,163),range(0,163), indexing='ij')
    calculated_mutational_frustration_dataframe["i"]=i.ravel()
    calculated_mutational_frustration_dataframe["j"]=j.ravel()
    ###
    decoy_fluctuations=(model.decoy_fluctuation(kind='mutational'))/4.184
    weighted_decoy_fluctations=np.average(decoy_fluctuations.reshape(seq_len * seq_len, 21 * 21), weights=model.contact_freq.flatten(), axis=-1)
    calculated_mutational_frustration_dataframe["Weighted_Decoy_Fluctuations"]=weighted_decoy_fluctations.ravel()
    calculated_mutational_frustration_dataframe["Test_Mean_Decoy_Energy"]=calculated_mutational_frustration_dataframe["Test_Native_Energy"]+calculated_mutational_frustration_dataframe["Weighted_Decoy_Fluctuations"]
    calculated_mutational_frustration_dataframe["STD_Decoy_Energy"]=np.average((decoy_fluctuations.reshape(seq_len * seq_len, 21 * 21)-calculated_mutational_frustration_dataframe["Weighted_Decoy_Fluctuations"][:,np.newaxis]) ** 2,weights=model.contact_freq.flatten(), axis=-1)
    calculated_mutational_frustration_dataframe["STD_Decoy_Energy"]=np.sqrt(calculated_mutational_frustration_dataframe["STD_Decoy_Energy"])
    
    merged_dataframe=calculated_mutational_frustration_dataframe.merge(lammps_mutational_frustration_dataframe,on=["i","j"])

    assert (abs(np.array(merged_dataframe["<decoy_energies>"]-merged_dataframe["Test_Mean_Decoy_Energy"])) < 1.2E-1).all()
    assert (abs(np.array(merged_dataframe["std(decoy_energies)"]-merged_dataframe["STD_Decoy_Energy"])) < 1.2E-1).all()



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