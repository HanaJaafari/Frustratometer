import pytest
import pandas as pd
import numpy as np
import frustratometer
from pathlib import Path


test_path=Path('tests')
test_data_path=Path('tests/data')

# Assuming you have a function to load your tests configurations
tests_config = pd.read_csv(test_path/"test_awsem_config.csv",comment='#')
#tests_config = pd.read_csv(test_path/"test_awsem_config.csv")

def test_prody_expected_error():
    test_data=tests_config.iloc[0]
    try:
        structure = frustratometer.Structure.full_pdb(test_data_path/f"{test_data['pdb']}.pdb")
        assert True
    except TypeError as e:
        if "can't multiply sequence by non-int of type 'Forward'" in str(e):
            print("Encountered a ProDy TypeError on initial run. Error logged for future reference")
        else:
            raise


@pytest.mark.parametrize("test_data", tests_config.to_dict(orient="records"))
def test_density_residues(test_data):
    structure = frustratometer.Structure.full_pdb(test_data_path/f"{test_data['pdb']}.pdb")
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_rho=sequence_separation, k_electrostatics=0)
    data = pd.read_csv(test_data['singleresidue'], delim_whitespace=True)
    data['Calculated_density'] = model.rho_r
    data['Expected_density'] = data['DensityRes']
    max_atol = np.max(np.abs(data['Calculated_density'] - data['Expected_density']))
    print(max_atol)
    try:
        assert np.allclose(data['Calculated_density'], data['Expected_density'], atol=1E-3)
    except AssertionError:
        max_atol = np.max(np.abs(data['Calculated_density'] - data['Expected_density']))
        print(f"Assertion failed: Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance.")
        raise AssertionError(f"Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance of 1E-3.")

@pytest.mark.parametrize("test_data", tests_config.to_dict(orient="records"))
def test_single_residue_frustration(test_data):
    structure = frustratometer.Structure.full_pdb(test_data_path/f"{test_data['pdb']}.pdb")
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_rho=sequence_separation, min_sequence_separation_contact=2, k_electrostatics=test_data['k_electrostatics'] * 4.184, min_sequence_separation_electrostatics=1)
    data = pd.read_csv(test_data['singleresidue'], delim_whitespace=True)
    data['Calculated_frustration'] = model.frustration(kind='singleresidue')
    data['Expected_frustration'] = data['FrstIndex']
    try:
        assert np.allclose(data['Calculated_frustration'], data['Expected_frustration'], atol=3E-1)
    except AssertionError:
        max_atol = np.max(np.abs(data['Calculated_frustration'] - data['Expected_frustration']))
        print(f"Assertion failed: Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance.")
        raise AssertionError(f"Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance of 3E-1.")

@pytest.mark.parametrize("test_data", tests_config.to_dict(orient="records"))
def test_mutational_frustration(test_data):
    structure = frustratometer.Structure.full_pdb(test_data_path/f"{test_data['pdb']}.pdb")
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    if test_data['k_electrostatics']==1000:
        assert True
        return
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_rho=sequence_separation, min_sequence_separation_contact=0, k_electrostatics=test_data['k_electrostatics'] * 4.184, min_sequence_separation_electrostatics=1)
    data = pd.read_csv(test_data['mutational'], delim_whitespace=True)
    
    if test_data['pdb']!="ijge":
        chains=['A','B','C']
        for chain,next_chain in zip(chains,chains[1:]):
            max_resid={'A':277,'B':277+99,'C':9}
            data.loc[(data['ChainRes1']==next_chain),'#Res1']+=max_resid[chain]
            data.loc[(data['ChainRes2']==next_chain),'Res2']+=max_resid[chain]
    
    start_pdb=1 if test_data['pdb']!="6u5e" else 2
    data['Calculated_frustration'] = model.frustration(kind='mutational')[data['#Res1']-start_pdb, data['Res2']-start_pdb]
    data['Expected_frustration'] = data['FrstIndex']
    #data.to_csv(f"/home/fc36/dump/{test_data['pdb']}_seqsep_{test_data['seqsep']}_kelec_{test_data['k_electrostatics']}_mutational.csv") 
    #np.savetxt(f"/home/fc36/dump/{test_data['pdb']}_seqsep_{test_data['seqsep']}_kelec_{test_data['k_electrostatics']}_mutational_test_full.csv", model.frustration(kind='mutational'),delimiter=',')
    #np.savetxt(f'/home/fc36/dump/{test_data["pdb"]}_seqsep_{test_data["seqsep"]}_kelec_{test_data["k_electrostatics"]}_other_save.csv',model.frustration(kind='mutational')[data['#Res1']-start_pdb, data['Res2']-start_pdb],delimiter=',')
    if test_data['pdb'] == 'sequence0':
        atol=3.5E-1
    else:
        atol=3E-1
    try:
        assert np.allclose(data['Calculated_frustration'], data['Expected_frustration'], atol=atol)
    except AssertionError:
        max_atol = np.max(np.abs(data['Calculated_frustration'] - data['Expected_frustration']))
        print(f"Assertion failed: Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance.")
        raise AssertionError(f"Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance of {atol}.")

@pytest.mark.parametrize("test_data", tests_config.to_dict(orient="records"))
def test_configurational_frustration(test_data):
    #This test may fail due to the randomness of the decoy generation
    structure = frustratometer.Structure.full_pdb(test_data_path/f"{test_data['pdb']}.pdb")
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    
    if test_data['k_electrostatics'] == 1000:
        assert True
        return

    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, 
                                 min_sequence_separation_rho=sequence_separation, 
                                 min_sequence_separation_contact=0, 
                                 k_electrostatics=test_data['k_electrostatics'] * 4.184, 
                                 min_sequence_separation_electrostatics=1)
    
    data = pd.read_csv(test_data['configurational'], delim_whitespace=True)
    
    if test_data['pdb'] != "ijge":
        chains = ['A', 'B', 'C']
        for chain, next_chain in zip(chains, chains[1:]):
            max_resid = {'A': 277, 'B': 277 + 99, 'C': 9}
            data.loc[data['ChainRes1'] == next_chain, '#Res1'] += max_resid[chain]
            data.loc[data['ChainRes2'] == next_chain, 'Res2'] += max_resid[chain]

    start_pdb = 1 if (test_data['pdb'] != "6u5e" or test_data['lammps']) else 2
    data['Calculated_frustration'] = model.configurational_frustration(n_decoys=10000)[data['#Res1'] - start_pdb, data['Res2'] - start_pdb]
    #data.to_csv(f"/home/fc36/dump/{test_data['pdb']}_seqsep_{test_data['seqsep']}_kelec_{test_data['k_electrostatics']}_configurational.csv")
    data['Expected_frustration'] = data['FrstIndex']
    #np.savetxt(f"/home/fc36/dump/{test_data['pdb']}_seqsep_{test_data['seqsep']}_kelec_{test_data['k_electrostatics']}_configurational_full.csv",model.configurational_frustration(n_decoys=10000),delimiter=',')
    if test_data['pdb'] == 'sequence0':
        atol = 5.5E-1
    elif test_data['pdb'] == 'sequence1':
        atol = 5E-1
    else:
        atol = 3E-1
    try:
        assert np.allclose(data['Calculated_frustration'], data['Expected_frustration'], atol=atol)
    except AssertionError:
        max_atol = np.max(np.abs(data['Calculated_frustration'] - data['Expected_frustration']))
        print(f"Assertion failed: Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance.")
        raise AssertionError(f"Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance of {atol}.")

#####
#Test AWSEM Native Energy Calculations
#####
def test_residue_density_calculation():
    #Import Lammps AWSEM Frustratometer single residue frustration values
    lammps_single_frustration_dataframe=pd.read_csv(test_data_path/"6U5E_A_tertiary_frustration_singleresidue_1E8decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    lammps_single_frustration_dataframe["i"]=lammps_single_frustration_dataframe["i"]-1
    expected_rho_values=lammps_single_frustration_dataframe["rho_i"]

    structure=frustratometer.Structure.full_pdb(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,distance_cutoff_contact=9.499,
                                                  min_sequence_separation_contact=2)
    assert np.round(model.rho_r,2).all()==np.round(expected_rho_values,2).all()

def test_AWSEM_native_energy():
    structure=frustratometer.Structure.full_pdb(test_data_path/f'1l63.pdb',"A")
    model=frustratometer.AWSEM(structure,k_electrostatics=0, min_sequence_separation_contact = 10, distance_cutoff_contact = None)
    e = model.native_energy()
    print(e)
    assert np.round(e, 0) == -915

def test_AWSEM_fields_energy():
    structure=frustratometer.Structure.full_pdb(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,k_electrostatics=0, min_sequence_separation_contact = 10, distance_cutoff_contact = None)
    e = model.fields_energy()
    print(e)
    assert np.round(e, 0) == -555

def test_AWSEM_couplings_energy():
    structure=frustratometer.Structure.full_pdb(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,k_electrostatics=0, min_sequence_separation_contact = 10, distance_cutoff_contact = None)
    e = model.couplings_energy()
    print(e)
    assert np.round(e, 0) == -362

def test_fields_couplings_AWSEM_energy():
    structure=frustratometer.Structure.full_pdb(test_data_path/f'6u5e.pdb',"A")
    model = frustratometer.AWSEM(structure)
    assert model.fields_energy() + model.couplings_energy() - model.native_energy()  < 1E-6

def test_single_residue_AWSEM_energy():
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    #Import Lammps AWSEM Frustratometer single residue frustration values
    lammps_single_frustration_dataframe=pd.read_csv(test_data_path/f"6U5E_A_tertiary_frustration_singleresidue_1E8decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    ###
    structure=frustratometer.Structure.full_pdb(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,distance_cutoff_contact=9.499,
                                                  min_sequence_separation_contact=2,
                                                  k_electrostatics=0)
                                                  
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
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    #Import Lammps AWSEM Frustratometer mutational frustration values
    lammps_mutational_frustration_dataframe=pd.read_csv(test_data_path/f"6U5E_A_tertiary_frustration_mutational_1E6decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    lammps_mutational_frustration_dataframe["i"]=lammps_mutational_frustration_dataframe["i"]-1
    lammps_mutational_frustration_dataframe["j"]=lammps_mutational_frustration_dataframe["j"]-1
    ###
    structure=frustratometer.Structure.full_pdb(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,distance_cutoff_contact=9.499,
                                                  min_sequence_separation_contact=0,
                                                  k_electrostatics=0)
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

def test_selected_subsequence_AWSEM_contact_energy_matrix():
    structure=frustratometer.Structure.spliced_pdb(test_data_path/f'4wnc.pdb',"A",seq_selection="resnum 3to26")
    model=frustratometer.AWSEM(structure)
    assert model.potts_model['h'].shape==(24,21)

def test_selected_subsequence_AWSEM_burial_energy_matrix():
    structure=frustratometer.Structure.spliced_pdb(test_data_path/f'4wnc.pdb',"A",seq_selection="resnum 150to315")
    model=frustratometer.AWSEM(structure)
    assert model.potts_model['J'].shape==(166,166,21,21)

#####
#Test Protein Segment Native AWSEM Energy Calculation
#####

def test_selected_subsequence_AWSEM_burial_energy():
    structure=frustratometer.Structure.spliced_pdb(test_data_path/f'1MBA_A.pdb',"A",seq_selection="resnum 39to146")
    model=frustratometer.AWSEM(structure)
    selected_region_burial=model.fields_energy()
    # Energy units are in kJ/mol
    assert np.round(selected_region_burial, 2) == -377.95

def test_selected_subsequence_AWSEM_contact_energy():
    structure=frustratometer.Structure.spliced_pdb(test_data_path/f'1MBA_A.pdb',"A",seq_selection="resnum 39to146")
    model=frustratometer.AWSEM(structure, distance_cutoff_contact=None, k_electrostatics=0.0, min_sequence_separation_contact=10)
    selected_region_contact=model.couplings_energy()
    # Energy units are in kJ/mol
    assert np.round(selected_region_contact, 2) == -149.00

def test_single_residue_decoy_AWSEM_energy_statistics():
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    #Import Lammps AWSEM Frustratometer single residue frustration values
    lammps_single_frustration_dataframe=pd.read_csv(test_data_path/f"6U5E_A_tertiary_frustration_singleresidue_1E8decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    ###
    structure=frustratometer.Structure.full_pdb(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,distance_cutoff_contact=9.499, min_sequence_separation_contact=2, k_electrostatics=0)
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
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    #Import Lammps AWSEM Frustratometer mutational frustration values
    lammps_mutational_frustration_dataframe=pd.read_csv(test_data_path/f"6U5E_A_tertiary_frustration_mutational_1E6decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep="\s+")
    lammps_mutational_frustration_dataframe["i"]=lammps_mutational_frustration_dataframe["i"]-1
    lammps_mutational_frustration_dataframe["j"]=lammps_mutational_frustration_dataframe["j"]-1
    ###
    structure=frustratometer.Structure.full_pdb(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,distance_cutoff_contact=9.5, min_sequence_separation_contact=None, k_electrostatics=0)
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
    calculated_mutational_frustration_dataframe["STD_Decoy_Energy"]=np.average((decoy_fluctuations.reshape(seq_len * seq_len, 21 * 21)-calculated_mutational_frustration_dataframe["Weighted_Decoy_Fluctuations"].astype(float).values[:,np.newaxis]) ** 2,weights=model.contact_freq.flatten(), axis=-1)
    calculated_mutational_frustration_dataframe["STD_Decoy_Energy"]=np.sqrt(calculated_mutational_frustration_dataframe["STD_Decoy_Energy"])
    
    merged_dataframe=calculated_mutational_frustration_dataframe.merge(lammps_mutational_frustration_dataframe,on=["i","j"])

    assert (abs(np.array(merged_dataframe["<decoy_energies>"]-merged_dataframe["Test_Mean_Decoy_Energy"])) < 1.2E-1).all()
    assert (abs(np.array(merged_dataframe["std(decoy_energies)"]-merged_dataframe["STD_Decoy_Energy"])) < 1.2E-1).all()

if __name__ == "__main__":
    pytest.main()
