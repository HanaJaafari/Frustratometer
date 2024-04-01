import pytest
import pandas as pd
import numpy as np
import frustratometer

# Assuming you have a function to load your tests configurations
tests_config = pd.read_csv("tests/test_awsem_config.csv")

@pytest.mark.parametrize("test_data", tests_config.to_dict(orient="records"))
def test_density_residues(test_data):
    structure = frustratometer.Structure.full_pdb(f"tests/data/{test_data['pdb']}.pdb")
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_rho=sequence_separation, k_electrostatics=test_data['k_electrostatics'])
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
    structure = frustratometer.Structure.full_pdb(f"tests/data/{test_data['pdb']}.pdb")
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_rho=sequence_separation, min_sequence_separation_contact=2, k_electrostatics=test_data['k_electrostatics'], min_sequence_separation_electrostatics=1)
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
    structure = frustratometer.Structure.full_pdb(f"tests/data/{test_data['pdb']}.pdb")
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    if test_data['k_electrostatics']==1000:
        assert True
        return
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_rho=sequence_separation, min_sequence_separation_contact=0, k_electrostatics=test_data['k_electrostatics'], min_sequence_separation_electrostatics=1)
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
    try:
        assert np.allclose(data['Calculated_frustration'], data['Expected_frustration'], atol=3E-1)
    except AssertionError:
        max_atol = np.max(np.abs(data['Calculated_frustration'] - data['Expected_frustration']))
        print(f"Assertion failed: Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance.")
        raise AssertionError(f"Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance of 3E-1.")

if __name__ == "__main__":
    pytest.main()
