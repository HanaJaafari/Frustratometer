from frustratometer.optimization import *

def test_energy_term():
    et = EnergyTerm()
    et.test()


def test_heterogeneity_approximation():
    sequence = list("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLP")
    het = heterogeneity(sequence)
    het_approx = heterogeneity_approximation(optimization.sequence_to_index(sequence))
    assert np.isclose(het, het_approx), f"Heterogeneity: {het}, Approximation: {het_approx}"

def test_heterogeneity_difference_permutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_swap(seq_index, model.potts_model['h'], model.potts_model['J'], model.mask)
    het = heterogeneity_approximation(seq_index)
    new_het = heterogeneity_approximation(new_sequence)
    het_difference2 = new_het - het
    assert np.isclose(het_difference, het_difference2), f"Heterogeneity difference: {het_difference}, {het_difference2}"

def test_heterogeneity_difference_mutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_mutation(seq_index, model.potts_model['h'], model.potts_model['J'],model.mask)
    het = heterogeneity_approximation(seq_index)
    new_het = heterogeneity_approximation(new_sequence)
    het_difference2 = new_het - het
    assert np.isclose(het_difference, het_difference2), f"Heterogeneity difference: {het_difference}, {het_difference2}"

def test_energy_difference_permutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_swap(seq_index, model.potts_model['h'], model.potts_model['J'],model.mask)
    energy = model_energy(seq_index, model.potts_model['h'], model.potts_model['J'],model.mask)
    new_energy = model_energy(new_sequence, model.potts_model['h'], model.potts_model['J'],model.mask)
    energy_difference2 = new_energy - energy
    assert np.isclose(energy_difference, energy_difference2), f"Energy difference: {energy_difference}, {energy_difference2}"

def test_energy_difference_mutation():
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index(model.sequence)
    new_sequence, het_difference, energy_difference = sequence_mutation(seq_index, model.potts_model['h'], model.potts_model['J'],model.mask)
    energy = model_energy(seq_index, model.potts_model['h'], model.potts_model['J'],model.mask)
    new_energy = model_energy(new_sequence, model.potts_model['h'], model.potts_model['J'],model.mask)
    energy_difference2 = new_energy - energy
    assert np.isclose(energy_difference, energy_difference2), f"Energy difference: {energy_difference}, {energy_difference2}"

def test_mean_inner_product():
    native_sequences=[
        [0,1],
        [0,1,1],
        [0,1,1,1],
        [0,1,2],
        [0,1,2,2],
        [0,1,1,2,2],
        [0,1,1,2,2,2],
        [0,1,2,3,4],
        [0,1,2,3,4,5],
        [0,1,1,2,3,4],
        [0,0,1,1,2,2],
        [0,1,1,1,1],
        [0,1,1,1,1,1],
        [0,1,1,1,1,1,1],
        [0,0,1,1],
        [0,0,1,1,1],
        [0,0,0,1,1,1],
        [0,1,2,3],
        [0,0,1,2,3],
        [0,0,0,1,2,3],
        [0,0,0,1,1,2,3], 
        [0,0,0,0,1,2,3],
    ]

    # Test each case
    for i, seq in enumerate(native_sequences):
        print(f"Testing case {i+1}/{len(native_sequences)}: {seq}")
        indicator1D_0 = np.random.rand(len(seq))
        indicator1D_1 = np.random.rand(len(seq))
        indicator2D_0 = np.random.rand(len(seq), len(seq))
        indicator2D_1 = np.random.rand(len(seq), len(seq))
        indicator2D_0=(indicator2D_0+indicator2D_0.T)/2
        indicator2D_1=(indicator2D_1+indicator2D_1.T)/2
        values_numpy_permutation=permutation_numpy_inner_product(seq,[indicator1D_0, indicator1D_1, indicator2D_0, indicator2D_1])

        elements = np.unique(seq)
        aa_repetitions = np.array([(seq == k).sum() for k in elements])
        values_numba=build_mean_inner_product_matrix(aa_repetitions,[indicator1D_0, indicator1D_1], [indicator2D_0, indicator2D_1])

        assert values_numba.shape == values_numpy_permutation.shape, f"Shapes differ combinatorial_numpy_inner_product shape.shape {values_numpy_permutation.shape} build_mean_inner_product_matrix.shape{values_numba.shape}"
        
        # Compare the results
        if np.allclose(values_numba, values_numpy_permutation):
            print("Results are the same!")
        else:
            print("Results differ!")
            # Optionally, show where they differ
            print("Difference:")
            print(values_numba - values_numpy_permutation)
            break