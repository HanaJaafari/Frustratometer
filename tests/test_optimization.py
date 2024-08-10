from frustratometer.optimization import *
from frustratometer.optimization.inner_product import *

def test_energy_term():
    et = EnergyTerm()
    et.test()

##################
# Test functions #
##################

def template_test_compute_region_means(compute_func_numpy, compute_func, setup_func, length_range, atol=1e-8):
    """
    test function to compare the outputs of compute_func_numpy and compute_func.
    Ensures that the computed region means by both functions are equal within a tolerance.
    
    Raises an AssertionError if the computed dictionaries do not match for any key in any iteration.
    """
    for length in length_range:
        print(length)
        indicator_0, indicator_1 = setup_func(length)

        # Compute region means using both functions
        s0 = compute_func_numpy(indicator_0, indicator_1)
        s1 = compute_func(indicator_0, indicator_1)

        # Ensure keys match
        if len(s0) != len(s1):
            raise AssertionError(f"Arrays of different length for length {length}. Numpy: {len(s0)}, Original: {len(s1)}")
        
        # Extract keys and values
        s0_values = s0
        s1_values = s1

        # Find indices where values are not close
        not_close = ~np.isclose(s0_values, s1_values, atol=atol)
        if np.any(not_close):
            s0_diff = s0_values[not_close]
            s1_diff = s1_values[not_close]
            diffs = "\n".join([f"{key}: numpy={s0_val}, original={s1_val}" 
                               for key, s0_val, s1_val in zip(not_close, s0_diff, s1_diff)])
            raise AssertionError(f"Mismatch found in results for length {length}:\n{diffs}")

# Setup functions for each test scenario
def setup_1_by_1(length):
    return np.random.rand(length), np.random.rand(length)

def setup_1_by_2(length):
    indicator_0 = np.random.rand(length)
    indicator_1 = np.random.rand(length, length)
    indicator_1 = (indicator_1 + indicator_1.T) / 2
    return indicator_0, indicator_1

def setup_2_by_2(length):
    indicator_0 = np.random.rand(length, length)
    indicator_1 = np.random.rand(length, length)
    indicator_0 = (indicator_0 + indicator_0.T) / 2
    indicator_1 = (indicator_1 + indicator_1.T) / 2
    return indicator_0, indicator_1

# Specific test functions
def test_compute_region_means_1_by_1():
    template_test_compute_region_means(compute_region_means_1_by_1_numpy, compute_region_means_1_by_1, setup_1_by_1, range(1, 200))

def test_compute_region_means_1_by_2():
    template_test_compute_region_means(compute_region_means_1_by_2_numpy, compute_region_means_1_by_2, setup_1_by_2, range(1, 100))

def test_compute_region_means_2_by_2():
    template_test_compute_region_means(compute_region_means_2_by_2_numpy, compute_region_means_2_by_2, setup_2_by_2, range(1, 40))

def test_compute_region_means_2_by_2_parallel():
    template_test_compute_region_means(compute_region_means_2_by_2, compute_region_means_2_by_2_parallel, setup_2_by_2, range(1, 50))


def test_compute_region_means():
    template_test_compute_region_means(compute_region_means, compute_region_means_1_by_1, setup_1_by_1, range(1, 400,5))
    template_test_compute_region_means(compute_region_means, compute_region_means_1_by_2, setup_1_by_2, range(1, 400,5))
    template_test_compute_region_means(compute_region_means, compute_region_means_2_by_2_parallel, setup_2_by_2, range(1, 300,5))

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
        values_numpy_permutation=permutation_numpy_inner_product(seq,[indicator1D_0, indicator1D_1, indicator1D_0, indicator2D_0, indicator2D_1, indicator2D_0])

        elements = np.unique(seq)
        aa_repetitions = np.array([(seq == k).sum() for k in elements])
        values_numba=build_mean_inner_product_matrix(aa_repetitions,np.array([indicator1D_0, indicator1D_1, indicator1D_0]),np.array([indicator2D_0, indicator2D_1,indicator2D_0]),
                                                     compute_all_region_means(np.array([indicator1D_0, indicator1D_1, indicator1D_0]), np.array([indicator2D_0, indicator2D_1,indicator2D_0])))

        print("type aa_repetitions",type(aa_repetitions))
        print("aa_repetitions.shape",aa_repetitions.shape)
        print("type indicator1D",type([indicator1D_0, indicator1D_1]))
        print("types indicator1D",[type(i) for i in [indicator1D_0, indicator1D_1]])
        print("shapes indicator1D",[i.shape for i in [indicator1D_0, indicator1D_1]])
        print("types indicator2D",[type(i) for i in [indicator2D_0, indicator2D_1]])
        print("shapes indicator2D",[i.shape for i in [indicator2D_0, indicator2D_1]])
        print("type values_numba",type(values_numba))

        assert values_numba.shape == values_numpy_permutation.shape, f"Shapes differ combinatorial_numpy_inner_product shape.shape {values_numpy_permutation.shape} build_mean_inner_product_matrix.shape{values_numba.shape}"
        
        # Compare the results
        if np.allclose(values_numba, values_numpy_permutation):
            print("Results are the same!")
        else:
            print("Results differ!")
            # Optionally, show where they differ
            print("Difference:")
            print(values_numba - values_numpy_permutation)
            raise AssertionError("Results differ!")
        
def test_diff_inner_product(n_elements=20):
    
    #Check that the difference of two build_mean_inner_product_matrix is the same as diff_build_mean_inner_product_matrix
    matrix1d=np.random.rand(n_elements)
    matrix2d=np.random.rand(n_elements,n_elements)
    seq_len=50
    repetitions=np.random.randint(0,n_elements,size=seq_len)
    r0=np.random.randint(0,n_elements)
    r1=np.random.randint(0,n_elements)
    seq_len=repetitions.sum()
    indicator1D_0 = np.random.rand(seq_len)
    indicator1D_1 = np.random.rand(seq_len)
    indicator1D_2 = np.random.rand(seq_len)
    indicator2D_0 = np.random.rand(seq_len, seq_len)
    indicator2D_1 = np.random.rand(seq_len, seq_len)
    indicator2D_2 = np.random.rand(seq_len, seq_len)
    indicator2D_0 = (indicator2D_0 + indicator2D_0.T)/2
    indicator2D_1 = (indicator2D_1 + indicator2D_1.T)/2
    indicator2D_2 = (indicator2D_2 + indicator2D_2.T)/2

    repetitions_new=repetitions.copy()
    if repetitions_new[r0]>0:
        repetitions_new[r0]-=1
        repetitions_new[r1]+=1

    region_means = compute_all_region_means(np.array([indicator1D_0,indicator1D_1,indicator1D_2]), 
                                            np.array([indicator2D_0,indicator2D_1,indicator2D_2]))

    diff_direct=(build_mean_inner_product_matrix(repetitions_new,np.array([indicator1D_0, indicator1D_1, indicator1D_2]),np.array([indicator2D_0, indicator2D_1,indicator2D_2]),region_means) -
                 build_mean_inner_product_matrix(repetitions,np.array([indicator1D_0, indicator1D_1, indicator1D_2]),np.array([indicator2D_0, indicator2D_1,indicator2D_2]),region_means))
    

    diff_indirect = diff_mean_inner_product_matrix(r0,r1,repetitions,
                                                   np.array([indicator1D_0,indicator1D_1,indicator1D_2]), 
                                                   np.array([indicator2D_0,indicator2D_1,indicator2D_2])
                                                   ,region_means)

    if np.allclose(diff_direct, diff_indirect):
        print("Results are the same!")
    else:
        print("Results differ!")
        # Optionally, show where they differ
        print("Difference:")
        print(diff_direct - diff_indirect)
        raise AssertionError("Results differ!")
    
def test_diff_mean_inner_product_2_by_2(n_elements = 10):
    failed=False
    for i in range(10):
        matrix2d_0 = np.random.rand(n_elements, n_elements)
        matrix2d_1 = np.random.rand(n_elements, n_elements)
        repetitions = np.random.randint(0, 1000, size=n_elements)
        r0, r1 = np.random.choice(n_elements, 2, replace=False)
        
        region_mean = compute_region_means_2_by_2(matrix2d_0, matrix2d_1)
        
        # Original function with adjusted repetitions
        m = repetitions.copy()
        n = m.copy()
        n[r0] -= 1
        n[r1] += 1
        result_adjusted = diff_mean_inner_product_2_by_2(r0, r1, repetitions, region_mean)
        
        # Recompute the functions for new and original repetitions directly
        result_new_reps = mean_inner_product_2_by_2(n, region_mean)
        result_original_reps = mean_inner_product_2_by_2(repetitions, region_mean)

        # Check if the results are equivalent
        if np.allclose(result_adjusted, result_new_reps - result_original_reps):
            continue
        else:
            failed=True
            # Optionally, show where they differ
            result_diff=result_adjusted.reshape(n_elements,n_elements,n_elements,n_elements)
            result_main=(result_new_reps - result_original_reps).reshape(n_elements,n_elements,n_elements,n_elements)
            for i in range(n_elements):
                for j in range(n_elements):
                    for k in range(n_elements):
                        for l in range(n_elements):
                            if not np.allclose(result_diff[i,j,k,l],result_main[i,j,k,l]):
                                print(f"i={i},j={j},k={k},l={l},result_diff={result_diff[i,j,k,l]},result_main={result_main[i,j,k,l]}")
                                break

    if failed:
        raise AssertionError("Results differ!")

def test_diff_mean_inner_product_1_by_2(n_elements = 10):
    failed=False
    for i in range(10):
        matrix1d_0 = np.random.rand(n_elements)
        matrix2d_1 = np.random.rand(n_elements, n_elements)
        repetitions = np.random.randint(0, 1000, size=n_elements)
        r0, r1 = np.random.choice(n_elements, 2, replace=False)
        
        region_mean = compute_region_means_1_by_2(matrix1d_0, matrix2d_1)
        
        # Original function with adjusted repetitions
        m = repetitions.copy()
        n = m.copy()
        n[r0] -= 1
        n[r1] += 1
        result_adjusted = diff_mean_inner_product_1_by_2(r0, r1, repetitions, region_mean)
        
        # Recompute the functions for new and original repetitions directly
        result_new_reps = mean_inner_product_1_by_2(n, region_mean)
        result_original_reps = mean_inner_product_1_by_2(repetitions, region_mean)

        # Check if the results are equivalent
        if np.allclose(result_adjusted, result_new_reps - result_original_reps):
            continue
        else:
            failed=True
            # Optionally, show where they differ
            result_diff=result_adjusted.reshape(n_elements,n_elements,n_elements)
            result_main=(result_new_reps - result_original_reps).reshape(n_elements,n_elements,n_elements)
            for i in range(n_elements):
                for j in range(n_elements):
                    for k in range(n_elements):
                        if not np.allclose(result_diff[i,j,k],result_main[i,j,k]):
                            print(f"i={i},j={j},k={k},result_diff={result_diff[i,j,k]},result_main={result_main[i,j,k]}")
                            break

    if failed:
        raise AssertionError("Results differ!")
    
def test_diff_mean_inner_product_1_by_1(n_elements = 10):
    failed=False
    for i in range(10):
        matrix1d_0 = np.random.rand(n_elements)
        matrix1d_1 = np.random.rand(n_elements)
        repetitions = np.random.randint(0, 1000, size=n_elements)
        r0, r1 = np.random.choice(n_elements, 2, replace=False)
        
        region_mean = compute_region_means_1_by_1(matrix1d_0, matrix1d_1)
        
        # Original function with adjusted repetitions
        m = repetitions.copy()
        n = m.copy()
        n[r0] -= 1
        n[r1] += 1
        result_adjusted = diff_mean_inner_product_1_by_1(r0, r1, repetitions, region_mean)
        
        # Recompute the functions for new and original repetitions directly
        result_new_reps = mean_inner_product_1_by_1(n, region_mean)
        result_original_reps = mean_inner_product_1_by_1(repetitions, region_mean)

        # Check if the results are equivalent
        if np.allclose(result_adjusted, result_new_reps - result_original_reps):
            continue
        else:
            failed=True
            # Optionally, show where they differ
            result_diff=result_adjusted.reshape(n_elements,n_elements)
            result_main=(result_new_reps - result_original_reps).reshape(n_elements,n_elements)
            for i in range(n_elements):
                for j in range(n_elements):
                    if not np.allclose(result_diff[i,j],result_main[i,j]):
                        print(f"i={i},j={j},result_diff={result_diff[i,j]},result_main={result_main[i,j]}")
                        break

    if failed:
        raise AssertionError("Results differ!")

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