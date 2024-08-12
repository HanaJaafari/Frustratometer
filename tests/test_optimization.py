import pytest
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

################
# Energy Terms #
################

_AA = '-ACDEFGHIKLMNPQRSTVWY'

@pytest.fixture(params=[(10, 2, 0.0), (10, 2, 4.15), (None, 10, 4.15)])
@pytest.mark.parametrize(["distance_cutoff_contact", "min_sequence_separation_contact", "k_electrostatics"], [])
def model(request):
    native_pdb = "tests/data/1bfz.pdb"
    distance_cutoff_contact, min_sequence_separation_contact, k_electrostatics = request.param
    structure = Structure.full_pdb(native_pdb, "A")
    model = AWSEM(structure, distance_cutoff_contact=distance_cutoff_contact, min_sequence_separation_contact=min_sequence_separation_contact, expose_indicator_functions=True, k_electrostatics=k_electrostatics)   
    return model

@pytest.mark.parametrize("reduced_alphabet", [_AA,''.join([a for a in _AA if a not in ['-','C','P']]),''.join([a for a in _AA if a != '-'] + ['-'])])
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("use_numba", [True, False])
def test_heterogeneity(reduced_alphabet, exact, use_numba):
    seq_indices = np.random.randint(0, len(reduced_alphabet), size=(1,5))
    het=Heterogeneity(exact=exact,use_numba=use_numba,alphabet=reduced_alphabet)
    het.test(seq_indices[0])

@pytest.mark.parametrize("reduced_alphabet", [_AA,''.join([a for a in _AA if a not in ['-','C','P']]),''.join([a for a in _AA if a != '-'] + ['-'])])
@pytest.mark.parametrize("use_numba", [True, False])
def test_awsem_energy(model,reduced_alphabet,use_numba):
    seq_indices = np.random.randint(0, len(reduced_alphabet), size=(1,len(model.sequence)))
    awsem_energy = AwsemEnergy(use_numba=use_numba, model=model, alphabet=reduced_alphabet)
    awsem_energy.test(seq_indices[0])
    awsem_energy.regression_test()

@pytest.mark.parametrize("reduced_alphabet", [_AA,''.join([a for a in _AA if a not in ['-','C','P']]),''.join([a for a in _AA if a != '-'] + ['-'])])
@pytest.mark.parametrize("use_numba", [True, False])
def test_awsem_energy_variance(model, reduced_alphabet, use_numba):
    seq_indices = np.random.randint(0, len(reduced_alphabet), size=(1,len(model.sequence)))
    awsem_de2 = AwsemEnergyVariance(use_numba=use_numba, model=model, alphabet=reduced_alphabet)
    awsem_de2.test(seq_indices[0])
    awsem_de2.regression_test(seq_indices[0])


def test_awsem_energy_variance_sample(model):
    seq_index=sequence_to_index(model.sequence,alphabet=_AA)
    
    def compute_energy_variance_sample(seq_index,n_decoys=10000):
        energies=[]
        shuffled_index=seq_index.copy()
        for i in range(n_decoys):
            np.random.shuffle(shuffled_index)
            energies.append(model.native_energy(index_to_sequence(shuffled_index,alphabet=_AA)))
            #if i%(n_decoys//100)==0:
                #energies_array=np.array(energies)
                #print(i,energies_array.mean(),energies_array.var())
        #Split the energies into 10 groups and compute the variance of each group to get an error estimate
        energies_array=np.array(energies)
        energies_array=energies_array.reshape(10,-1)
        energy_variances=np.var(energies_array,axis=1)
        mean_variance=energy_variances.mean()
        error_variance=energy_variances.std()
        print(f"Decoy Variance: {mean_variance} +/- {3*error_variance}") #3 sigma error
        print(f"Expected variance: {awsem_de2.compute_energy(seq_index)}")
        return np.var(energies), awsem_de2.compute_energy(seq_index)
    
    print(compute_energy_variance_sample(seq_index))

    def compute_energy_variance_permutation(seq_index):
        from itertools import permutations
        decoy_sequences = np.array(list(permutations(seq_index)))
        energies=[]
        for seq in decoy_sequences:
            energies.append(model.native_energy(index_to_sequence(seq,alphabet=_AA)))

        print(f"Decoy Variance: {np.var(energies)}") # Exact variance
        print(f"Expected variance: {awsem_de2.compute_energy(seq_index)}")
        return np.var(energies), awsem_de2.compute_energy(seq_index)
    
    print(compute_energy_variance_permutation(seq_index))

    from itertools import permutations
    decoy_sequences = np.array(list(permutations(seq_index)))
    indicators1D=np.array(model.indicators[:3])
    indicators2D=np.array(model.indicators[3:])
    indicator_arrays=[]
    energies=[]
    for decoy_index in decoy_sequences:
        ind1D=np.zeros((len(indicators1D),21))
        for i in range(len(ind1D)):
            ind1D[i] = np.bincount(decoy_index, weights=indicators1D[i], minlength=21)

        decoy_index2D=decoy_index[np.newaxis,:]*21+decoy_index[:,np.newaxis]
        ind2D=np.zeros((len(indicators2D),21*21))
        for i in range(len(ind2D)):
            ind2D[i] =np.bincount(decoy_index2D.ravel(), weights=indicators2D[i].ravel(), minlength=21*21)

        indicator_array = np.concatenate([ind1D.ravel(),ind2D.ravel()])
        gamma_array = np.concatenate([a.ravel() for a in model.gamma_array])

        energy_i = gamma_array @ indicator_array
        assert np.isclose(model.native_energy(index_to_sequence(decoy_index,alphabet=_AA)),energy_i), f"Expected energy {model.native_energy(index_to_sequence(decoy_index,alphabet=_AA))} but got {energy_i}"
        energies.append(energy_i)
        indicator_arrays.append(indicator_array)

    indicator_arrays = np.array(indicator_arrays)
    energies = np.array(energies)
    assert np.isclose(gamma_array@indicator_arrays.mean(axis=0),energies.mean()), f"Expected mean energy {gamma_array@indicator_arrays.mean(axis=0)} but got {np.mean(energies)}"

    # I will code something like this using numpy einsums:
    # np.array([[np.outer(indicator_arrays[:,i],indicator_arrays[:,j]).mean() - indicator_arrays[:,i].mean()*indicator_arrays[:,i].mean() for i in range(indicator_arrays.shape[1])] for j in range(indicator_arrays.shape[1])])
    outer_product = np.einsum('ij,ik->ijk', indicator_arrays, indicator_arrays)
    mean_outer_product = outer_product.mean(axis=0)
    mean_outer_product -= np.outer(indicator_arrays.mean(axis=0), indicator_arrays.mean(axis=0))
    assert np.allclose(gamma_array @ mean_outer_product @ gamma_array, energies.var()), "Covariance matrix is not correct"

    # Indicator tests    
    indicators1D=np.array(model.indicators[0:3])
    indicators2D=np.array(model.indicators[3:])
    gamma=model.gamma_array
    true_indicator1D=np.array([indicators1D[:,model_seq_index==i].sum(axis=1) for i in range(21)]).T
    true_indicator2D=np.array([indicators2D[:,model_seq_index==i][:,:, model_seq_index==j].sum(axis=(1,2)) for i in range(21) for j in range(21)]).reshape(21,21,3).T
    true_indicator=np.concatenate([true_indicator1D.ravel(),true_indicator2D.ravel()])
    burial_gamma=np.concatenate(model.gamma_array[:3])
    burial_energy_predicted = (burial_gamma * np.concatenate(true_indicator1D)).sum()
    burial_energy_expected = -model.potts_model['h'][range(len(model_seq_index)), model_seq_index].sum()
    assert np.isclose(burial_energy_predicted,burial_energy_expected), f"Expected energy {burial_energy_expected} but got {burial_energy_predicted}"
    contact_gamma=np.concatenate([a.ravel() for a in model.gamma_array[3:]])
    contact_energy_predicted = (contact_gamma * np.concatenate([a.ravel() for a in true_indicator2D])).sum()
    contact_energy_expected = model.couplings_energy()
    assert np.isclose(contact_energy_predicted,contact_energy_expected), f"Expected energy {contact_energy_expected} but got {contact_energy_predicted}"



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