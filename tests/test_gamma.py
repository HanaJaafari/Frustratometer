import unittest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import frustratometer

#Tests
# Initialization Tests

# Test initializing the Gamma class with a numpy array and validate all attributes are set correctly.
# Test initializing from an existing Gamma instance (from_instance class method) and ensure it's a deep copy.
# Test initializing from a file (from_file class method) with correct and incorrect file paths.
# Test initializing with custom segment_definition and alphabet, including edge cases like empty or incorrect formats.
# Segment Handling Tests

class TestGammaInitialization(unittest.TestCase):

    def setUp(self):
        # Setup reusable components for the tests
        self.gamma_array = np.random.random(2*5+2*5*5)
        self.segment_definition = {
            '1D': (2, 5),  # Simplified for testing
            '2D': (2, 5,5)  # Simplified for testing
        }
        self.description = "Test Gamma"
        self.alphabet = ['A', 'B', 'C', 'D', 'E']  # Simplified for testing

    def test_init_from_array(self):
        gamma = frustratometer.Gamma(self.gamma_array, segment_definition=self.segment_definition, description=self.description, alphabet=self.alphabet)
        self.assertTrue(np.array_equal(gamma.gamma_array, self.gamma_array))
        self.assertEqual(gamma.description, self.description)
        self.assertEqual(gamma.segment_definition, self.segment_definition)
        self.assertEqual(gamma.alphabet, self.alphabet)

    def test_init_from_instance(self):
        gamma = frustratometer.Gamma(self.gamma_array, segment_definition=self.segment_definition, description=self.description, alphabet=self.alphabet)
        gamma_copy = frustratometer.Gamma.from_instance(gamma)
        self.assertTrue(np.array_equal(gamma_copy.gamma_array, gamma.gamma_array))
        self.assertIsNot(gamma_copy, gamma)
        self.assertIsNot(gamma_copy.gamma_array, gamma.gamma_array)

    def test_init_from_file(self):
        # Create a temporary file to simulate reading from a file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        data = {
            "gamma_array": self.gamma_array.tolist(),
            "description": self.description,
            "segment_definition": self.segment_definition,
            "alphabet": self.alphabet
        }
        with open(temp_file.name, 'w') as f:
            json.dump(data, f)

        gamma = frustratometer.Gamma.from_file(temp_file.name)
        self.assertTrue(np.array_equal(gamma.gamma_array, self.gamma_array))
        self.assertEqual(gamma.description, self.description)
        self.assertEqual(gamma.segment_definition, self.segment_definition)
        self.assertEqual(gamma.alphabet, self.alphabet)

        # Clean up the temporary file
        os.unlink(temp_file.name)

    def test_init_with_default_segment_definition_and_alphabet(self):
        gamma = frustratometer.Gamma(np.random.rand(1260))
        self.assertEqual(len(gamma.gamma_array), 1260)
        self.assertEqual(gamma.segment_definition, frustratometer.Gamma.default_segment_definition)
        self.assertEqual(gamma.alphabet, frustratometer.Gamma.default_alphabet)
    
    def test_init_with_incorrect_types(self):
        with self.assertRaises(TypeError):
            frustratometer.Gamma(123)  # Passing an unsupported type should raise TypeError

    def test_init_from_file_nonexistent(self):
        """Test initialization from a nonexistent file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            frustratometer.Gamma.from_file("nonexistent_file.json")

class TestGammaAWSEMFiles(unittest.TestCase):

    def test_load_from_awsem_files(self):
       
        # Load from AWSEM files
        gamma_loaded = frustratometer.Gamma.from_awsem_files()
        
        # Perform checks to verify loading was successful
        self.assertIsNotNone(gamma_loaded.gamma_array)
        # Additional assertions can be added based on expected data properties

    def test_save_to_awsem_files(self):
        # Setup a Gamma instance
        gamma_original = frustratometer.Gamma(np.random.random(1260))  # Example size, adjust accordingly
        
        # Use temporary files for saving
        with tempfile.NamedTemporaryFile(delete=False) as burial_temp, tempfile.NamedTemporaryFile(delete=False) as gamma_dat_temp:
            burial_gamma_filepath = Path(burial_temp.name)
            gamma_dat_filepath = Path(gamma_dat_temp.name)
            
            gamma_original.to_awsem_files(str(burial_gamma_filepath), str(gamma_dat_filepath))
            
            # Verify that the files exist
            self.assertTrue(burial_gamma_filepath.exists())
            self.assertTrue(gamma_dat_filepath.exists())
            
            # Clean up the temporary files
            burial_gamma_filepath.unlink()
            gamma_dat_filepath.unlink()

    def test_save_and_load_round_trip(self):
        # Setup a Gamma instance
        gamma_original = frustratometer.Gamma(np.arange(1260))  # Example size, adjust accordingly
        gamma_original = gamma_original.symmetrize()

        # Use temporary files for saving and loading
        with tempfile.NamedTemporaryFile(delete=False) as burial_temp, tempfile.NamedTemporaryFile(delete=False) as gamma_dat_temp:
            burial_gamma_filepath = Path(burial_temp.name)
            gamma_dat_filepath = Path(gamma_dat_temp.name)

            gamma_original.to_awsem_files(str(burial_gamma_filepath), str(gamma_dat_filepath))

            # Load from AWSEM files
            gamma_loaded = frustratometer.Gamma.from_awsem_files(str(burial_gamma_filepath), str(gamma_dat_filepath))

            # Test equality of data
            if not np.array_equal(gamma_original.gamma_array, gamma_loaded.gamma_array):
                # Find the first 10 elements that are different
                diff_indices = np.where(gamma_original.gamma_array != gamma_loaded.gamma_array)[0][:]
                diff_elements0 = gamma_original.gamma_array[diff_indices]
                diff_elements1 = gamma_loaded.gamma_array[diff_indices]
                diff_indices_str = ', '.join(str(idx) for idx in diff_indices)
                diff_elements_str0 = ', '.join(str(elem) for elem in diff_elements0)
                diff_elements_str1 = ', '.join(str(elem) for elem in diff_elements1)
                message = f"Arrays are different. First 10 different elements: \nGamma_original:\n{diff_elements_str0}\nGamma_reloaded:\n{diff_elements_str1}\nIndices:\n{diff_indices_str}"
                self.fail(message)

            # Clean up the temporary files
            burial_gamma_filepath.unlink()
            gamma_dat_filepath.unlink()

class TestGammaSegmentHandling(unittest.TestCase):

    def setUp(self):
        # Setup with a simplified segment definition and a corresponding gamma array
        self.alphabet = ['A', 'R', 'N', 'D', 'C']  # Simplified for testing
        self.segment_definition = {
            'Segment1': (2, 5),  # Two segments, each of length 5
            'Segment2': (1, 5)   # One segment of length 5
        }
        # Create a gamma array that matches the expected size from segment_definition
        self.gamma_array = np.arange(15)  # Total length = 2*5 + 1*5 = 15
        self.gamma = frustratometer.Gamma(self.gamma_array, segment_definition=self.segment_definition)

    def test_divide_into_segments(self):
        segments = self.gamma.divide_into_segments()
        # Ensure all defined segments are present
        self.assertTrue('Segment1' in segments and 'Segment2' in segments)
        # Validate the shapes and contents of the segments
        self.assertEqual(segments['Segment1'].shape, (2, 5))
        self.assertTrue(np.array_equal(segments['Segment1'], np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])))
        self.assertTrue(np.array_equal(segments['Segment2'], np.array([[10, 11, 12, 13, 14]])))

    def test_divide_into_segments_with_custom_definition(self):
        custom_segment_definition = {'CustomSegment': (3, 5)}  # Three segments, each of length 5
        gamma_custom = frustratometer.Gamma(self.gamma_array, segment_definition=custom_segment_definition)
        segments = gamma_custom.divide_into_segments()
        # Validate the custom segment division
        self.assertTrue('CustomSegment' in segments)
        self.assertEqual(len(segments['CustomSegment']), 3)
        for i, segment in enumerate(segments['CustomSegment']):
            expected_segment = np.arange(i*5, (i+1)*5)
            self.assertTrue(np.array_equal(segment, expected_segment))

    def test_divide_into_segments_edge_cases(self):
        # Test with an empty gamma array
        empty_gamma = frustratometer.Gamma(np.array([]), segment_definition={'EmptySegment': (0, 0)})
        segments = empty_gamma.divide_into_segments()
        self.assertEqual(len(segments['EmptySegment']), 0)

        # Test with a segment definition that does not match the gamma array size
        with self.assertRaises(ValueError):
            mismatched_gamma = frustratometer.Gamma(np.array(range(10)), segment_definition={'Mismatched': (1, 20)})
            mismatched_gamma.divide_into_segments()


class TestGammaReordering(unittest.TestCase):

    def setUp(self):
        self.gamma_array = np.arange(2*5+2*5*5)
        self.segment_definition = {
            '1D': (2, 5),  # Simplified for testing
            '2D': (2, 5,5)  # Simplified for testing
        }
        self.description = "Test Gamma"
        self.alphabet = ['A', 'B', 'C', 'D', 'E']  # Simplified for testing

        self.gamma = frustratometer.Gamma(self.gamma_array, segment_definition=self.segment_definition, description=self.description, alphabet=self.alphabet)

    def test_reorder_valid_alphabet(self):
        new_alphabet = ['A', 'D', 'C', 'B', 'E']  # A valid shuffled alphabet
        reordered = self.gamma.reorder(alphabet=new_alphabet)
        # Check if the alphabet is reordered correctly
        self.assertEqual(reordered.alphabet, new_alphabet)
        self.assertEqual(reordered.gamma_array[1], 3)
        self.assertEqual(reordered.gamma_array[3], 1)
        self.assertEqual(reordered.gamma_array[2], 2)


    def test_reorder_invalid_alphabet(self):
        # Test with an alphabet of incorrect length
        with self.assertRaises(ValueError):
            self.gamma.reorder(alphabet=['A', 'B', 'C'])
        # Test with an alphabet containing invalid characters
        with self.assertRaises(ValueError):
            self.gamma.reorder(alphabet=['A', 'B', 'C', 'D', 'F'])  # 'Z' is not in the original alphabet

    def test_reorder_no_change(self):
        # Reorder with the same alphabet should result in no change
        reordered = self.gamma.reorder(alphabet=self.alphabet)
        self.assertEqual(reordered.alphabet, self.alphabet)
        self.assertTrue(np.array_equal(reordered.gamma_array, self.gamma_array))

    def test_reorder_immutability(self):
        new_alphabet = ['C', 'A', 'D', 'B', 'E']
        original_array_copy = self.gamma.gamma_array.copy()
        self.gamma.reorder(alphabet=new_alphabet)
        # Ensure the original instance is not modified
        self.assertTrue(np.array_equal(self.gamma.gamma_array, original_array_copy))

    def test_reorder_twice(self):
        reordered = self.gamma.reorder(alphabet=['A', 'D', 'C', 'B', 'E'])
        reordered2 = reordered.reorder(alphabet=['D', 'E', 'A', 'C', 'B'])
        reordered3 = reordered2.reorder(alphabet=self.gamma.alphabet)
        # Ensure the original instance is not modified
        self.assertTrue(np.array_equal(reordered3.gamma_array, self.gamma.gamma_array))

    def test_reorder_segment(self):
        reordered = self.gamma.reorder(segment_definition={'2D': (2, 5, 5),  '1D': (2, 5)})
        self.assertEqual(reordered.gamma_array[0], 10)
        self.assertEqual(reordered.gamma_array[-1], 9)

    def test_reorder_segment_and_alphabet(self):
        reordered = self.gamma.reorder(alphabet=['A', 'D', 'C', 'B', 'E'],segment_definition={'2D': (2, 5,5),'1D': (2, 5)})
        self.assertTrue(np.array_equal(reordered.gamma_array, np.array([10, 13, 12, 11, 14, 25, 28, 27, 26, 29, 
                                                                        20, 23, 22, 21, 24, 15, 18, 17, 16, 19,
                                                                        30, 33, 32, 31, 34, 35, 38, 37, 36, 39, 
                                                                        50, 53, 52, 51, 54, 45, 48, 47, 46, 49,
                                                                        40, 43, 42, 41, 44, 55, 58, 57, 56, 59, 
                                                                        0, 3,  2,  1,  4,  5,  8,  7,  6,  9])))

class TestGammaItemAccess(unittest.TestCase):

    def setUp(self):
        self.gamma = frustratometer.Gamma(np.arange(35), segment_definition={'Burial': (2, 5), 'Direct': (1, 5,5)}, description="Test")

    def test_get_item(self):
        burial_segment = self.gamma['Burial']
        self.assertTrue(np.array_equal(burial_segment, np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])))

    def test_get_item_2D(self):
        direct_segment = self.gamma['Direct']
        self.assertTrue(np.array_equal(direct_segment, np.arange(10,35).reshape(1,5,5)))

    def test_set_item(self):
        new_segment = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        self.gamma['Burial'] = new_segment
        self.assertTrue(np.array_equal(self.gamma['Burial'], new_segment))

    def test_set_item_2D(self):
        new_segment = np.arange(50,75).reshape(1,5,5)
        self.gamma['Direct'] = new_segment
        self.assertTrue(np.array_equal(self.gamma['Direct'], new_segment))

    def test_set_item_invalid_shape(self):
        with self.assertRaises(ValueError):
            self.gamma['Burial'] = np.array([1, 2, 3])  # Incorrect shape

    def test_set_item_invalid_key(self):
        with self.assertRaises(KeyError):
            self.gamma['Nonexistent'] = np.array([1, 2, 3])

    def test_get_item_invalid_key(self):
        with self.assertRaises(KeyError):
            _ = self.gamma['Nonexistent']

### Compaction and Expansion Tests

class TestGammaCompaction(unittest.TestCase):

    def setUp(self):
        self.segment_definition = {
            '1D': (2, 5),  # Two 1D segments of length 5
            '2D': (2, 5, 5)  # Two 2D segments of shape 5x5
        }
        self.gamma_array = np.arange(2*5 + 2*5*5, dtype=np.float64)  # Array to match the segment definition
        self.gamma = frustratometer.Gamma(self.gamma_array, segment_definition=self.segment_definition, description="Test Gamma", alphabet=['A', 'B', 'C', 'D', 'E'])
        self.gamma['2D'] = (self.gamma['2D'] + self.gamma['2D'].transpose(0,2,1))

    def test_compact_segments(self):
        compacted = self.gamma.compact_segments()
        expected_length = 40  # Sum of products of dimensions in segment_definition
        self.assertEqual(len(compacted), expected_length, "Compacted array length does not match expected length.")

    def test_expand_segments(self):
        compacted = self.gamma.compact_segments()
        expanded = self.gamma.expand_segments(compacted)
        expected_shape = self.gamma_array.shape
        self.assertEqual(expanded.gamma_array.shape, expected_shape, "Expanded array shape does not match original array shape.")
        self.assertTrue(np.array_equal(expanded.gamma_array, self.gamma.gamma_array), "Compaction and expansion did not result in the original array.")

    def test_divide_into_segments(self):
        segments = self.gamma.divide_into_segments()
        expected_segments = ['1D', '2D']
        self.assertTrue(all(key in segments for key in expected_segments), "Not all segments are present.")
        self.assertTrue(all(isinstance(segments[key], np.ndarray) for key in segments), "Not all segments are numpy arrays.")
        for key, shape in [('1D', (2, 5)), ('2D', (2, 5, 5))]:
            self.assertEqual(segments[key].shape, shape, f"Segment {key} does not match expected shape {shape}.")

    def test_compact_and_expand_sum(self):
        self.gamma_array = np.arange(2*5 + 2*5*5, dtype=np.float64)  # Array to match the segment definition
        self.gamma = frustratometer.Gamma(self.gamma_array, segment_definition=self.segment_definition, description="Test Gamma", alphabet=['A', 'B', 'C', 'D', 'E'])
        self.gamma['2D'] = (self.gamma['2D'] + self.gamma['2D'].transpose(0,2,1))/2
        compacted = self.gamma.compact_segments(sum_offdiagonal=True)
        expanded = self.gamma.expand_segments(compacted,sum_offdiagonal=True)
        print(self.gamma.gamma_array)
        print(compacted)
        print(expanded)

        print(self.gamma['2D'])
        print(expanded['2D'])
        expected_shape = self.gamma_array.shape
        self.assertEqual(expanded.gamma_array.shape, expected_shape, "Expanded array shape does not match original array shape.")
        self.assertTrue(np.array_equal(expanded.gamma_array, self.gamma.gamma_array), "Compaction and expansion did not result in the original array.")
        


class TestGammaNormalization(unittest.TestCase):
    def setUp(self):
        self.segment_definition = {
            '1D': (2, 5),  # Two 1D segments of length 5
            '2D': (2, 5, 5)  # Two 2D segments of shape 5x5
        }
        self.gamma_array = np.arange(2*5 + 2*5*5, dtype=np.float64)  # Array to match the segment definition
        self.gamma = frustratometer.Gamma(self.gamma_array, segment_definition=self.segment_definition, description="Test Gamma", alphabet=['A', 'B', 'C', 'D', 'E'])
        self.gamma['2D'] = (self.gamma['2D'] + self.gamma['2D'].transpose(0,2,1))

    def test_normalize(self):
        # Call the normalize function
        normalized_gamma = self.gamma.normalize()

        # Check if the gamma_array is normalized
        np.testing.assert_almost_equal((normalized_gamma.gamma_array**2).sum(), 1)
        

    def test_center(self):
        # Call the center function
        centered_gamma = self.gamma.center()

        # Check if the gamma_array is centered
        expected_centered_gamma_array = np.array([-2, -1, 0, 1, 2])
        np.testing.assert_allclose(centered_gamma.gamma_array.mean(), 0)

    def test_symmetrize(self):
        # Call the symmetrize function
        symmetrized_gamma = self.gamma.symmetrize()

        # Check if the gamma_array is symmetrized
        self.assertNotEqual(symmetrized_gamma.gamma_array[0], symmetrized_gamma.gamma_array[5])
        self.assertNotEqual(symmetrized_gamma.gamma_array[3], symmetrized_gamma.gamma_array[5])
        self.assertNotEqual(symmetrized_gamma.gamma_array[10], symmetrized_gamma.gamma_array[11])
        self.assertEqual(symmetrized_gamma.gamma_array[11], symmetrized_gamma.gamma_array[15])
        self.assertEqual(symmetrized_gamma.gamma_array[12], symmetrized_gamma.gamma_array[20])
        self.assertEqual(symmetrized_gamma.gamma_array[36], symmetrized_gamma.gamma_array[40])

### Correlation Tests

class TestGammaCorrelation(unittest.TestCase):

    def setUp(self):
        self.gamma1 = frustratometer.Gamma(np.arange(0,1260,1))
        self.gamma2 = frustratometer.Gamma(np.arange(0,1260,1)*5+10)
        self.gamma3 = frustratometer.Gamma(np.arange(1260,0,-1)*2-4)

    def test_correlate_with_compatible_instances(self):
        correlation = self.gamma1.correlate(self.gamma2)
        self.assertIsNotNone(correlation)
        self.assertEqual(correlation, 1.0)

        correlation = self.gamma1.correlate(self.gamma3)
        self.assertEqual(correlation, -1.0)


    def test_correlate_segments_with_compatible_instances(self):
        correlations = self.gamma1.correlate_segments(self.gamma2)
        expected_correlations = {'Burial': 1.0, 'Direct': 1.0, 'Water': 1.0, 'Protein': 1.0}
        self.assertDictEqual(correlations, expected_correlations)

        correlations = self.gamma1.correlate_segments(self.gamma3)
        expected_correlations = {'Burial': -1.0, 'Direct': -1.0, 'Water': -1.0, 'Protein': -1.0}
        self.assertDictEqual(correlations, expected_correlations)


    def test_correlate_with_incompatible_instances(self):
        # Incompatible due to different segment definitions
        incompatible_gamma = frustratometer.Gamma(np.arange(10), {'DifferentSegment': (2, 5)}, alphabet=['A', 'B' ,'C', 'D', 'E'])
        with self.assertRaises(ValueError):
            self.gamma1.correlate(incompatible_gamma)


### File I/O Tests

class TestGammaFileIO(unittest.TestCase):

    def setUp(self):
        # Simple setup for file I/O tests
        self.segment_definition = {'Segment': (1, 5)}
        self.gamma = frustratometer.Gamma(np.arange(5), self.segment_definition)
        self.temp_file_path = "gamma_test_file.json"

    def tearDown(self):
        # Cleanup created file
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def test_to_file(self):
        # Test serialization to file
        self.gamma.to_file(self.temp_file_path)
        self.assertTrue(os.path.exists(self.temp_file_path))
        with open(self.temp_file_path, 'r') as file:
            data = json.load(file)
            #self.assertEqual(data['segment_definition'], self.gamma.segment_definition)
            self.assertTrue('gamma_array' in data)

    def test_from_file(self):
        # Test deserialization from file
        self.gamma.to_file(self.temp_file_path)
        loaded_gamma = frustratometer.Gamma.from_file(self.temp_file_path)
        self.assertTrue(np.array_equal(loaded_gamma.gamma_array, self.gamma.gamma_array))
        self.assertEqual(loaded_gamma.segment_definition, self.gamma.segment_definition)

# Add more tests as needed...

# Copy Tests

# Test deepcopy to ensure it creates a true deep copy of the instance, with no shared references with the original.

class TestGammaDeepCopy(unittest.TestCase):

    def setUp(self):
        self.gamma = frustratometer.Gamma(np.random.rand(1260), description="Test Gamma")

    def test_deepcopy(self):
        gamma_copy = self.gamma.deepcopy()
        gamma_copy.description='Copy Gamma'
        # Ensure the copy is not the same as the original
        self.assertIsNot(gamma_copy, self.gamma)
        self.assertIsNot(gamma_copy.gamma_array, self.gamma.gamma_array)
        self.assertIsNot(gamma_copy.segment_definition, self.gamma.segment_definition)
        self.assertIsNot(gamma_copy.alphabet, self.gamma.alphabet)
        self.assertIsNot(gamma_copy.description, self.gamma.description)
        # Ensure the content is the same
        self.assertTrue(np.array_equal(gamma_copy.gamma_array, self.gamma.gamma_array))
        # Ensure the deepcopy includes a deep copy of mutable attributes
        gamma_copy.gamma_array[25] = 99
        self.assertNotEqual(self.gamma.gamma_array[25], gamma_copy.gamma_array[25])

    def test_copy_initialization(self):
        gamma_copy = frustratometer.Gamma(self.gamma)
        gamma_copy.description='Copy Gamma'
        # Ensure the copy is not the same as the original
        self.assertIsNot(gamma_copy, self.gamma)
        self.assertIsNot(gamma_copy.gamma_array, self.gamma.gamma_array)
        self.assertIsNot(gamma_copy.segment_definition, self.gamma.segment_definition)
        self.assertIsNot(gamma_copy.alphabet, self.gamma.alphabet)
        self.assertIsNot(gamma_copy.description, self.gamma.description)
        # Ensure the content is the same
        self.assertTrue(np.array_equal(gamma_copy.gamma_array, self.gamma.gamma_array))
        # Ensure the deepcopy includes a deep copy of mutable attributes
        gamma_copy.gamma_array[25] = 99
        self.assertNotEqual(self.gamma.gamma_array[25], gamma_copy.gamma_array[25])

    def test_from_instance(self):
        gamma_copy = frustratometer.Gamma.from_instance(self.gamma)
        gamma_copy.description='Copy Gamma'
        # Ensure the copy is not the same as the original
        self.assertIsNot(gamma_copy, self.gamma)
        self.assertIsNot(gamma_copy.gamma_array, self.gamma.gamma_array)
        self.assertIsNot(gamma_copy.segment_definition, self.gamma.segment_definition)
        self.assertIsNot(gamma_copy.alphabet, self.gamma.alphabet)
        self.assertIsNot(gamma_copy.description, self.gamma.description)
        # Ensure the content is the same
        self.assertTrue(np.array_equal(gamma_copy.gamma_array, self.gamma.gamma_array))
        # Ensure the deepcopy includes a deep copy of mutable attributes
        gamma_copy.gamma_array[25] = 99
        self.assertNotEqual(self.gamma.gamma_array[25], gamma_copy.gamma_array[25])


# Magic Methods Tests

# For each arithmetic magic method (__add__, __sub__, __mul__, __truediv__, and their reciprocal and in-place versions), test with compatible and incompatible inputs, including other Gamma instances and scalar values.
# Test the __repr__ method to ensure it returns a string representation accurately.
class TestGammaMagicMethods(unittest.TestCase):

    def setUp(self):
        self.gamma1 = frustratometer.Gamma(np.arange(1260), description="Gamma 1")
        self.gamma2 = frustratometer.Gamma(2*np.arange(1260), description="Gamma 2")

    def test_addition_with_instance(self):
        # Test addition
        gamma_add = self.gamma1 + self.gamma2
        expected_result = 3*np.arange(1260)
        np.testing.assert_array_equal(gamma_add.gamma_array, expected_result, "Addition with instance failed.")

    def test_subtraction_with_instance(self):
        # Test subtraction
        gamma_sub = self.gamma1 - self.gamma2
        expected_result = -1*np.arange(1260)
        np.testing.assert_array_equal(gamma_sub.gamma_array, expected_result, "Subtraction with instance failed.")

    def test_multiplication_with_instance(self):
        # Test multiplication
        gamma_mul = self.gamma1 * self.gamma2
        expected_result = 2*(np.arange(1260)**2)
        np.testing.assert_array_equal(gamma_mul.gamma_array, expected_result, "Multiplication with instance failed.")

    def test_division_with_instance(self):
        # Test true division
        gamma_div = self.gamma2 / self.gamma1
        expected_result = np.full(1260, 2, dtype=np.float64)
        expected_result[0] = np.inf  # Division by zero for the first element
        np.testing.assert_array_equal(gamma_div.gamma_array[1:], expected_result[1:], "Division with instance failed.")

    def test_addition_with_scalar(self):
        # Test scalar addition
        gamma_add_scalar = self.gamma1 + 10
        expected_result = np.arange(1260) + 10
        np.testing.assert_array_equal(gamma_add_scalar.gamma_array, expected_result, "Addition with scalar failed.")

    def test_multiplication_with_scalar(self):
        # Test scalar multiplication
        gamma_mul_scalar = self.gamma1 * 10
        expected_result = 10*np.arange(1260)
        np.testing.assert_array_equal(gamma_mul_scalar.gamma_array, expected_result, "Multiplication with scalar failed.")

    def test_reciprocal_addition(self):
        # Test reciprocal operations for addition
        gamma_radd = 10 + self.gamma1
        expected_result = np.arange(1260) + 10
        np.testing.assert_array_equal(gamma_radd.gamma_array, expected_result, "Reciprocal addition failed.")

    def test_reciprocal_multiplication(self):
        # Test reciprocal operations for multiplication
        gamma_rmul = 10 * self.gamma1
        expected_result = 10*np.arange(1260)
        np.testing.assert_array_equal(gamma_rmul.gamma_array, expected_result, "Reciprocal multiplication failed.")

    # In-place tests
    def test_inplace_addition(self):
        # Make a copy before in-place operation
        gamma_copy = self.gamma1.deepcopy()
        gamma_copy += self.gamma2
        self.assertTrue(np.array_equal(gamma_copy.gamma_array, 3 * np.arange(1260)))

    def test_inplace_subtraction(self):
        # Make a copy before in-place operation
        gamma_copy = self.gamma1.deepcopy()
        gamma_copy -= self.gamma2
        self.assertTrue(np.array_equal(gamma_copy.gamma_array, -1 * np.arange(1260)))

    def test_inplace_multiplication(self):
        # Make a copy before in-place operation
        gamma_copy = self.gamma1.deepcopy()
        gamma_copy *= self.gamma2
        self.assertTrue(np.array_equal(gamma_copy.gamma_array, 2 * np.arange(1260) ** 2))

    def test_inplace_division(self):
        # Make a copy before in-place operation
        gamma_copy = self.gamma2.deepcopy()
        gamma_copy /= 5
        self.assertTrue(np.allclose(gamma_copy.gamma_array, 2*np.arange(1260)/5))

    def test_repr_method(self):
        # Test __repr__ method
        repr_str = repr(self.gamma1)
        expected_repr = f"Gamma(array={np.arange(1260)}, description='Gamma 1')"
        self.assertEqual(repr_str, expected_repr, "__repr__ method failed.")


# Plotting Tests (if applicable)

# Since plotting typically doesn't return a value, consider testing that the plot_gamma method runs without errors for valid inputs. Mocking or a visual inspection approach might be needed.
from unittest.mock import patch
import unittest

class TestGammaPlotting(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def test_plot_gamma(self, mock_show):
        gamma = frustratometer.Gamma(np.arange(1260), description="Test plot")
        try:
            gamma.plot_gamma()
        except Exception as e:
            self.fail(f"plot_gamma() raised an exception {e}")

# Error Handling Tests

# Test all methods for expected error conditions, such as passing incompatible types, invalid file paths, or incorrect data formats.
class TestGammaErrorHandling(unittest.TestCase):

    def test_invalid_initialization(self):
        with self.assertRaises(TypeError):
            frustratometer.Gamma([0,1,2])

    def test_invalid_file_path(self):
        with self.assertRaises(FileNotFoundError):
            frustratometer.Gamma.from_file("nonexistent_file.json")

    def test_incompatible_addition(self):
        gamma1 = frustratometer.Gamma(np.arange(1260))
        with self.assertRaises(TypeError):
            _ = gamma1 + "not a gamma or scalar"

class TestGammaEdgeCases(unittest.TestCase):

    def test_empty_gamma_array(self):
        with self.assertRaises(ValueError):
            frustratometer.Gamma(np.array([]))  # Assuming an empty array is invalid

    def test_empty_segment_definition(self):
        with self.assertRaises(ValueError):
            frustratometer.Gamma(np.array([1, 2, 3]), segment_definition={})