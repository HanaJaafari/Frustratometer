import unittest
import numpy as np
from dca_frustratometer import Map  # Ensure to import the Map class correctly

class TestMap(unittest.TestCase):
    def setUp(self):
        # Setup that can be reused across tests
        self.map_array = np.array([[-1,  0,  1,  2,  3,  4, -1, -1],
                                   [ 0,  1, -1,  2,  3, -1,  4,  5]])
        self.path = ((0, 0), (0, 1), (1, 2), (2, 2), (4, 4), (5, 4), (5, 6))
        self.sequence_a = 'ACAEA'
        self.sequence_b = 'AAAECD'
    
    def test_init(self):
        # Test initialization with map_array
        test_map = Map(self.map_array)
        self.assertEqual(test_map.map_array.tolist(), self.map_array.tolist())
        self.assertEqual(test_map.seq1_len, len(self.sequence_a))
        self.assertEqual(test_map.seq2_len, len(self.sequence_b))
    
    def test_from_path(self):
        # Test initialization from path
        test_map = Map.from_path(self.path)
        np.testing.assert_allclose(test_map.map_array, self.map_array)
        self.assertEqual(test_map.seq1_len, len(self.sequence_a))
        self.assertEqual(test_map.seq2_len, len(self.sequence_b))
    
    def test_from_sequences(self):
        # Test initialization from sequences
        test_map = Map.from_sequences(self.sequence_a, self.sequence_b)
        np.testing.assert_allclose(test_map.map_array, self.map_array)
        self.assertEqual(test_map.seq1_len, len(self.sequence_a))
        self.assertEqual(test_map.seq2_len, len(self.sequence_b))
    
    def test_map_forward(self):
        # Test mapping in forward direction
        test_map = Map(self.map_array)
        print(test_map)
        mapped_sequence = test_map.map(sequence=self.sequence_b, reverse=False)
        # Replace the expected output with the correct one based on your implementation
        self.assertEqual(mapped_sequence, "A-AE-")
    
    def test_map_reverse(self):
        # Test mapping in reverse direction
        test_map = Map(self.map_array)
        mapped_sequence = test_map.map(sequence=self.sequence_a, reverse=True)
        # Replace the expected output with the correct one based on your implementation
        self.assertEqual(mapped_sequence, "-AAE--")

    def test_map_wrong_sequence(self):
        # Test mapping with wrong sequence
        test_map = Map(self.map_array)
        with self.assertRaises(IndexError):
            test_map.map(sequence=self.sequence_a, reverse=False)
        with self.assertRaises(IndexError):
            test_map.map(sequence=self.sequence_b, reverse=True)
    
    def test_repr(self):
        # Test the string representation
        test_map = Map(self.map_array)
        expected_repr = "<class 'dca_frustratometer.classes.Map.Map'>\n"\
                        "Seq1: -SSSSS--\n"\
                        "Map:  -|-||---\n"\
                        "Seq2: SS-SS-SS"
        self.assertEqual(test_map.__repr__(), expected_repr)

    def test_copy(self):
        # Test copying the map
        test_map = Map(self.map_array)
        copy_map = test_map.copy()
        np.testing.assert_allclose(copy_map.map_array, self.map_array)
        self.assertEqual(copy_map.seq1_len, len(self.sequence_a))
        self.assertEqual(copy_map.seq2_len, len(self.sequence_b))
        copy_map.map_array = np.array([[0, 1, 2, 3],
                                       [0, 1, 2, 3]])
        self.assertNotEqual(copy_map.map_array.tolist(), test_map.map_array.tolist())
        self.assertNotEqual(copy_map.seq1_len, test_map.seq1_len)
        self.assertNotEqual(copy_map.seq2_len, test_map.seq2_len)

    def test_reverse(self):
        # Test reversing the map
        test_map = Map(self.map_array)
        reverse_map = test_map.reverse()
        np.testing.assert_allclose(reverse_map.map_array, self.map_array[[1,0]])
        self.assertEqual(reverse_map.seq1_len, len(self.sequence_b))
        self.assertEqual(reverse_map.seq2_len, len(self.sequence_a))
        with self.assertRaises(IndexError):
            reverse_map.map(sequence=self.sequence_b, reverse=False)



if __name__ == '__main__':
    unittest.main()