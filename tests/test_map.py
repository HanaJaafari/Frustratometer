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
    
    def test_repr(self):
        # Test the string representation
        test_map = Map(self.map_array)
        expected_repr = "<class 'dca_frustratometer.classes.Map.Map'>\n"\
                        "Seq1: -SSSSS--\n"\
                        "Map:  -|-||---\n"\
                        "Seq2: SS-SS-SS"
        self.assertEqual(test_map.__repr__(), expected_repr)

if __name__ == '__main__':
    unittest.main()