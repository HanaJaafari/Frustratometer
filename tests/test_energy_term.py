import pytest
import numpy as np
from frustratometer.optimization import *

def test_energy_term():
    et = EnergyTerm()
    et.test()


class TestEnergyTerm(EnergyTerm):
    @staticmethod
    def compute_energy(seq_index: np.ndarray):
        return float(np.sum(seq_index * np.arange(len(seq_index))))

    @staticmethod
    def compute_denergy_mutation(seq_index: np.ndarray, pos: int, aa: int):
        return float((aa - seq_index[pos]) * pos)

    @staticmethod
    def compute_denergy_swap(seq_index: np.ndarray, pos1: int, pos2: int):
        return float((seq_index[pos1] - seq_index[pos2]) * (pos2 - pos1))

@pytest.fixture
def energy_term():
    return TestEnergyTerm()

@pytest.fixture
def seq_index():
    return np.array([0, 1, 2, 3])

def test_energy(energy_term, seq_index):
    expected = float(np.sum(seq_index * np.arange(len(seq_index))))
    assert energy_term.energy(seq_index) == expected, f"Energy function is not working, expected {expected}, got {energy_term.energy(seq_index)}"

@pytest.mark.parametrize("pos,aa", [(0, 1)])
def test_denergy_mutation(energy_term, seq_index, pos, aa):
    expected = float((aa - seq_index[pos]) * pos)
    assert energy_term.denergy_mutation(seq_index, pos, aa) == expected, f"Mutation energy change is not working, expected {expected}, got {energy_term.denergy_mutation(seq_index, pos, aa)}"

@pytest.mark.parametrize("pos1,pos2", [(2, 3)])
def test_denergy_swap(energy_term, seq_index, pos1, pos2):
    expected = float((seq_index[pos1] - seq_index[pos2]) * (pos2 - pos1))
    assert energy_term.denergy_swap(seq_index, pos1, pos2) == expected, f"Swap energy change is not working, expected {expected}, got {energy_term.denergy_swap(seq_index, pos1, pos2)}"

@pytest.mark.parametrize("operation,value,expected_factor", [
    (lambda et, v: et + v, 1, lambda x: x + 1),
    (lambda et, v: et * v, 2, lambda x: x * 2),
    (lambda et, v: et - v, 3, lambda x: x - 3),
    (lambda et, v: et / v, 4, lambda x: x / 4),
    (lambda et, v: v + et, 5, lambda x: x + 5),
    (lambda et, v: v * et, 6, lambda x: x * 6),
    (lambda et, v: v - et, 7, lambda x: 7 - x),
    (lambda et, v: v / et, 8, lambda x: 8 / x),
])
def test_constant_operations(energy_term, seq_index, operation, value, expected_factor):
    et2 = operation(energy_term, value)
    original_energy = energy_term.energy(seq_index)
    expected_energy = expected_factor(original_energy)
    assert et2.energy(seq_index) == pytest.approx(expected_energy), f"Expected {expected_energy}, got {et2.energy(seq_index)}"
    et2.test()

@pytest.mark.parametrize("operation,expected_factor", [
    (lambda et: et + 9 * et, lambda x: 10 * x),
    (lambda et: (10 * et) * et, lambda x: 10 * x ** 2),
    (lambda et: (11 * et) - et, lambda x: 10 * x),
    (lambda et: (12 * et) / et, lambda x: 12),
])
def test_combination_operations(energy_term, seq_index, operation, expected_factor):
    et2 = operation(energy_term)
    original_energy = energy_term.energy(seq_index)
    expected_energy = expected_factor(original_energy)
    assert et2.energy(seq_index) == pytest.approx(expected_energy), f"Expected {expected_energy}, got {et2.energy(seq_index)}"
    et2.test()