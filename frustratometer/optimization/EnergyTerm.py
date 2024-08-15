import abc
import numpy as np
# from typing import Callable
# from functools import lru_cache  #TODO: Implement lru_cache for memoization of energy functions

try:
    import numba
    import numba.core.registry
    from numba import types
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class EnergyTerm(abc.ABC):
    """
    Abstract base class for Energy terms in sequence optimization.

    This class provides a template for calculating the energy of a sequence,
    the energy difference of a mutation, and the energy difference of swapping two amino acids.
    It aims to leverage the numba library for just-in-time compilation of energy functions when available.
    The energy, denergy_mutation and denergy_swap methods should be numba compatible.

    To create custom energy terms, inherit from this class and implement the abstract methods.
    """
    def __init__(self, use_numba=True):
        """
        Initialize the EnergyTerm.

        Args:
            use_numba (bool): Whether to use numba for JIT compilation. Defaults to True.
        """
        self.use_numba = use_numba and NUMBA_AVAILABLE

    def energy(self, seq_index:np.ndarray):
        """ Returns the energy of a sequence. """
        return self.energy_function(seq_index)
    
    def denergy_mutation(self, seq_index:np.ndarray, pos:int, aa:int):
        """ Returns the energy change of a mutation in a sequence. """
        return self.denergy_mutation_function(seq_index, pos, aa)
    
    def denergy_swap(self, seq_index:np.ndarray, pos1:int, pos2:int):
        """ Returns the energy change of swapping two amino acids in a sequence. """
        return self.denergy_swap_function(seq_index, pos1, pos2)
    
    @property
    def use_numba(self):
        """ Getter for the use_numba property. """
        return self._use_numba
    
    @use_numba.setter
    def use_numba(self, value):
        """ Setter for the use_numba property. """
        self._use_numba = value

    @property
    def numbify(self):
        """ Returns the numba decorator if use_numba is True, otherwise returns the same dummy decorator. """
        if self.use_numba:
            return numba.njit
        else:
            return self.dummy_decorator

    @staticmethod
    def dummy_decorator(func, *args, **kwargs):
        """ Dummy decorator for functions that do not require numba compilation. """
        return func

    @property
    def energy_function(self):
        """ Returns the energy function as a numba dispatcher. """
        if self.use_numba:
            return numba.njit(types.float64(types.Array(types.int64, 1, 'A', readonly=True)))(self.compute_energy)
        return self.compute_energy

    @property
    def denergy_mutation_function(self):
        """ Returns the mutation energy change function as a numba dispatcher. """
        if self.use_numba:
            return numba.njit(types.float64(types.Array(types.int64, 1, 'A', readonly=True),types.int64,types.int64))(self.compute_denergy_mutation)
        return self.compute_denergy_mutation

    @property
    def denergy_swap_function(self):
        """ Returns the swap energy change function as a numba dispatcher. """
        if self.use_numba:
            return numba.njit(types.float64(types.Array(types.int64, 1, 'A', readonly=True),types.int64,types.int64))(self.compute_denergy_swap)
        return self.compute_denergy_swap
    
    @staticmethod
    #@abc.abstractmethod #TODO: Add abstract method decorator. Currently not working due to the late initialization of the methods.
    def compute_energy(seq_index:np.ndarray):
        """ Abstract method for computing the energy of a sequence."""
        return 0.
    
    @staticmethod
    #@abc.abstractmethod
    def compute_denergy_mutation(seq_index:np.ndarray, pos:int, aa):
        """ Abstract method for computing the energy change of a mutation in a sequence."""
        return 0.
    
    @staticmethod
    #@abc.abstractmethod
    def compute_denergy_swap(seq_index:np.ndarray, pos1:int, pos2:int):
        """ Abstract method for computing the energy change of swapping two amino acids in a sequence."""
        return 0.

    def __add__(self, other):
        new_energy_term = EnergyTerm()
        if isinstance(other, EnergyTerm):
            new_energy_term.use_numba = self.use_numba and other.use_numba
            e1=self.energy_function; e2=other.energy_function
            m1=self.denergy_mutation_function; m2=other.denergy_mutation_function
            s1=self.denergy_swap_function; s2=other.denergy_swap_function

            new_energy_term.compute_energy = lambda seq_index: e1(seq_index) + e2(seq_index)
            new_energy_term.compute_denergy_mutation = lambda seq_index,pos,aa: m1(seq_index,pos,aa) + m2(seq_index,pos,aa)
            new_energy_term.compute_denergy_swap = lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) + s2(seq_index,pos1,pos2)
        elif isinstance(other, (int, float)):
            new_energy_term.use_numba = self.use_numba
            e1=self.energy_function

            new_energy_term.compute_energy = lambda seq_index: e1(seq_index) + other
            new_energy_term.compute_denergy_mutation = self.compute_denergy_mutation
            new_energy_term.compute_denergy_swap = self.compute_denergy_swap
        return new_energy_term
        
    def __mul__(self, other):
        new_energy_term = EnergyTerm()
        if isinstance(other, EnergyTerm):
            new_energy_term.use_numba = self.use_numba and other.use_numba
            e1=self.energy_function; e2=other.energy_function
            m1=self.denergy_mutation_function; m2=other.denergy_mutation_function
            s1=self.denergy_swap_function; s2=other.denergy_swap_function

            new_energy_term.compute_energy = lambda seq_index: e1(seq_index) * e2(seq_index)

            def compute_denergy_mutation(seq_index, pos, aa):
                mr1 = m1(seq_index, pos, aa)
                mr2 = m2(seq_index, pos, aa)
                return mr1 * mr2 + mr1 * e2(seq_index) + e1(seq_index) * mr2

            def compute_denergy_swap(seq_index, pos1, pos2):
                sr1 = s1(seq_index, pos1, pos2)
                sr2 = s2(seq_index, pos1, pos2)
                return sr1 * sr2 + sr1 * e2(seq_index) + e1(seq_index) * sr2

            # Assigning the functions to new_energy_term
            new_energy_term.compute_denergy_swap = compute_denergy_swap
            new_energy_term.compute_denergy_mutation = compute_denergy_mutation
            # new_energy_term.compute_denergy_swap = lambda seq_index,pos,aa: (mr1 := m1(seq_index,pos,aa)) * (mr2 := m2(seq_index,pos,aa)) + mr1 * e2(seq_index) + e1(seq_index) * mr2
            # new_energy_term.compute_denergy_mutation = lambda seq_index,pos1,pos2: (sr1 := s1(seq_index,pos1,pos2)) * (sr2 := s2(seq_index,pos1,pos2)) + sr1 * e2(seq_index) + e1(seq_index) * sr2
        elif isinstance(other, (int, float)):
            new_energy_term.use_numba = self.use_numba
            e1=self.energy_function
            m1=self.denergy_mutation_function
            s1=self.denergy_swap_function

            new_energy_term.compute_energy = lambda seq_index: e1(seq_index) * other
            new_energy_term.compute_denergy_mutation = lambda seq_index,pos,aa: m1(seq_index,pos,aa) * other
            new_energy_term.compute_denergy_swap = lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) * other
        return new_energy_term
        
    def __sub__(self, other):
        new_energy_term = EnergyTerm()
        if isinstance(other, EnergyTerm):
            new_energy_term.use_numba = self.use_numba and other.use_numba
            e1=self.energy_function; e2=other.energy_function
            m1=self.denergy_mutation_function; m2=other.denergy_mutation_function
            s1=self.denergy_swap_function; s2=other.denergy_swap_function

            new_energy_term.compute_energy = lambda seq_index: e1(seq_index) - e2(seq_index)
            new_energy_term.compute_denergy_mutation = lambda seq_index,pos,aa: m1(seq_index,pos,aa) - m2(seq_index,pos,aa)
            new_energy_term.compute_denergy_swap = lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) - s2(seq_index,pos1,pos2)
        elif isinstance(other, (int, float)):
            new_energy_term.use_numba = self.use_numba
            e1=self.energy_function

            new_energy_term.compute_energy = lambda seq_index: e1(seq_index) - other
            new_energy_term.compute_denergy_mutation = self.compute_denergy_mutation
            new_energy_term.compute_denergy_swap = self.compute_denergy_swap
        return new_energy_term

    def __truediv__(self, other):
        new_energy_term = EnergyTerm()
        if isinstance(other, EnergyTerm):
            new_energy_term.use_numba = self.use_numba and other.use_numba
            e1=self.energy_function; e2=other.energy_function
            m1=self.denergy_mutation_function; m2=other.denergy_mutation_function
            s1=self.denergy_swap_function; s2=other.denergy_swap_function

            new_energy_term.compute_energy = lambda seq_index: e1(seq_index) / e2(seq_index)

            def compute_denergy_mutation(seq_index, pos, aa):
                er1 = e1(seq_index)
                mr1 = m1(seq_index, pos, aa)
                er2 = e2(seq_index)
                mr2 = m2(seq_index, pos, aa)
                
                return (mr1 * er2  - mr2 * er1) / (er2**2 + er2*mr2)

            def compute_denergy_swap(seq_index, pos1, pos2):
                er1 = e1(seq_index)
                sr1 = s1(seq_index, pos1, pos2)
                er2 = e2(seq_index)
                sr2 = s2(seq_index, pos1, pos2)
                
                return (sr1 * er2  - sr2 * er1) / (er2**2 + er2*sr2)

            new_energy_term.compute_denergy_mutation = compute_denergy_mutation
            new_energy_term.compute_denergy_swap = compute_denergy_swap
            # new_energy_term.compute_denergy_mutation = lambda seq_index,pos,aa: (e1(seq_index) * (mr2 := m2(seq_index,pos,aa))) - ((er2 := e2(seq_index)) * m1(seq_index,pos,aa)) / (er2 **2 -er2*mr2)
            # new_energy_term.compute_denergy_swap = lambda seq_index,pos1,pos2: (e1(seq_index) * (sr2 := m2(seq_index,pos1,pos2))) - ((er2 := e2(seq_index) * s1(seq_index,pos1,pos2))) / (er2 **2 -er2*sr2)

        elif isinstance(other, (int, float)):
            new_energy_term.use_numba = self.use_numba
            e1=self.energy_function
            m1=self.denergy_mutation_function
            s1=self.denergy_swap_function

            new_energy_term.compute_energy = lambda seq_index: e1(seq_index) / other
            new_energy_term.compute_denergy_mutation = lambda seq_index,pos,aa: m1(seq_index,pos,aa) / other
            new_energy_term.compute_denergy_swap = lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) / other
        return new_energy_term

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return (self * -1).__add__(other)
    
    def __rtruediv__(self, other):
        new_energy_term = EnergyTerm()
        if isinstance(other, EnergyTerm):
            return other / self
        elif isinstance(other, (int, float)):
            new_energy_term.use_numba = self.use_numba
            e1=self.energy_function
            m1=self.denergy_mutation_function
            s1=self.denergy_swap_function

            new_energy_term.compute_energy = lambda seq_index: other / e1(seq_index)

            def compute_denergy_mutation(seq_index, pos, aa):
                mr1 = m1(seq_index, pos, aa)
                er1 = e1(seq_index)
                
                return - other * mr1 / er1 / (er1 + mr1)

            def compute_denergy_swap(seq_index, pos1, pos2):
                sr1 = s1(seq_index, pos1, pos2)
                er1 = e1(seq_index)
                
                return - other * sr1 / er1 / (er1 + sr1)

            # Assigning the functions to new_energy_term
            new_energy_term.compute_denergy_mutation = compute_denergy_mutation
            new_energy_term.compute_denergy_swap = compute_denergy_swap
            
            # new_energy_term.compute_denergy_mutation = lambda seq_index,pos,aa: other * (mr1:=m1(seq_index,pos,aa)) / (er1:=e1(seq_index)) / (er1 - mr1)
            # new_energy_term.compute_denergy_swap = lambda seq_index,pos1,pos2: other * (sr1:=s1(seq_index,pos1,pos2)) / (er1:=e1(seq_index)) / (er1 - sr1)
        return new_energy_term

    def __new__(cls, *args, **kwargs):
        new_instance = super().__new__(cls)
        new_instance.use_numba = True
        return new_instance
        
    def test_energy(self,seq_index=np.array([0,1,2,3,4])):
        assert type(self.energy(seq_index)) in [float,np.float64], f"Energy function should return a float."

    def test_denergy_mutation(self,seq_index=np.array([0,1,2,3,4]), pos=0, aa=1):
        assert type(self.denergy_mutation(seq_index, pos, aa)) in [float,np.float64], "Mutation energy change function should return a float"

    def test_denergy_swap(self,seq_index=np.array([0,1,2,3,4]) ,pos1=0, pos2=1):
        assert type(self.denergy_swap(seq_index, pos1, pos2)) in [float,np.float64], "Swap energy change function should return a float"

    def test_numba(self):
        if self.use_numba:
            assert type(self.energy_function) is numba.core.registry.CPUDispatcher, "Energy function should be a numba dispatcher"
            assert type(self.denergy_mutation_function) is numba.core.registry.CPUDispatcher, "Mutation energy change function should be a numba dispatcher"
            assert type(self.denergy_swap_function) is numba.core.registry.CPUDispatcher, "Swap energy change function should be a numba dispatcher"
        else:
            assert type(self.energy_function) is not numba.core.registry.CPUDispatcher, "Energy function should not be a numba dispatcher"
            assert type(self.denergy_mutation_function) is not numba.core.registry.CPUDispatcher, "Mutation energy change function should not be a numba dispatcher"
            assert type(self.denergy_swap_function) is not numba.core.registry.CPUDispatcher,   "Swap energy change function should not be a numba dispatcher"

    def test_denergy_mutation_accuracy(self, seq_index, pos, aa):
        dE_fast = self.denergy_mutation(seq_index, pos, aa)
        seq_index2=seq_index.copy()
        seq_index2[pos] = aa
        dE_slow = self.energy(seq_index2) - self.energy(seq_index)
        assert np.allclose(dE_fast, dE_slow), f"Mutation energy change is not the same as the Energy function \nFast: {dE_fast}, Slow: {dE_slow}, pos: {pos}, aa: {aa}, seq_index: {seq_index}, seq_index2: {seq_index2}, energy: {self.energy(seq_index)}, energy2: {self.energy(seq_index2)}, energy_diff: {self.energy(seq_index2) - self.energy(seq_index)}"

    def test_denergy_swap_accuracy(self, seq_index, pos1, pos2):
        dE_fast = self.denergy_swap(seq_index, pos1, pos2)
        seq_index2=seq_index.copy()
        seq_index2[pos1], seq_index2[pos2] = seq_index2[pos2], seq_index2[pos1]
        dE_slow = self.energy(seq_index2) - self.energy(seq_index)
        assert np.allclose(dE_fast, dE_slow), f"Swap energy change is not the same as the Energy function \nFast: {dE_fast}, Slow: {dE_slow}, pos1: {pos1}, pos2: {pos2}, seq_index: {seq_index}, seq_index2: {seq_index2}, energy: {self.energy(seq_index)}, energy2: {self.energy(seq_index2)}, energy_diff: {self.energy(seq_index2) - self.energy(seq_index)}"


    def test(self,seq_index=np.array([0,1,2,3,4,0,1,2,3,4])):
        self.test_energy(seq_index)
        self.test_denergy_mutation(seq_index, np.random.randint(len(seq_index)), 1)
        self.test_denergy_swap(seq_index, np.random.randint(len(seq_index)), np.random.randint(len(seq_index)))
        self.test_numba()
        self.test_denergy_mutation_accuracy(seq_index, np.random.randint(len(seq_index)), 0)
        self.test_denergy_swap_accuracy(seq_index, np.random.randint(len(seq_index)), np.random.randint(len(seq_index)))
        print("All tests passed!")

    def benchmark(self,seq_indices):
        import time
        
        # Compile functions (run once and discard result)
        energy_function = self.energy_function
        denergy_mutation_function = self.denergy_mutation_function
        denergy_swap_function = self.denergy_swap_function

        energy_function(seq_indices[0])
        denergy_mutation_function(seq_indices[0], 0, 1)
        denergy_swap_function(seq_indices[0], 0, 1)
        
        # energy_function_loop = self.numbify(lambda seq_indices: [(energy_function(seq_index) for seq_index in seq_indices])])
        # denergy_mutation_function_loop = self.numbify(lambda seq_indices: [denergy_mutation_function(seq_index, 0, 1) for seq_index in seq_indices])
        # denergy_swap_function_loop = self.numbify(lambda seq_indices: [denergy_swap_function(seq_index, 1, 0) for seq_index in seq_indices])

        # energy_function_loop_instantiated=energy_function_loop
        # denergy_mutation_function_loop_instantiated =denergy_mutation_function_loop
        # denergy_swap_function_loop_instantiated=denergy_swap_function_loop

        # energy_function_loop_instantiated(seq_indices[0:1])
        # denergy_mutation_function_loop_instantiated(seq_indices[0:1])
        # denergy_swap_function_loop_instantiated(seq_indices[0:1])
        
        # TODO: Implement looped functions in numba for benchmarking
        t0 = time.time()
        for seq_index in seq_indices:
            energy_function(seq_index)
        t1 = time.time()
        print(f"Energy function took {(t1-t0)/len(seq_indices)*1E6} microseconds per sequence")

        t0 = time.time()
        for seq_index in seq_indices:
            denergy_mutation_function(seq_index, 0, 1)
        t1 = time.time()
        print(f"Mutation energy change function took {(t1-t0)/len(seq_indices)*1E6} microseconds per sequence")

        t0 = time.time()
        for seq_index in seq_indices:
            denergy_swap_function(seq_index, 1, 0)
        t1 = time.time()
        print(f"Swap energy change function took {(t1-t0)/len(seq_indices)*1E6} microseconds per sequence")


if __name__ == "__main__":
    class TestEnergyTerm(EnergyTerm):
        
        @staticmethod
        def compute_energy(seq_index:np.ndarray):
            return float(np.sum(seq_index*np.arange(len(seq_index))))

        @staticmethod
        def compute_denergy_mutation(seq_index:np.ndarray, pos:int, aa:int):
            return float((aa-seq_index[pos])*pos)

        @staticmethod
        def compute_denergy_swap(seq_index:np.ndarray, pos1:int, pos2:int):
            return float((seq_index[pos1]-seq_index[pos2])*(pos2-pos1))

    et = TestEnergyTerm()
    et.test()

    seq_index=np.array([0,1,2,3])
    print("No numba benchmark")
    et.use_numba = False
    et.benchmark([seq_index for _ in range(1000)])

    print("Numba benchmark")
    et.use_numba = True
    et.benchmark([seq_index for _ in range(1000)])