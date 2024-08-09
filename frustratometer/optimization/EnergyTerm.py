import numpy as np
import numba
import numba.core.registry
import abc

class EnergyTerm(abc.ABC):
    """ Abstract class for Energy terms for sequence optimization.
        The class provides method templates for calculating the energy of a sequence, 
        the energy difference of a mutation and the energy difference of swapping two amino acids.

        This class aims to leverage the numba library for just-in-time compilation of the energy functions.
        The energy, denergy_mutation and denergy_swap methods should be numba compatible.
        
        The class can be inherited to create custom energy terms.
        The energy, denergy_mutation and denergy_swap methods should be implemented in the child class.
        
        """
    def __init__(self, use_numba=True):
        self.use_numba = use_numba

    def energy(self, seq_index:np.ndarray):
        return self.energy_function(seq_index)
    
    def denergy_mutation(self, seq_index:np.ndarray, pos:int, aa:int):
        return self.denergy_mutation_function(seq_index, pos, aa)
    
    def denergy_swap(self, seq_index:np.ndarray, pos1:int, pos2:int):
        return self.denergy_swap_function(seq_index, pos1, pos2)
    
    @property
    def use_numba(self):
        return self._use_numba
    
    @use_numba.setter
    def use_numba(self, value):
        self._use_numba = value

    @property
    def numbify(self):
        if self.use_numba:
            return numba.njit
        else:
            return self.dummy_decorator

    @staticmethod
    def dummy_decorator(func, *args, **kwargs):
        return func

    @property
    def energy_function(self):
        if self.use_numba:
            return numba.njit(self.compute_energy, cache=True)
        return self.compute_energy

    @property
    def denergy_mutation_function(self):
        if self.use_numba:
            return numba.njit(self.compute_denergy_mutation, cache=True)
        return self.compute_denergy_mutation

    @property
    def denergy_swap_function(self):
        if self.use_numba:
            return numba.njit(self.compute_denergy_swap, cache=True)
        return self.compute_denergy_swap
    
    @energy_function.setter
    def energy_function(self, func):
        self._energy_function = func

    @denergy_mutation_function.setter
    def denergy_mutation_function(self, func):
        self._denergy_mutation_function = func

    @denergy_swap_function.setter
    def denergy_swap_function(self, func):
        self._denergy_swap_function = func
    
    @staticmethod
    def compute_energy(seq_index:np.ndarray):
        return 0.
    
    @staticmethod
    def compute_denergy_mutation(seq_index:np.ndarray, pos:int, aa):
        return 0.
    
    @staticmethod
    def compute_denergy_swap(seq_index:np.ndarray, pos1:int, pos2:int):
        return 0.

    def __add__(self, other):
        if isinstance(other, EnergyTerm):
            new_energy_term = EnergyTerm()
            if self.use_numba and other.use_numba:
                new_energy_term.use_numba = True
            else:
                new_energy_term.use_numba = False
            
            e1=self.energy_function; e2=other.energy_function
            new_energy_term.energy_function = new_energy_term.numbify(lambda seq_index: e1(seq_index) + e2(seq_index), cache=True)
            
            m1=self.denergy_mutation_function; m2=other.denergy_mutation_function
            new_energy_term.denergy_mutation_function = new_energy_term.numbify(lambda seq_index,pos,aa: m1(seq_index,pos,aa) + m2(seq_index,pos,aa), cache=True)
            
            s1=self.denergy_swap_function; s2=other.denergy_swap_function
            new_energy_term.denergy_swap_function = new_energy_term.numbify(lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) + s2(seq_index,pos1,pos2), cache=True)
            return new_energy_term
        elif isinstance(other, (int, float)):
            new_energy_term = EnergyTerm()
            new_energy_term.use_numba = self.use_numba
            
            e1=self.energy_function
            new_energy_term.energy_function = new_energy_term.numbify(lambda seq_index: e1(seq_index) + other, cache=True)
            
            m1=self.denergy_mutation_function
            new_energy_term.denergy_mutation_function = new_energy_term.numbify(lambda seq_index,pos,aa: m1(seq_index,pos,aa) + other, cache=True)
            
            s1=self.denergy_swap_function
            new_energy_term.denergy_swap_function = new_energy_term.numbify(lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) + other, cache=True)
            return new_energy_term
        
    def __mul__(self, other):
        if isinstance(other, EnergyTerm):
            new_energy_term = EnergyTerm()
            if self.use_numba and other.use_numba:
                new_energy_term.use_numba = True
            else:
                new_energy_term.use_numba = False
            
            e1=self.energy_function; e2=other.energy_function
            new_energy_term.energy_function = new_energy_term.numbify(lambda seq_index: e1(seq_index) * e2(seq_index), cache=True)
            
            m1=self.denergy_mutation_function; m2=other.denergy_mutation_function
            new_energy_term.denergy_mutation_function = new_energy_term.numbify(lambda seq_index,pos,aa: m1(seq_index,pos,aa) * m2(seq_index,pos,aa), cache=True)
            
            s1=self.denergy_swap_function; s2=other.denergy_swap_function
            new_energy_term.denergy_swap_function = new_energy_term.numbify(lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) * s2(seq_index,pos1,pos2), cache=True)
            return new_energy_term
        elif isinstance(other, (int, float)):
            new_energy_term = EnergyTerm()
            new_energy_term.use_numba = self.use_numba
            
            e1=self.energy_function
            new_energy_term.energy_function = new_energy_term.numbify(lambda seq_index: e1(seq_index) * other, cache=True)
            
            m1=self.denergy_mutation_function
            new_energy_term.denergy_mutation_function = new_energy_term.numbify(lambda seq_index,pos,aa: m1(seq_index,pos,aa) * other, cache=True)
            
            s1=self.denergy_swap_function
            new_energy_term.denergy_swap_function = new_energy_term.numbify(lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) * other, cache=True)
            return new_energy_term
        
        def __sub__(self, other):
            if isinstance(other, EnergyTerm):
                new_energy_term = EnergyTerm()
                if self.use_numba and other.use_numba:
                    new_energy_term.use_numba = True
                else:
                    new_energy_term.use_numba = False
                
                e1=self.energy_function; e2=other.energy_function
                new_energy_term.energy_function = new_energy_term.numbify(lambda seq_index: e1(seq_index) - e2(seq_index), cache=True)
                
                m1=self.denergy_mutation_function; m2=other.denergy_mutation_function
                new_energy_term.denergy_mutation_function = new_energy_term.numbify(lambda seq_index,pos,aa: m1(seq_index,pos,aa) - m2(seq_index,pos,aa), cache=True)
                
                s1=self.denergy_swap_function; s2=other.denergy_swap_function
                new_energy_term.denergy_swap_function = new_energy_term.numbify(lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) - s2(seq_index,pos1,pos2), cache=True)
                return new_energy_term
            elif isinstance(other, (int, float)):
                new_energy_term = EnergyTerm()
                new_energy_term.use_numba = self.use_numba
                
                e1=self.energy_function
                new_energy_term.energy_function = new_energy_term.numbify(lambda seq_index: e1(seq_index) - other, cache=True)
                
                m1=self.denergy_mutation_function
                new_energy_term.denergy_mutation_function = new_energy_term.numbify(lambda seq_index,pos,aa: m1(seq_index,pos,aa) - other, cache=True)
                
                s1=self.denergy_swap_function
                new_energy_term.denergy_swap_function = new_energy_term.numbify(lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) - other, cache=True)
                return new_energy_term
            
    def __truediv__(self, other):
        if isinstance(other, EnergyTerm):
            new_energy_term = EnergyTerm()
            if self.use_numba and other.use_numba:
                new_energy_term.use_numba = True
            else:
                new_energy_term.use_numba = False
            
            e1=self.energy_function; e2=other.energy_function
            new_energy_term.energy_function = new_energy_term.numbify(lambda seq_index: e1(seq_index) / e2(seq_index), cache=True)
            
            m1=self.denergy_mutation_function; m2=other.denergy_mutation_function
            new_energy_term.denergy_mutation_function = new_energy_term.numbify(lambda seq_index,pos,aa: m1(seq_index,pos,aa) / m2(seq_index,pos,aa), cache=True)
            
            s1=self.denergy_swap_function; s2=other.denergy_swap_function
            new_energy_term.denergy_swap_function = new_energy_term.numbify(lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) / s2(seq_index,pos1,pos2), cache=True)
            return new_energy_term
        elif isinstance(other, (int, float)):
            new_energy_term = EnergyTerm()
            new_energy_term.use_numba = self.use_numba
            
            e1=self.energy_function
            new_energy_term.energy_function = new_energy_term.numbify(lambda seq_index: e1(seq_index) / other, cache=True)
            
            m1=self.denergy_mutation_function
            new_energy_term.denergy_mutation_function = new_energy_term.numbify(lambda seq_index,pos,aa: m1(seq_index,pos,aa) / other, cache=True)
            
            s1=self.denergy_swap_function
            new_energy_term.denergy_swap_function = new_energy_term.numbify(lambda seq_index,pos1,pos2: s1(seq_index,pos1,pos2) / other, cache=True)
            return new_energy_term

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __new__(cls, *args, **kwargs):
        new_instance = super().__new__(cls)
        new_instance.use_numba = True
        return new_instance
        
    def test_energy(self,seq_index=np.array([0,1,2,3,4])):
        assert type(self.energy(seq_index)) in [float,np.float64], "Energy function should return a float"

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

if __name__ == "__main__":
    et = EnergyTerm()
    et.test()
