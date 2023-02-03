from .. import pdb
import prody
import os
import numpy as np
import subprocess

__all__ = ['Structure']

class Structure:
    
    @classmethod
    def full_pdb(cls,pdb_file:str, chain:str, distance_matrix_method:str = 'CB', pdb_directory: str=os.getcwd()):
        self=cls()
        if pdb_file[-4:]!=".pdb":
            self.pdbID=pdb_file
            pdb_file=pdb.download(self.pdbID, pdb_directory)
        
        self.pdb_file=pdb_file
        self.pdbID=os.path.basename(pdb_file).replace(".pdb","")
        self.chain=chain
        self.distance_matrix_method=distance_matrix_method

        self.structure = prody.parsePDB(self.pdb_file, chain=self.chain).select('protein')
        self.sequence=pdb.get_sequence(self.pdb_file,self.chain)
        self.distance_matrix=pdb.get_distance_matrix(pdb_file=self.pdb_file,chain=self.chain,
                                                     method=self.distance_matrix_method)
        return self
    
    @classmethod
    def spliced_pdb(cls,pdb_file:str, chain:str, init_index:int, fin_index:int, 
                    distance_matrix_method:str = 'CB', pdb_directory: str = os.getcwd()):
        self=cls()
        if pdb_file[-4:]!=".pdb":
            self.pdbID=pdb_file
            pdb_file=pdb.download(self.pdbID, pdb_directory)
        #TODO:
        #remove heteroatoms, fix pdb indexing for those starting off at values other than 1, 
        #put assertion error for findex greater than protein length

        self.pdb_file=pdb_file
        self.pdbID=os.path.basename(pdb_file).replace(".pdb","")
        self.chain=chain
        self.distance_matrix_method=distance_matrix_method

        #Account for pdbs that have starting indices greater than one.
        pdb_init_index=subprocess.check_output(["grep","-m","1","^ATOM",self.pdb_file])
        self.pdb_init_index=int(pdb_init_index.decode().split()[5])

        self.structure = prody.parsePDB(self.pdb_file, chain=self.chain).select(f'resnum {str(init_index+self.pdb_init_index)}to{str(fin_index+self.pdb_init_index)}')
        self.sequence=pdb.get_sequence(self.pdb_file,self.chain)
        self.distance_matrix=pdb.get_distance_matrix(pdb_file=self.pdb_file,chain=self.chain,
                                                     method=self.distance_matrix_method)

        self.init_index=init_index
        self.fin_index=fin_index

        self.distance_matrix=self.distance_matrix[self.init_index:self.fin_index+1,self.init_index:self.fin_index+1]
        self.sequence=self.sequence[init_index:fin_index+1]
        return self

    # @property
    # def sequence(self):
    #     return self._sequence
    
    # # Set a new sequence in case someone needs to calculate the energy of a diferent sequence with the same structure
    # @sequence.setter
    # def sequence(self, value :str):
    #     assert len(value) == len(self._sequence)
    #     self._sequence = value

    # @property
    # def pdb_file(self):
    #     return str(self._pdb_file)
    
    # @pdb_file.setter
    # def pdb_file(self,value: str):
    #     self._pdb_file=value

    # @property
    # def chain(self):
    #     return self._chain
    
    # @chain.setter
    # def chain(self,value):
    #     self._chain=value
    
    # @property
    # def distance_matrix_method(self):
    #     return self._distance_matrix_method
    
    # @distance_matrix_method.setter
    # def distance_matrix_method(self,value):
    #     self._distance_matrix_method = value

    # @property
    # def init_index(self):
    #     return self._init_index
    
    # @init_index.setter
    # def init_index(self,value):
    #     self._init_index = value

    # @property
    # def fin_index(self):
    #     return self._fin_index
    
    # @fin_index.setter
    # def fin_index(self,value):
    #     self._fin_index = value
