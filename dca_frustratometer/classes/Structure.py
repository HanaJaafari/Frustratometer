from .. import pdb
import prody
import os
import subprocess
import Bio.PDB.Polypeptide as poly

__all__ = ['Structure']

residue_names=[]

class Structure:
    
    @classmethod
    def full_pdb(cls,pdb_file:str, chain:str, distance_matrix_method:str = 'CB', pdb_directory: str=os.getcwd(),
                 repair_pdb:bool = False):
        self=cls()
        if pdb_file[-4:]!=".pdb":
            self.pdbID=pdb_file
            pdb_file=pdb.download(self.pdbID, pdb_directory)
        
        self.pdb_file=pdb_file
        self.pdbID=os.path.basename(pdb_file).replace(".pdb","")
        self.chain=chain
        self.distance_matrix_method=distance_matrix_method
        self.init_index=1

        if self.chain==None:
            raise ValueError("Please provide a chain name")
        
        if repair_pdb:
            fixer=pdb.repair_pdb(pdb_file, chain, pdb_directory)
            self.pdb_file=f"{pdb_directory}/{self.pdbID}_cleaned.pdb"

        self.structure = prody.parsePDB(self.pdb_file, chain=self.chain).select('protein')
        self.sequence=pdb.get_sequence(self.pdb_file,self.chain)
        self.distance_matrix=pdb.get_distance_matrix(pdb_file=self.pdb_file,chain=self.chain,
                                                     method=self.distance_matrix_method)
        return self
    
    @classmethod
    def spliced_pdb(cls,pdb_file:str, chain:str, init_index:int, fin_index:int, 
                    distance_matrix_method:str = 'CB', pdb_directory: str = os.getcwd(),repair_pdb:bool = False):
        #Provide the indices according to the original pdb numbering
        self=cls()
        if pdb_file[-4:]!=".pdb":
            self.pdbID=pdb_file
            pdb_file=pdb.download(self.pdbID, pdb_directory)
        #TODO:
        #put assertion error for findex greater than protein length

        self.pdb_file=pdb_file
        self.pdbID=os.path.basename(pdb_file).replace(".pdb","")
        self.chain=chain
        self.distance_matrix_method=distance_matrix_method

        if self.chain==None:
            raise ValueError("Please provide a chain name")

        #Account for pdbs that have starting indices greater than one.
        with open(pdb_file,"r") as f:
            for line in f:
                if line.split()[0]=="ATOM" and poly.is_aa(line.split()[3]):
                    self.pdb_init_index=int(line.split()[5])
                    break

        if not repair_pdb:
            self.init_index=init_index
            self.fin_index=fin_index
            self.init_index_shift=(self.init_index-self.pdb_init_index)
            self.fin_index_shift=self.fin_index-self.pdb_init_index+1
        else:
            fixer=pdb.repair_pdb(pdb_file, chain, pdb_directory)
            #Account for missing residues in beginning of PDB
            keys = fixer.missingResidues.keys()
            miss_init_residues=[i for i in keys if i[1]==0]
            self.pdb_file=f"{pdb_directory}/{self.pdbID}_cleaned.pdb"
            self.init_index=init_index+len(miss_init_residues)-self.pdb_init_index+1
            self.fin_index=fin_index+len(miss_init_residues)-self.pdb_init_index+1
            self.init_index_shift=self.init_index-1
            self.fin_index_shift=self.fin_index
        

        self.structure = prody.parsePDB(self.pdb_file, chain=self.chain).select(f"protein and resnum `{str(self.init_index)}to{str(self.fin_index)}`")
        self.sequence=pdb.get_sequence(self.pdb_file,self.chain)
        self.distance_matrix=pdb.get_distance_matrix(pdb_file=self.pdb_file,chain=self.chain,
                                                     method=self.distance_matrix_method)

        self.distance_matrix=self.distance_matrix[self.init_index_shift:self.fin_index_shift,
                                                  self.init_index_shift:self.fin_index_shift]
        self.sequence=self.sequence[self.init_index_shift:self.fin_index_shift]
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
