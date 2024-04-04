from .. import pdb
import prody
import os
import Bio.PDB.Polypeptide as poly
import numpy as np
from typing import Union
from pathlib import Path

__all__ = ['Structure']

residue_names=[]

class Structure:
    
    @classmethod
    def full_pdb(cls,pdb_file: Union[Path,str], chain: Union[str,None]=None, aligned_sequence: str = None, filtered_aligned_sequence: str = None,
                distance_matrix_method:str = 'CB', pdb_directory: Path = Path.cwd(), repair_pdb:bool = False):

        """
        Generates structure object 

        Parameters
        ----------
        pdb_file :  str
            PDB file path

        chain : str
            PDB chain name. If "chain=None", all chains will be included.
            
        distance_matrix_method: str
            The method to use for calculating the distance matrix. 
            Defaults to 'CB', which uses the CB atom for all residues except GLY, which uses the CA atom. 
            Other options are 'CA' for using only the CA atom, 
            and 'minimum' for using the minimum distance between all atoms in each residue.    

        pdb_directory: str
            Directory where repaired pdb will be downloaded

        repair_pdb: bool
            If False, provided pdb file will be repaired with missing residues inserted and heteroatoms removed.

        Returns
        -------
        Structure object
        """
        self=cls()

        try:
            #Check if file exists
            pdb_file=Path(pdb_file)
            assert pdb_file.exists()
        except AssertionError:
            #Attempt to download pdb file
            pdb_file=str(pdb_file)
            if len(pdb_file)==4:
                self.pdbID=pdb_file
                print(f"Downloading {self.pdbID} from the PDB")
                pdb_file=pdb.download(self.pdbID, pdb_directory)
            else:
                raise FileNotFoundError(f"Provided pdb file {pdb_file} does not exist")

        
        self.pdbID=pdb_file.stem
        self.pdb_file=pdb_file
        self.chain=chain
        self.distance_matrix_method=distance_matrix_method
        self.filtered_aligned_sequence=filtered_aligned_sequence
        self.aligned_sequence=aligned_sequence
        self.init_index_shift=0
        
        if repair_pdb:
            fixer=pdb.repair_pdb(pdb_file, chain, pdb_directory)
            self.pdb_file=str(pdb_directory/f"{self.pdbID}_cleaned.pdb")

        if chain is None:
            self.structure = prody.parsePDB(str(self.pdb_file)).select('protein')
        else:
            self.structure = prody.parsePDB(str(self.pdb_file), chain=self.chain).select('protein')
        self.sequence=pdb.get_sequence(self.pdb_file,self.chain)
        self.distance_matrix=pdb.get_distance_matrix(pdb_file=self.pdb_file,chain=self.chain,
                                                     method=self.distance_matrix_method)

        if self.aligned_sequence is not None:
            self.full_to_aligned_index_dict=pdb.full_to_filtered_aligned_mapping(self.aligned_sequence,self.filtered_aligned_sequence)
            self.mapped_distance_matrix=np.full((len(self.filtered_aligned_sequence), len(self.filtered_aligned_sequence)), np.inf)
            pos1, pos2 = np.meshgrid(list(self.full_to_aligned_index_dict.keys()), list(self.full_to_aligned_index_dict.keys()), 
                                    indexing='ij', sparse=True)
            modpos1, modpos2 = np.meshgrid(list(self.full_to_aligned_index_dict.values()), list(self.full_to_aligned_index_dict.values()), 
                                    indexing='ij', sparse=True)
            self.mapped_distance_matrix[modpos1,modpos2]=self.distance_matrix[pos1,pos2]
            np.fill_diagonal(self.mapped_distance_matrix, 0)

        else:
            self.full_to_aligned_index_dict=dict(zip(range(len(self.sequence)), range(len(self.sequence))))
            self.mapped_distance_matrix=self.distance_matrix

        return self
    
    @classmethod
    def spliced_pdb(cls,pdb_file:Union[Path,str], chain:str, seq_selection: str, aligned_sequence: str = None,
                    filtered_aligned_sequence: str = None, distance_matrix_method:str = 'CB',repair_pdb:bool = True, 
                    pdb_directory: str = os.getcwd()):
        """
        Generates substructure object 

        Parameters
        ----------
        pdb_file :  str
            PDB file path

        chain : str
            PDB chain name

        seq_selection: str
            Subsequence selection command, using Prody select module. 

            *If wanting to use original PDB indexing, 
            set seq_selection as: "resnum `{initial_index}to{final_index}`"

            *If wanting to use absolute PDB indexing (first residue has index=0),
            set seq_selection as: "resindex `{initial_index}to{final_index}`"

            Note that using "to" will create a substructure including both the initial and final designated residue.
            If you would like to not include the final desginated residue, replace "to" with ":"

            Note that the algorithm will account for any gaps in the PDB and adjust the provided residue
            range accordingly.
            
        distance_matrix_method: str
            The method to use for calculating the distance matrix. 
            Defaults to 'CB', which uses the CB atom for all residues except GLY, which uses the CA atom. 
            Other options are 'CA' for using only the CA atom, 
            and 'minimum' for using the minimum distance between all atoms in each residue.    

        pdb_directory: str
            Directory where repaired pdb will be downloaded

        repair_pdb: bool
            If True, provided pdb file will be repaired with missing residues inserted and heteroatoms removed.

        Returns
        -------
        Substructure object
        """
        #Provide the indices according to the original pdb numbering
        self=cls()
        try:
            #Check if file exists
            pdb_file=Path(pdb_file)
            assert pdb_file.exists()
        except AssertionError:
            #Attempt to download pdb file
            pdb_file=str(pdb_file)
            if len(pdb_file)==4:
                self.pdbID=pdb_file
                print(f"Downloading {self.pdbID} from the PDB")
                pdb_file=pdb.download(self.pdbID, pdb_directory)
            else:
                raise FileNotFoundError(f"Provided pdb file {pdb_file} does not exist")

        
        self.pdbID=pdb_file.stem
        self.pdb_file=pdb_file
        self.chain=chain
        self.seq_selection=seq_selection
        self.init_index=int(self.seq_selection.replace("to"," to ").replace(":"," : ").split()[1].replace("`",""))
        self.fin_index=int(self.seq_selection.replace("to"," to ").replace(":"," : ").split()[3].replace("`",""))
        self.distance_matrix_method=distance_matrix_method
        self.filtered_aligned_sequence=filtered_aligned_sequence
        self.aligned_sequence=aligned_sequence

        assert len(self.seq_selection.replace("to"," to ").replace(":"," : ").split())>=4, "Please correctly input your residue selection"

        if self.chain==None:
            raise ValueError("Please provide a chain name")

        #Account for pdbs that have starting indices greater than one and find any gaps in the pdb.
        gap_indices=[]; atom_line_count=0
        with open(pdb_file,"r") as f:
            for line in f:
                if line.split()[0]=="ATOM" and line.split()[4]==self.chain:
                    try:
                        res_index=''.join(i for i in line.split()[5] if i.isdigit())
                        next_res_index=''.join(i for i in next(f).split()[5] if i.isdigit())
                        if int(next_res_index)-int(res_index)>1:
                            gap_indices.extend(list(range(int(res_index)+1,int(next_res_index))))
                        if atom_line_count==0 and poly.is_aa(line.split()[3]):
                            self.pdb_init_index=int(line.split()[5])
                        atom_line_count+=1
                    except:
                        continue

        if "resnum" in self.seq_selection:
            assert self.init_index>=self.pdb_init_index, "Please pick an initial index within the pdb's original indices"
            self.init_index_shift=self.init_index-self.pdb_init_index
            self.fin_index_shift=self.fin_index-self.pdb_init_index+1
            if repair_pdb:
                fixer=pdb.repair_pdb(pdb_file, chain, pdb_directory)
                self.pdb_file=f"{pdb_directory}/{self.pdbID}_cleaned.pdb"
                self.select_gap_indices=[i for i in gap_indices if self.init_index<=i<=self.fin_index]
                self.fin_index_shift-=len(self.select_gap_indices)
                self.seq_selection=f"resnum `{self.init_index_shift+1}to{self.fin_index_shift}`"
                #Account for missing residues in beginning of PDB
                # keys = fixer.missingResidues.keys()
                # miss_init_residues=[i for i in keys if i[1]==0]
                # self.init_index=init_index+len(miss_init_residues)-self.pdb_init_index+1
                # self.fin_index=fin_index+len(miss_init_residues)-self.pdb_init_index+1
                # self.init_index_shift=self.init_index-1
                # self.fin_index_shift=self.fin_index
        elif "resindex" in self.seq_selection:
            self.init_index_shift=self.init_index
            self.fin_index_shift=self.fin_index+1
            if repair_pdb:
                fixer=pdb.repair_pdb(pdb_file, chain, pdb_directory)
                self.pdb_file=f"{pdb_directory}/{self.pdbID}_cleaned.pdb"
                self.chain="A"
                # #Account for missing residues in beginning of PDB
                # keys = fixer.missingResidues.keys()
                # miss_init_residues=[i for i in keys if i[1]==0]
                # self.init_index=init_index+len(miss_init_residues)
                # self.fin_index=fin_index+len(miss_init_residues)
                # self.init_index_shift=self.init_index
                # self.fin_index_shift=self.fin_index+1  
    

        self.structure = prody.parsePDB(str(self.pdb_file), chain=self.chain).select(f"protein and {self.seq_selection}")
        self.sequence=pdb.get_sequence(self.pdb_file,self.chain)

        self.distance_matrix=pdb.get_distance_matrix(pdb_file=self.pdb_file,chain=self.chain,
                                                     method=self.distance_matrix_method)

        self.distance_matrix=self.distance_matrix[self.init_index_shift:self.fin_index_shift,
                                                  self.init_index_shift:self.fin_index_shift]
        self.sequence=self.sequence[self.init_index_shift:self.fin_index_shift]

        if self.filtered_aligned_sequence is not None:
            self.full_to_aligned_index_dict=pdb.full_to_filtered_aligned_mapping(self.aligned_sequence,self.filtered_aligned_sequence)
            self.mapped_distance_matrix=np.full((len(self.filtered_aligned_sequence), len(self.filtered_aligned_sequence)), np.inf)
            # self.full_to_aligned_index_dict = {key-1:value for (key,value) in self.full_to_aligned_index_dict.items()}
            pos1, pos2 = np.meshgrid(list(self.full_to_aligned_index_dict.keys()), list(self.full_to_aligned_index_dict.keys()), 
                                    indexing='ij', sparse=True)
            modpos1, modpos2 = np.meshgrid(list(self.full_to_aligned_index_dict.values()), list(self.full_to_aligned_index_dict.values()), 
                                    indexing='ij', sparse=True)
            self.mapped_distance_matrix[modpos1,modpos2]=self.distance_matrix[pos1,pos2]
            np.fill_diagonal(self.mapped_distance_matrix, 0)

        else:
            self.full_to_aligned_index_dict=dict(zip(range(self.init_index_shift,self.fin_index_shift+1), range(len(self.sequence))))
            self.mapped_distance_matrix=self.distance_matrix
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
