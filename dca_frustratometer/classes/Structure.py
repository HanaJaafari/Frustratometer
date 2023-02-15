from .. import pdb
from .. import frustration
import prody
import os
import Bio.PDB.Polypeptide as poly
from Bio.PDB import *

__all__ = ['Structure']

residue_names=[]

class Structure:
    
    @classmethod
    def full_pdb(cls,pdb_file:str, chain:str, distance_matrix_method:str = 'CB', pdb_directory: str=os.getcwd(),
                 repair_pdb:bool = True):
        """
        Generates structure object 

        Parameters
        ----------
        pdb_file :  str
            PDB file path

        chain : str
            PDB chain name

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
        Structure object
        """
        self=cls()
        if pdb_file[-4:]!=".pdb":
            self.pdbID=pdb_file
            pdb_file=pdb.download(self.pdbID, pdb_directory)
        
        self.pdb_file=pdb_file
        self.pdbID=os.path.basename(pdb_file).replace(".pdb","")
        self.chain=chain
        self.distance_matrix_method=distance_matrix_method
        self.init_index_shift=0

        if self.chain==None:
            raise ValueError("Please provide a chain name")
        
        if repair_pdb:
            fixer=pdb.repair_pdb(pdb_file, chain, pdb_directory)
            self.pdb_file=f"{pdb_directory}/{self.pdbID}_cleaned.pdb"

        self.structure = prody.parsePDB(self.pdb_file, chain=self.chain).select('protein')
        self.sequence=pdb.get_sequence(self.pdb_file,self.chain)
        self.distance_matrix=pdb.get_distance_matrix(pdb_file=self.pdb_file,chain=self.chain,
                                                     method=self.distance_matrix_method)
        
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)  
        return self
    
    @classmethod
    def spliced_pdb(cls,pdb_file:str, chain:str, seq_selection: str, 
                    distance_matrix_method:str = 'CB',repair_pdb:bool = True,pdb_directory: str = os.getcwd()):
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
        if pdb_file[-4:]!=".pdb":
            self.pdbID=pdb_file
            pdb_file=pdb.download(self.pdbID, pdb_directory)

        self.pdb_file=pdb_file
        self.pdbID=os.path.basename(pdb_file).replace(".pdb","")
        self.chain=chain
        self.distance_matrix_method=distance_matrix_method

        assert self.chain!=None, "Please provide a chain name"

        self.seq_selection=seq_selection
        self.init_index=int(self.seq_selection.replace("to"," to ").replace(":"," : ").split()[1].replace("`",""))
        self.fin_index=int(self.seq_selection.replace("to"," to ").replace(":"," : ").split()[3].replace("`",""))
        
        assert len(self.seq_selection.replace("to"," to ").replace(":"," : ").split())>=4, "Please correctly input your residue selection"

        #Account for pdbs that have starting indices greater than one and find any gaps in the pdb.
        p = PDBParser()
        pdb_structure = p.get_structure("X",self.pdb_file)
        res_list = Selection.unfold_entities(pdb_structure[0][self.chain], "R")
        all_pdb_indices=[int(i.__repr__().split()[3].split("=")[-1]) for i in res_list if len(i.__repr__().split()[2].split("="))==2]
        self.pdb_init_index=all_pdb_indices[0]
        all_correct_pdb_indices=[x for x in range(all_pdb_indices[0], all_pdb_indices[-1] + 1)]
        gap_indices=list(set(all_pdb_indices) ^ set(all_correct_pdb_indices))

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
                # #Account for missing residues in beginning of PDB
                # keys = fixer.missingResidues.keys()
                # miss_init_residues=[i for i in keys if i[1]==0]
                # self.init_index=init_index+len(miss_init_residues)
                # self.fin_index=fin_index+len(miss_init_residues)
                # self.init_index_shift=self.init_index
                # self.fin_index_shift=self.fin_index+1  
    
        self.structure = prody.parsePDB(self.pdb_file, chain=self.chain).select(f"protein and {self.seq_selection}")
        self.sequence=pdb.get_sequence(self.pdb_file,self.chain)
        self.distance_matrix=pdb.get_distance_matrix(pdb_file=self.pdb_file,chain=self.chain,
                                                     method=self.distance_matrix_method)
        #Splice distance matrix
        self.distance_matrix=self.distance_matrix[self.init_index_shift:self.fin_index_shift,
                                                  self.init_index_shift:self.fin_index_shift]

        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)
        #Splice sequence
        self.sequence=self.sequence[self.init_index_shift:self.fin_index_shift]     
        return self
    
