from Bio.PDB import PDBParser
import prody
import scipy.spatial.distance as sdist
import pandas as pd
import numpy as np
import itertools
import os
from pathlib import Path

three_to_one = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
                'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I',
                'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
                'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}


def download(pdbID: str,directory: Union[Path,str]=Path.cwd()) -> Path:
    """
    Downloads a single pdb file

    Parameters
    ----------
    pdbID: str,
        PDB ID
    directory: Path or str,
        Directory where PDB file will be downloaded.

    Returns
    -------
    pdb_file : Path
        PDB file location.
    """

    import urllib.request
    pdb_file=Path(directory) / f'{pdbID}.pdb'
    urllib.request.urlretrieve('http://www.rcsb.org/pdb/files/%s.pdb' % pdbID, pdb_file)
    return pdb_file

def get_sequence(pdb_file: str, 
                 chain: str
                 ) -> str:
    """
    Get a protein sequence from a pdb file

    Parameters
    ----------
    pdb_file : str,
        PDB file location.
    chain: str,
        Chain ID of the selected protein.

    Returns
    -------
    sequence : str
        Protein sequence.
    """
    """
    Get a protein sequence from a PDB file
    
    :param pdb: PDB file location
    :param chain: chain name of PDB file to get sequence
    :return: protein sequence
    """

    parser = PDBParser()
    structure = parser.get_structure('name', pdb_file)
    if chain==None:
        all_chains=[i.get_id() for i in structure.get_chains()]
    else:
        all_chains=[chain]
    sequence = ""
    for chain in all_chains:
        c = structure[0][chain]
        chain_seq = ""
        for residue in c:
            is_regular_res = residue.has_id('CA') and residue.has_id('O')
            res_id = residue.get_id()[0]
            if (res_id==' ' or res_id=='H_MSE' or res_id=='H_M3L' or res_id=='H_CAS') and is_regular_res:
                residue_name = residue.get_resname()
                chain_seq += three_to_one[residue_name]
        sequence += chain_seq
    return sequence



def get_distance_matrix(pdb_file: Union[Path,str],
                        chain: str,
                        method: str = 'CB'
                        ) -> np.array:
    """
    Calculate the distance matrix of the specified atoms in a PDB file.

    Parameters
    ----------
    pdb_file: Path or str 
        The path to the PDB file.
    chain: str
        The chainID or chainIDs (space separated) of the protein.
    method: str
        The method to use for calculating the distance matrix. 
        Defaults to 'CB', which uses the CB atom for all residues except GLY, which uses the CA atom. 
        Options:
            'CA' for using only the CA atom,
            'minimum' for using the minimum distance between all atoms in each residue,
            'CB_force' computes a new coordinate for the CB atom based on the CA, C, and N atoms and uses CB distance even for glycine.

    Returns:
        np.array: The distance matrix of the selected atoms.

    Raises:
        IndexError: If the selection of atoms is empty.
        ValueError: If the method is not recognized.
    """

    structure = prody.parsePDB(str(pdb_file))
    chain_selection = '' if chain is None else f' and chain {chain}'

    if method == 'CA':
        coords = structure.select('protein and name CA' + chain_selection).getCoords()
    elif method == 'CB':
        coords = structure.select('(protein and (name CB) or (resname GLY and name CA))' + chain_selection).getCoords()
    elif method == 'minimum':
        selection = structure.select('protein' + chain_selection)
        coords = selection.getCoords()
        distance_matrix = sdist.squareform(sdist.pdist(coords))
        resids = selection.getResindices()
        residues = pd.Series(resids).unique()
        selections = np.array([resids == a for a in residues])
        dm = np.zeros((len(residues), len(residues)))
        for i, j in itertools.combinations(range(len(residues)), 2):
            d = distance_matrix[selections[i]][:, selections[j]].min()
            dm[i, j] = d
            dm[j, i] = d
        return dm
    elif method == 'CB_force':
        sel_CA = structure.select('name CA')
        sel_N = structure.select('name N')
        sel_C = structure.select('name C')
        
        # Base vectors
        vector_CA_C=sel_C.getCoords() - sel_CA.getCoords()
        vector_CA_N=sel_N.getCoords() - sel_CA.getCoords()

        vector_CA_C/=np.linalg.norm(vector_CA_C,axis=1,keepdims=True)
        vector_CA_N/=np.linalg.norm(vector_CA_N,axis=1,keepdims=True)

        #First Ortogonal vector
        cross_CA_C_CA_N=np.cross(vector_CA_C,vector_CA_N)
        cross_CA_C_CA_N/=np.linalg.norm(cross_CA_C_CA_N,axis=1,keepdims=True)

        #Second Ortogonal vector
        cross_cross_CA_N=np.cross(cross_CA_C_CA_N,vector_CA_N)
        cross_cross_CA_N/= np.linalg.norm(cross_cross_CA_N,axis=1,keepdims=True)
        
        #Precomputed CB coordinates
        coords = -0.531020*vector_CA_N-1.206181*cross_CA_C_CA_N+0.789162*cross_cross_CA_N+sel_CA.getCoords()
    else:
        raise ValueError(f"Invalid method '{method}'. Accepted methods are 'CA', 'CB', 'minimum', and 'CB_force'.")

    if len(coords) == 0:
        raise IndexError('Empty selection for distance map')

    distance_matrix = sdist.squareform(sdist.pdist(coords))
    return distance_matrix


def full_to_filtered_aligned_mapping(aligned_sequence: str,
                                    filtered_aligned_sequence: str)->dict:

    """
    Get a dictionary mapping residue positions in the full pdb sequence to the aligned pdb sequence

    Parameters
    ----------
    aligned_sequence : str,
        Raw aligned PDB sequence.
    filtered_aligned_sequence: str,
        Filtered aligned PDB sequence (columns with insertions and deletions, i.e. dashes, that are
        typically filtered in MSA file processing are removed)

    Returns
    -------
    full_to_aligned_index_dict : dict
        Dictionary
    """
    full_to_aligned_index_dict={}; counter=0
    for i,x in enumerate(aligned_sequence):
        if x != "-" and x==x.upper():
            full_to_aligned_index_dict[counter]=i
        if x!="-":
            counter+=1

    dash_indices=[i for i,x in enumerate(filtered_aligned_sequence) if x!="-"]
    counter=0
    for entry in full_to_aligned_index_dict:
        full_to_aligned_index_dict[entry]=dash_indices[counter]
        counter+=1

    return full_to_aligned_index_dict