from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import prody
import scipy.spatial.distance as sdist
import pandas as pd
import numpy as np
import itertools
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
import os

def download(pdbID: str,directory: str):
    """
    Downloads a single pdb file
    """
    import urllib.request
    pdb_file="%s%s.pdb" % (directory, pdbID)
    urllib.request.urlretrieve('http://www.rcsb.org/pdb/files/%s.pdb' % pdbID, pdb_file)
    return pdb_file

def get_sequence(pdb_file: str, 
                 chain: str
                 ) -> str:
    """
    Get a protein sequence from a pdb file

    Parameters
    ----------
    pdb : str,
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
    Letter_code = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                   'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                   'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                   'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
                   'NGP': 'A', 'IPR': 'P', 'IGL': 'G'}
    ppb=PPBuilder()
    sequence=ppb.build_peptides(structure[0][chain])[0].get_sequence()

    # if chain is None:
    #     # If chain is None then chain can be any chain
    #     residues = [residue for residue in structure.get_residues() if (
    #                     residue.has_id('CA') and
    #                     residue.resname not in [' CA','PBC'])]

    # else:
    #     residues = [residue for residue in structure.get_residues() if (
    #                     residue.has_id('CA') and
    #                     residue.get_parent().get_id() == str(chain) and 
    #                     residue.resname not in [' CA','PBC','NDP'])]
        

    # sequence = ''.join([Letter_code[r.resname] for r in residues])
    return sequence

def repair_pdb(pdb_file: str, chain: str, pdb_directory: str):
    pdbID=os.path.basename(pdb_file).replace(".pdb","")
    fixer = PDBFixer(pdb_file)

    chains = list(fixer.topology.chains())
    chains_to_remove = [i for i, x in enumerate(chains) if x.id not in chain]
    fixer.removeChains(chains_to_remove)

    fixer.findMissingResidues()
    #Filling in missing residues inside chain
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain_tmp = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain_tmp.residues())):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    try:
        fixer.addMissingAtoms()
    except:
        print("Unable to add missing atoms")

    fixer.addMissingHydrogens(7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(f"{pdb_directory}/{pdbID}_cleaned.pdb", 'w'))
    return fixer

def get_distance_matrix(pdb_file: str, 
                        chain: str, 
                        method: str = 'CB'
                        ) -> np.array:
    """
    Calculate the distance matrix of the specified atoms in a PDB file.
    
    Parameters:
        pdb_file (str): The path to the PDB file.
        chain (str): The chainID or chainIDs (space separated) of the protein.
        method (str): The method to use for calculating the distance matrix. 
                      Defaults to 'CB', which uses the CB atom for all residues except GLY, which uses the CA atom. 
                      Options:
                       'CA' for using only the CA atom,
                       'minimum' for using the minimum distance between all atoms in each residue
                       'CB_force' computes a new coordinate for the CB atom bassed on the CA, C and N atoms and uses CB distance even for glycine


    
    Returns:
        np.array: The distance matrix of the selected atoms.
    
    Raises:
        IndexError: If the selection of atoms is empty.
    """
    
    structure = prody.parsePDB(pdb_file)
    chain_selection = '' if chain is None else f' and chain {chain}'
    if method == 'CA':
        selection = structure.select('protein and name CA' + chain_selection)
        if len(selection) == 0:
            raise IndexError('Empty selection')
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        return distance_matrix
    elif method == 'CB':
        selection = structure.select('(protein and (name CB) or (resname GLY and name CA))' + chain_selection)
        if len(selection) == 0:
            raise IndexError('Empty selection')
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        return distance_matrix
    elif method == 'minimum':
        selection = structure.select('protein' + chain_selection)
        if len(selection) == 0:
            raise IndexError('Empty selection')
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
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
        new_CB = -0.531020*vector_CA_N-1.206181*cross_CA_C_CA_N+0.789162*cross_cross_CA_N+sel_CA.getCoords()

        distance_matrix = sdist.squareform(sdist.pdist(new_CB))
        return distance_matrix

