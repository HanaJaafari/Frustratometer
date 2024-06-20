import pandas as pd
from ..utils import _path
import os

def get_pfamID(pdb_file, chain):
    """
    Returns PFAM and Uniprot IDs

    Parameters
    ----------
    pdb_file :  str
        pdb file name
    chain : str
        Select chain from pdb
    Returns
    -------
    pfamID : str
        PFAM family ID
    """

    # TODO fix function
    # TODO appdirs?
    pdbID=os.path.basename(pdb_file)[:4]
    df = pd.read_csv(f'{_path}/data/pdb_chain_pfam.csv', header=1)
    if sum((df['PDB'] == pdbID.lower()) & (df['CHAIN'] == chain.upper())) != 0:
        #Assumes one domain for the PDB
        pfamID = df.loc[(df['PDB'] == pdbID.lower()) & (df['CHAIN'] == chain.upper())]["PFAM_ID"].values[0]
    else:
        print('PFAM ID is unavailable')
        pfamID = 'null'
    return pfamID


def get_pfam_map(pdbID, chain):
    # TODO fix function
    import pandas as pd
    df = pd.read_table('%s/pdb_pfam_map.txt' % basedir, header=0)
    if sum((df['PDB_ID'] == pdbID.upper()) & (df['CHAIN_ID'] == chain.upper())) != 0:
        start = df.loc[(df['PDB_ID'] == pdbID.upper()) & (df['CHAIN_ID'] == chain.upper())]["PdbResNumStart"].values[0]
        end = df.loc[(df['PDB_ID'] == pdbID.upper()) & (df['CHAIN_ID'] == chain.upper())]["PdbResNumEnd"].values[0]
    else:
        print('data not found')
        pfamID = 'null'
    return int(start), int(end)

