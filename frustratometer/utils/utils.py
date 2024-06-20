from pathlib import Path
import logging

def create_directory(path):
    """
    Creates a directory after checking it does not exists

    Parameters
    ----------
    path :  str or Path
        Location of the new directory
    
    Returns
    -------
    path: Path
        Path of the new directory

    """
    path=Path(path)
    if not path.exists() and not path.is_symlink():
        logging.debug(f"Creating {path}")
        Path.mkdir(path)
    return path

def get_pfamID(pdbID, chain):
    """
    Returns PFAM and Uniprot IDs

    Parameters
    ----------
    pdbID :  str
        pdbID (4 characters)
    chain : str
        Select chain from pdb
    Returns
    -------
    pfamID : str
        PFAM family ID
    """

    # TODO fix function
    df = pd.read_csv(f'{_path}/data/pdb_chain_pfam.csv', header=1)
    if sum((df['PDB'] == pdbID.lower()) & (df['CHAIN'] == chain.upper())) != 0:
        #Assumes one domain for the PDB
        pfamID = df.loc[(df['PDB'] == pdbID.lower()) & (df['CHAIN'] == chain.upper())]["PFAM_ID"].values[0]
    else:
        print('PFAM ID is unavailable')
        pfamID = 'null'
    return pfamID

