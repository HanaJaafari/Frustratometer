import urllib.request
from pathlib import Path
import logging
import gzip
from ..utils import create_directory
import glob


#Download whole database
def download_database(path,
             name='PFAM_current',
             url="https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.full.uniprot.gz"):
    """
    Downloads and creates a pfam database in the Database folder

    Parameters
    ----------
    url :  str
        Adress of pfam database
    name: str
        Name of the new database folder

    Returns
    -------
    alignment_path: Path
        Path of the alignments
    """

    
    #Make sure that the path exists
    path=Path(path)
    assert path.exists(), "The path doesn't exist"
    assert path.is_dir(), "The path is not a directory" 

    # Create database directory
    
    data_path = create_directory(path / name)
    alignments_path = create_directory(data_path / 'Alignments')

    # Get file name
    file_name = url.split('/')[-1]

    # Download pfam alignments
    logging.debug(f"Downloading {url} to {data_path}/{file_name}")
    urllib.request.urlretrieve(f"{url}", f"{data_path}/{file_name}")

    # Split PFAM alignments
    with gzip.open(f'{data_path}/{file_name}') as in_file:
        new_lines = ''
        acc = 'Unknown'
        for line in in_file:
            line = line.decode('utf-8')
            if line.strip() == '//':
                if len(new_lines) > 0:
                    with open(f'{alignments_path}/{acc}.sto', 'w+') as out_file:
                        logging.debug(f'{alignments_path}/{acc}.sto')
                        out_file.write(new_lines)
                new_lines = ''
                acc = 'Unknown'
                continue
            new_lines += line
            l = line.strip().split()
            if len(l) == 3 and l[0] == "#=GF" and l[1] == "AC":
                acc = l[2]
        if len(new_lines) > 0:
            with open(f'{alignments_path}/{acc}.sto', 'w+') as out_file:
                logging.debug(f'{alignments_path}/{acc}.sto')
                out_file.write(new_lines)
    return alignments_path

# Get a single alignment
def get_alignment(pfamid, database_path):
    path = Path(database_path)
    assert path.exists(), "The path doesn't exist"
    assert path.is_dir(), "The path is not a directory" 
    files=glob.glob(str(path/f'{pfamid}')+'.*.sto')
    print(str(path/f'{pfamid}')+'.*.sto')
    if len(files)==0:
        raise(IOError,'File not found')
    if len(files)>1:
        raise(IOError, 'Multiple files found')
    return Path(files[0])

def alignment(pfamid,
              output_file,
              alignment_type='full'):
    """'
    Retrieves a pfam family alignment from interpro

    Parameters
    ----------
    pfamid : str,
        ID of PFAM family. ex: PF00001
    alignment_type: str,
        alignment type to retrieve. Options: full, seed, uniprot
    output_file: str
        location of the output file. Default: Temporary file

    Returns
    -------
    output : Path
        location of alignment
    
    """
    url = f'https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfamid}/?annotation=alignment:{alignment_type}'
    #url = f'https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{pfamid}/?annotation=alignment:{alignment_type}&download' 
    logging.debug(f'Downloading {url} to {output_file}')

    zipped_alignment = urllib.request.urlopen(url).read()
    alignment = gzip.decompress(zipped_alignment)
    output_file.write_bytes(alignment)
    return output_file
