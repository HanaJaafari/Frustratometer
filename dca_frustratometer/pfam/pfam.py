import urllib.request
from pathlib import Path
import logging
import gzip
import os
from ..utils import create_directory


#Download whole database
def database(path,
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
def get(pfamID, database_path):
    raise NotImplementedError

# Download single alignment
def download_full_alignment(PFAM_ID,
                            alignment_files_directory=os.getcwd()):
    """'
    Retrieves a pfam family alignment from interpro

    Parameters
    ----------
    PFAM_ID : str,
        ID of PFAM family. ex: PF00001
    alignment_files_directory:  str
        If selected TRUE for download_all_alignment_files_status, 
        provide filepath. Default is current directory. 

    Returns
    -------
    output : Path
        location of alignment
    
    """
    from urllib.request import urlopen
    import gzip

    url = f'https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{PFAM_ID}/?annotation=alignment:full&download'
    logging.debug(f'Downloading {url} to {output_file}')

    output_file =Path(f"{alignment_files_directory}/{PFAM_ID}_full_MSA.sto")

    zipped_alignment = urllib.request.urlopen(url).read()
    unzipped_alignment = gzip.decompress(zipped_alignment)
    output_file.write_bytes(unzipped_alignment)

    return output_file

# TODO: Get a single file from database

