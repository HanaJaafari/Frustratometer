
import urllib.request
from pathlib import Path
import logging
import gzip
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

#Download single alignment
def alignment(pfamID,
              output_file=None,
              alignment_type='uniprot'):
    """'
    Retrieves a pfam family alignment from interpro

    Parameters
    ----------
    pfamID : str,
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
    import tempfile
    from urllib.request import urlopen
    import gzip

    url = f'https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfamID}/?annotation=alignment:{alignment_type}'
    logging.debug(f'Downloading {url} to {output_file}')

    if output_file is None:
        output = tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_interpro.sto')
        output_file = Path(output.name)
    else:
        output_file = Path(output_file)

    output = urlopen(url).read()
    alignment = gzip.decompress(output)
    output_file.write_bytes(alignment)
    return output_file

# TODO: Get a single file from database

