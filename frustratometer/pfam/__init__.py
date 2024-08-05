"""pfam: PFAM functions

This module includes functions to download and manipulate PFAM alignments.

There are two options to obtain alignments from PFAM: downloading the database
locally or downloading specific alignments.

If you want to download the database locally, use the "download_database" function to
download or update the database. Then, use the "get_alignment" function to select a single
alignment from the downloaded database.

If you want to retrieve a single alignment from the PFAM database,
use the "download_alignment" function

Examples:
    
        $ frustratometer.pfam.database('/media/databases',
        https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.full.uniprot.gz)

Todo:
    * Create a function to get single alignments

"""
from .pfam import download_database, get_alignment, download_aligment

__all__ = ['download_database', 'get_alignment', 'download_aligment']