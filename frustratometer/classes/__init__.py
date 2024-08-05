from .DCA import DCA
from .AWSEM import AWSEM
from .Structure import Structure
from .Map import Map

"""Frustratometer Classes

The Frustratometer contains the AWSEM, DCA, Map, and Structure classes. The user should start off by creating 
an instance of the Structure class with their structure of choice. Then, the user can calculate frustration index values
and energy values associated with this structure using either the DCA or AWSEM classes. 

"""

__all__ = ['DCA', 'AWSEM', 'Structure', 'Map']
# __all__.extend(DCA.__all__)
# __all__.extend(AWSEM.__all__)
