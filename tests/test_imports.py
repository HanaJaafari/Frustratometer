import pytest
from unittest.mock import Mock, patch
import sys

def test_frustratometer_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "frustratometer" in sys.modules

def mock_import(module_name):
    """Raise ImportError if trying to import a specific module."""
    original_import = __builtins__['__import__']

    def new_import(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    return new_import

def test_missing_openmm_dependency():
    with patch('builtins.__import__', side_effect=mock_import('openmm')):
        import frustratometer
        with pytest.raises(ImportError) as excinfo:
            frustratometer.pdb.repair_pdb('test.pdb', 'A')
        assert f"openmm and pdbfixer must be installed" in str(excinfo.value)

def test_missing_pdbfixer_dependency():
    with patch('builtins.__import__', side_effect=mock_import('pdbfixer')):
        import frustratometer
        with pytest.raises(ImportError) as excinfo:
            frustratometer.pdb.repair_pdb('test.pdb', 'A')
        assert f"openmm and pdbfixer must be installed" in str(excinfo.value)

def test_missing_pydca_dependency():
    with patch('builtins.__import__', side_effect=mock_import('pydca')):
        import frustratometer
        with pytest.raises(ImportError) as excinfo:
            frustratometer.dca.pydca.mfdca('test.fasta')
        assert f"pydca must be installed" in str(excinfo.value)
        with pytest.raises(ImportError) as excinfo:
            frustratometer.dca.pydca.plmdca('test.fasta')
        assert f"pydca must be installed" in str(excinfo.value)

