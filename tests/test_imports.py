import pytest
from unittest.mock import Mock, patch
import sys

@pytest.fixture(autouse=True)
def clean_frustratometer():
    # List all modules that might be part of frustratometer package
    to_remove = [mod for mod in sys.modules if mod.startswith('frustratometer')]
    for mod in to_remove:
        del sys.modules[mod]
    assert "frustratometer" not in sys.modules

def test_frustratometer_imported(clean_frustratometer):
    import frustratometer
    assert "frustratometer" in sys.modules

def test_frustratometer_imported(clean_frustratometer):
    """Sample test, will always pass so long as import statement worked."""
    import frustratometer
    assert "frustratometer" in sys.modules

def mock_import(*module_names):
    """Raise ImportError if trying to import specific modules."""
    original_import = __builtins__['__import__']
    
    def new_import(name, *args, **kwargs):
        if name in module_names:
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)
    
    return new_import

def test_missing_pdbfixer_dependency(clean_frustratometer):
    with patch('builtins.__import__', side_effect=mock_import('pdbfixer')):
        import frustratometer
        with pytest.raises(ImportError) as excinfo:
            frustratometer.pdb.repair_pdb('test.pdb', 'A')
        assert f"openmm and pdbfixer must be installed" in str(excinfo.value)

def test_missing_pydca_dependency(clean_frustratometer):
    with patch('builtins.__import__', side_effect=mock_import('pydca')):
        import frustratometer
        with pytest.raises(ImportError) as excinfo:
            frustratometer.dca.pydca.mfdca('test.fasta')
        assert f"pydca must be installed" in str(excinfo.value)
        with pytest.raises(ImportError) as excinfo:
            frustratometer.dca.pydca.plmdca('test.fasta')
        assert f"pydca must be installed" in str(excinfo.value)

