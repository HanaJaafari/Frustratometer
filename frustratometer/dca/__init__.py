from . import matlab

try:
    from . import pydca
except ImportError as e:
    if 'pydca' in str(e):
        class InstallPydca:
            @staticmethod
            def plmdca(*args, **kwargs):
                InstallPydca.warn_pydca_not_installed()
                raise ImportError("pydca must be installed to use the plmdca() function.")

            @staticmethod
            def mfdca(*args, **kwargs):
                InstallPydca.warn_pydca_not_installed()
                raise ImportError("pydca must be installed to use the mfdca() function.")

            @staticmethod
            def warn_pydca_not_installed():
                import warnings
                warnings.warn(
                    "pydca is not installed but is needed for this function. Advanced functionalities will be unavailable.\n"
                    "To install pydca, please follow these steps:\n"
                    "1. Clone the repository: git clone https://github.com/cabb99/pydca.git\n"
                    "2. Enter the directory: cd pydca\n"
                    "3. Install using pip: pip install -e .\n"
                    "Alternatively, you can run: ./install.sh",
                    ImportWarning
                )

        pydca = InstallPydca()  # Create an instance of the InstallPydca class
    else:
        raise e

__all__ = ['matlab', 'pydca']