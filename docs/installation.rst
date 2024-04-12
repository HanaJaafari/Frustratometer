Installation
===============

To install this modules please clone this repository and install it using the following commands.

.. code-block:: sh

    git clone HanaJaafari/Frustratometer
    cd Frustratometer
    conda install -c conda-forge --file requirements.txt
    pip install -e .

A clean python environment is recommended.

Optional Dependencies
---------------------

Installing pydca
~~~~~~~~~~~~~~~~~

`pydca` is an optional dependency that enables advanced bioinformatics functionalities. To install `pydca`, follow these steps:

1. **Clone the pydca Repository**
   Use Git to clone the repository to your local machine:
   
   .. code-block:: sh

      git clone https://github.com/cabb99/pydca.git

2. **Enter the Repository Directory**
   Change into the directory of the cloned repository:
   
   .. code-block:: sh

      cd pydca

3. **Install with pip**
   You can install the package in editable mode using pip. This method is recommended as it makes the package easy to update:
   
   .. code-block:: sh

      pip install -e .

   Alternatively, if the repository provides a shell script for installation, you can run:
   
   .. code-block:: sh

      ./install.sh

Installing OpenMM and pdbfixer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`OpenMM` and `pdbfixer` are optional dependencies useful for fixing pdbs. They are usually installed from the `conda-forge` channel. To install these packages, run the following command:

.. code-block:: sh

    conda install -c conda-forge openmm pdbfixer


