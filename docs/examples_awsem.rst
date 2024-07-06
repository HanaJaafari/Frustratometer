AWSEM Frustratometer examples
=============================

Loading Protein Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Frustratometer package includes a prody-based Structure class to load the structure and calculate the properties needed for the AWSEM and DCA Frustratometers.

.. code-block:: python

    import frustratometer

    # Define the path to your PDB file
    pdb_path = Path('data/my_protein.pdb')
    # Load the structure
    structure = frustratometer.Structure.full_pdb(pdb_path)
    structure.sequence #The sequence of the structure

Creating an AWSEM Model Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After loading the structure, create an AWSEM model instance with the desired parameters. Here we provide some typical configurations that can be found elsewhere.

.. code-block:: python

    ## Single residue frustration with electrostatics
    model_singleresidue = frustratometer.AWSEM(structure, min_sequence_separation_contact=2) 
    ## Single residue frustration without electrostatics
    model_singleresidue_noelectrostatics = frustratometer.AWSEM(structure, min_sequence_separation_contact=2, k_electrostatics=0) 
    ## Mutational/Configurational frustration with electrostatics
    model_mutational = frustratometer.AWSEM(structure) 
    ## Mutational/Configurational frustration without electrostatics
    model_mutational_noelectrostatics = frustratometer.AWSEM(structure, k_electrostatics=0)
    ## Mutational frustration with sequence separation of 12
    model_mutational_seqsep12 = frustratometer.AWSEM(structure, min_sequence_separation_rho=13)
    ## Typical openAWSEM
    model_openAWSEM = frustratometer.AWSEM(structure, min_sequence_separation_contact = 10, distance_cutoff_contact = None)

Calculating Residue Densities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To calculate the density of residues in the structure.

.. code-block:: python

    calculated_densities = model_singleresidue.rho_r
    print(calculated_densities)

Calculating Frustration Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Frustration indices can be calculated for single residues or mutationally. This measurement helps identify energetically favorable or unfavorable interactions within the protein structure.

Single Residue Frustration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate single residue frustration
    single_residue_frustration = model_singleresidue.frustration(kind='singleresidue')
    print(single_residue_frustration)

Single Residue Decoy Fluctuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The frustratometer package also allows the quick calculation of the energies of all single residue and mutational decoys.

.. code-block:: python

    # Calculate single residue decoy energy fluctuations
    decoy_fluctuation = model_singleresidue.decoy_fluctuation(kind='singleresidue')
    print(decoy_fluctuation)

Mutational Frustration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate mutational frustration
    mutational_frustration = model_mutational.frustration(kind='mutational')
    print(mutational_frustration)

Energy Calculations
~~~~~~~~~~~~~~~~~~~

You can calculate different energy contributions, including fields energy (pseudo one-body terms like burial), couplings energy (pseudo two-body terms like contact and electrostatics), and their combination to determine the native energy of the protein structure. The fields, couplings, and native energies of other threaded sequences can be calculated like below by simply changing the "sequence" variable in the functions' arguments.

Fields Energy
~~~~~~~~~~~~~

.. code-block:: python

    fields_energy = model_openAWSEM.fields_energy()
    print(fields_energy)

Couplings Energy
~~~~~~~~~~~~~~~~

.. code-block:: python

    couplings_energy = model_openAWSEM.couplings_energy()
    print(couplings_energy)

Native Energy
~~~~~~~~~~~~~

Native energy can be considered as a combination of fields and couplings energy contributions.

.. code-block:: python

    native_energy = model_openAWSEM.native_energy()
    print(native_energy)
