
Installation 
------------

MGL requires a series of pacages, our suggestion is to create a new conda environment using our yml file 
(**specify location**) with ``conda env create -f environment.yml``.

Alternatively, we listed here the packages needed to correctly run MGL. We suggest to install CosmoPower 
first as it will likely create conflicts with other packages if installed later on.
First, create conda environment with ``python=3.10``, then install:

- `CosmoPower <https://alessiospuriomancini.github.io/cosmopower/installation/>`_

- matplotlib pandas

- `nautilus-sampler <https://nautilus-sampler.readthedocs.io/en/latest/>`_ with conda-forge

- scipy

- getdist

- sacc

- astropy-base with conda-forge

- PyYAML

- fast-pt
 
- `BCemu <https://github.com/sambit-giri/BCemu>`_
 
- `BACCO-emulator <https://baccoemu.readthedocs.io/en/latest/>`_ .
 
- `CAMB <https://camb.readthedocs.io/en/latest/>`_
 
- multiprocess
 
- `MGrowth <https://github.com/MariaTsedrik/MGrowth>`_

Finally:

- clone `MGlensing <https://github.com/MariaTsedrik/MGlensing/tree/main>`_ from github
	

-----------------------------------------------------------------------------------------------------------------------

Check the list of installed packages in the file ``spec-file.txt``, you can

- create an environment with the same characteristics using ``conda create --name myenv --file spec-file.txt``,

- or install listed packages into an existing environment with ``conda install --name myenv --file spec-file.txt``.

