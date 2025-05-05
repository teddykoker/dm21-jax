
# dm21-jax

A JAX implementation of the DM21 family of functionals introduced in [*Pushing
the Frontiers of Density Functionals by Solving the Fractional Electron
Problem*](https://www.science.org/doi/10.1126/science.abj6511), ported from the
original [PySCF
interface](https://github.com/google-deepmind/deepmind-research/tree/master/density_functional_approximation_dm21)
with serialized TensorFlow graphs. 

The model architecture, included in `dm21_jax/model.py`, could be easily ported
to PyTorch as well.

## Installation

```bash
pip install .
```

You can also set up a development environment using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Running `pytest` will run trial calculations for all functionals, and check that
the results are consistent with the original PySCF implementation.

## Usage

The library can be used just like the original PySCF interface, by patching the
`._numint` attribute of your DFT solver:

```python
from pyscf import gto
from pyscf import dft

import dm21_jax

# Create the molecule of interest and select the basis set.
mol = gto.Mole()
mol.atom = 'Ne 0.0 0.0 0.0'
mol.basis = 'cc-pVDZ'
mol.build()

# Create a DFT solver and insert the DM21 functional into the solver.
mf = dft.RKS(mol)
mf._numint = dm21_jax.NeuralNumInt(dm21_jax.Functional.DM21)

# Run the DFT calculation.
mf.kernel()
```

Please see the [original PySCF
interface](https://github.com/google-deepmind/deepmind-research/tree/master/density_functional_approximation_dm21)
for additional instruction on best practices.


## References

Original paper:

```bibtex
@article{kirkpatrick2021pushing,
  title={Pushing the frontiers of density functionals by solving the fractional electron problem},
  author={Kirkpatrick, James and McMorrow, Brendan and Turban, David HP and Gaunt, Alexander L and Spencer, James S and Matthews, Alexander GDG and Obika, Annette and Thiry, Louis and Fortunato, Meire and Pfau, David and others},
  journal={Science},
  volume={374},
  number={6573},
  pages={1385--1389},
  year={2021},
  publisher={American Association for the Advancement of Science}
}
```

Original interface: [deepmind-research/density_functional_approximation_dm21](https://github.com/google-deepmind/deepmind-research/tree/master/density_functional_approximation_dm21)


## License 

Note the original code is licensed under the [Apache 2.0 license](LICENSE),
therefore this code retains that license. The model weights contained in this
repository in `dm21_jax/checkpoints` are converted from the original TensorFlow
checkpoints, which are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).