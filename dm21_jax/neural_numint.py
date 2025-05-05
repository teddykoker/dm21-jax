# Copyright 2021 DeepMind Technologies Limited.
# Copyright 2025 Teddy Koker. Modifications:
# - removed tensorflow model and model loading
# - added support for JAX model
# - added eval_xc_eff() function, which allows the the package to be used with
#   newer versions of PySCF
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An interface to DM21 family of exchange-correlation functionals for PySCF."""

import enum
import os
from typing import Generator, Optional, Sequence, Tuple, Union

import attr
import chex
import equinox as eqx
import jax
import numpy as np
from pyscf import dft, gto
from pyscf.dft import numint, xc_deriv

from dm21_jax import compute_hfx_density
from dm21_jax.model import NNFunctional

# NOTE: from original code
# TODO(b/196260242): avoid depending upon private function
_dot_ao_ao = numint._dot_ao_ao  # pylint: disable=protected-access


@enum.unique
class Functional(enum.Enum):
    """Enum for exchange-correlation functionals in the DM21 family.

    Attributes:
      DM21: trained on molecules dataset, and fractional charge, and fractional
        spin constraints.
      DM21m: trained on molecules dataset.
      DM21mc: trained on molecules dataset, and fractional charge constraints.
      DM21mu: trained on molecules dataset, and electron gas constraints.
    """

    # Break pylint's preferred naming pattern to match the functional names used
    # in the paper.
    # pylint: disable=invalid-name
    DM21 = enum.auto()
    DM21m = enum.auto()
    DM21mc = enum.auto()
    DM21mu = enum.auto()
    # pylint: enable=invalid-name


# We use attr.s instead of here instead of dataclasses.dataclass as
# dataclasses.asdict returns a deepcopy of the attributes. This is wasteful in
# memory if they are large and breaks (as in the case of tf.Tensors) if they are
# not serializable. attr.asdict does not perform this copy and so works with
# both np.ndarrays and tf.Tensors.
# NOTE: we use chex.dataclass instead of attr.s as we want to use JAX arrays
@chex.dataclass
class FunctionalInputs:
    r""" "Inputs required for DM21 functionals.

    Depending upon the context, this is either a set of numpy arrays (feature
    construction) or JAX arrays (constructing placeholders/running functionals).

    Attributes:
      rho_a: Density information for the alpha electrons.
        PySCF for meta-GGAs supplies a single array for the total density
        (restricted calculations) and a pair of arrays, one for each spin channel
        (unrestricted calculations).
        Each array/tensor is of shape (6, N) and contains the density and density
        derivatives, where:
         rho(0, :) - density at each grid point
         rho(1, :) - norm of the derivative of the density at each grid point
                     along x
         rho(2, :) - norm of the derivative of the density at each grid point
                     along y
         rho(3, :) - norm of the derivative of the density at each grid point
                     along z
         rho(4, :) - \nabla^2 \rho [not used]
         rho(5, :) - tau (1/2 (\nabla \rho)^2) at each grid point.
        See pyscf.dft.numint.eval_rho for more details.
        We require separate inputs for both alpha- and beta-spin densities, even
        in restricted calculations (where rho_a = rho_b = rho/2, where rho is the
        total density).
      rho_b: as for rho_a for the beta electrons.
      hfx_a: local Hartree-Fock energy density at each grid point for the alpha-
        spin density for each value of omega.  Shape [N, len(omega_values)].
        See compute_hfx_density for more details.
      hfx_b: as for hfx_a for the beta-spin density.
      grid_coords: grid coordinates at which to evaluate the density. Shape
        (N, 3), where N is the number of grid points. Note that this is currently
        unused by the functional, but is still a required input.
      grid_weights: weight of each grid point. Shape (N).
    """

    rho_a: Union[np.ndarray, jax.Array]
    rho_b: Union[np.ndarray, jax.Array]
    hfx_a: Union[np.ndarray, jax.Array]
    hfx_b: Union[np.ndarray, jax.Array]
    grid_coords: Union[np.ndarray, jax.Array]
    grid_weights: Union[np.ndarray, jax.Array]


@attr.s(auto_attribs=True)
class _GridState:
    """Internal state required for the numerical grid.

    Attributes:
      coords: coordinates of the grid. Shape (N, 3), where N is the number of grid
          points.
      weight: weight associated with each grid point. Shape (N).
      mask: mask indicating whether a shell is zero at a grid point. Shape
          (N, nbas) where nbas is the number of shells in the basis set. See
          pyscf.dft.gen_grids.make_mask.
      ao: atomic orbitals evaluated on the grid. Shape (N, nao), where nao is the
          number of atomic orbitals, or shape (:, N, nao), where the 0-th element
          contains the ao values, the next three elements contain the first
          derivatives, and so on.
    """

    coords: np.ndarray
    weight: np.ndarray
    mask: np.ndarray
    ao: np.ndarray


@attr.s(auto_attribs=True)
class _SystemState:
    """Internal state required for system of interest.

    Attributes:
        mol: PySCF molecule
        dms: density matrix or matrices (unrestricted calculations only).
            Restricted calculations: shape (nao, nao), where nao is the number of
            atomic orbitals.
            Unrestricted calculations: shape (2, nao, nao) or a sequence (length 2) of
            arrays of shape (nao, nao), and dms[0] and dms[1] are the density matrices
            of the alpha and beta electrons respectively.
    """

    mol: gto.Mole
    dms: Union[np.ndarray, Sequence[np.ndarray]]


def _get_number_of_density_matrices(dms):
    """Returns the number of density matrices in dms."""
    # See pyscf.numint.NumInt._gen_rho_evaluator
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        return 1
    return len(dms)


class NeuralNumInt(numint.NumInt):
    """A wrapper around pyscf.dft.numint.NumInt for the DM21 functionals.

    In order to supply the local Hartree-Fock features required for the DM21
    functionals, we lightly wrap the NumInt class. The actual evaluation of the
    exchange-correlation functional is performed in NeuralNumInt.eval_xc.

    Usage:
        mf = dft.RKS(...)  # dft.ROKS and dft.UKS are also supported.
        # Specify the functional by monkey-patching mf._numint rather than using
        # mf._xc or mf._define_xc_.
        mf._numint = NeuralNumInt(Functional.DM21)
        mf.kernel()
    """

    def __init__(
        self, functional: Functional, *, checkpoint_path: Optional[str] = None
    ):
        """Constructs a NeuralNumInt object.

        Args:
            functional: member of Functional enum giving the name of the
                functional.
            checkpoint_path: Optional path to specify the directory containing the
                checkpoints of the DM21 family of functionals. If not specified, attempt
                to find the checkpoints using a path relative to the source code.
        """

        self._functional_name = functional.name
        if checkpoint_path:
            self._model_path = os.path.join(
                checkpoint_path, self._functional_name + ".eqx"
            )
        else:
            self._model_path = os.path.join(
                os.path.dirname(__file__), "checkpoints", self._functional_name + ".eqx"
            )

        # All DM21 functionals use local Hartree-Fock features with a non-range
        # separated 1/r kernel and a range-seperated kernel with \omega = 0.4.
        # Note an omega of 0.0 is interpreted by PySCF and libcint to indicate no
        # range-separation.
        self._omega_values = [0.0, 0.4]

        model = NNFunctional(key=jax.random.key(0))
        model = eqx.tree_deserialise_leaves(self._model_path, model)
        self._model = eqx.filter_jit(model)

        self._grid_state = None
        self._system_state = None
        self._vmat_hf = None
        super().__init__()

    # DM21* functionals include the hybrid term directly, so set the
    # range-separated and hybrid parameters expected by PySCF to 0 so PySCF
    # doesn't also add these contributions in separately.
    def rsh_coeff(self, *args):
        """Returns the range separated parameters, omega, alpha, beta."""
        return [0.0, 0.0, 0.0]

    def hybrid_coeff(self, *args, **kwargs):
        """Returns the fraction of Hartree-Fock exchange to include."""
        return 0.0

    def _xc_type(self, *args, **kwargs):
        return "MGGA"

    def nr_rks(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        xc_code: str,
        dms: Union[np.ndarray, Sequence[np.ndarray]],
        relativity: int = 0,
        hermi: int = 0,
        max_memory: float = 20000,
        verbose=None,
    ) -> Tuple[float, float, np.ndarray]:
        """Calculates RKS XC functional and potential matrix on a given grid.

        Args:
            mol: PySCF molecule.
            grids: grid on which to evaluate the functional.
            xc_code: XC code. Unused. NeuralNumInt hard codes the XC functional
                based upon the functional argument given to the constructor.
            dms: the density matrix or sequence of density matrices. Multiple density
                matrices are not currently supported. Shape (nao, nao), where nao is the
                number of atomic orbitals.
            relativity: Unused. (pyscf.numint.NumInt.nr_rks does not currently use
                this argument.)
            hermi: 0 if the density matrix is Hermitian, 1 if the density matrix is
                non-Hermitian.
            max_memory: the maximum cache to use, in MB.
            verbose: verbosity level. Unused. (PySCF currently does not handle the
                verbosity level passed in here.)

        Returns:
            nelec, excsum, vmat, where
                nelec is the number of electrons obtained by numerical integration of
                the density matrix.
                excsum is the functional's XC energy.
                vmat is the functional's XC potential matrix, shape (nao, nao).

        Raises:
            NotImplementedError: if multiple density matrices are supplied.
        """
        # Wrap nr_rks so we can store internal variables required to evaluate the
        # contribution to the XC potential from local Hartree-Fock features.
        # See pyscf.dft.numint.nr_rks for more details.
        ndms = _get_number_of_density_matrices(dms)
        if ndms > 1:
            raise NotImplementedError(
                "NeuralNumInt does not support multiple density matrices. "
                "Only ground state DFT calculations are currently implemented."
            )
        nao = mol.nao_nr()
        self._vmat_hf = np.zeros((nao, nao))
        self._system_state = _SystemState(mol=mol, dms=dms)
        nelec, excsum, vmat = super().nr_rks(
            mol=mol,
            grids=grids,
            xc_code=xc_code,
            dms=dms,
            relativity=relativity,
            hermi=hermi,
            max_memory=max_memory,
            verbose=verbose,
        )
        vmat += self._vmat_hf + self._vmat_hf.T

        # Clear internal state to prevent accidental re-use.
        self._system_state = None
        self._grid_state = None
        return nelec, excsum, vmat

    def nr_uks(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        xc_code: str,
        dms: Union[Sequence[np.ndarray], Sequence[Sequence[np.ndarray]]],
        relativity: int = 0,
        hermi: int = 0,
        max_memory: float = 20000,
        verbose=None,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Calculates UKS XC functional and potential matrix on a given grid.

        Args:
            mol: PySCF molecule.
            grids: grid on which to evaluate the functional.
            xc_code: XC code. Unused. NeuralNumInt hard codes the XC functional
                based upon the functional argument given to the constructor.
            dms: the density matrix or sequence of density matrices for each spin
                channel. Multiple density matrices for each spin channel are not
                currently supported. Each density matrix is shape (nao, nao), where nao
                is the number of atomic orbitals.
            relativity: Unused. (pyscf.dft.numint.NumInt.nr_rks does not currently use
                this argument.)
            hermi: 0 if the density matrix is Hermitian, 1 if the density matrix is
                non-Hermitian.
            max_memory: the maximum cache to use, in MB.
            verbose: verbosity level. Unused. (PySCF currently does not handle the
                verbosity level passed in here.)

        Returns:
            nelec, excsum, vmat, where
                nelec is the number of alpha, beta electrons obtained by numerical
                integration of the density matrix as an array of size 2.
                excsum is the functional's XC energy.
                vmat is the functional's XC potential matrix, shape (2, nao, nao), where
                vmat[0] and vmat[1] are the potential matrices for the alpha and beta
                spin channels respectively.

        Raises:
            NotImplementedError: if multiple density matrices for each spin channel
                are supplied.
        """
        # Wrap nr_uks so we can store internal variables required to evaluate the
        # contribution to the XC potential from local Hartree-Fock features.
        # See pyscf.dft.numint.nr_uks for more details.
        if isinstance(dms, np.ndarray) and dms.ndim == 2:  # RHF DM
            ndms = _get_number_of_density_matrices(dms)
        else:
            ndms = _get_number_of_density_matrices(dms[0])
        if ndms > 1:
            raise NotImplementedError(
                "NeuralNumInt does not support multiple density matrices. "
                "Only ground state DFT calculations are currently implemented."
            )

        nao = mol.nao_nr()
        self._vmat_hf = np.zeros((2, nao, nao))
        self._system_state = _SystemState(mol=mol, dms=dms)
        nelec, excsum, vmat = super().nr_uks(
            mol=mol,
            grids=grids,
            xc_code=xc_code,
            dms=dms,
            relativity=relativity,
            hermi=hermi,
            max_memory=max_memory,
            verbose=verbose,
        )
        vmat[0] += self._vmat_hf[0] + self._vmat_hf[0].T
        vmat[1] += self._vmat_hf[1] + self._vmat_hf[1].T

        # Clear internal state to prevent accidental re-use.
        self._system_state = None
        self._grid_state = None
        self._vmat_hf = None
        return nelec, excsum, vmat

    def block_loop(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        nao: Optional[int] = None,
        deriv: int = 0,
        max_memory: float = 2000,
        non0tab: Optional[np.ndarray] = None,
        blksize: Optional[int] = None,
        buf: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Loops over the grid by blocks. See pyscf.dft.numint.NumInt.block_loop.

        Args:
            mol: PySCF molecule.
            grids: grid on which to evaluate the functional.
            nao: number of basis functions. If None, obtained from mol.
            deriv: unused. The first functional derivatives are always computed.
            max_memory: the maximum cache to use for the information on the grid, in
                MB. Determines the size of each block if blksize is None.
            non0tab: mask determining if a shell in the basis set is zero at a grid
                point. Shape (N, nbas), where N is the number of grid points and nbas
                the number of shells in the basis set. Obtained from grids if not
                supplied.
            blksize: size of each block. Calculated from max_memory if None.
            buf: buffer to use for storing ao. If None, a new array for ao is created
                for each block.

        Yields:
            ao, mask, weight, coords: information on a block of the grid containing N'
            points, where
                ao: atomic orbitals evaluated on the grid. Shape (N', nao), where nao is
                the number of atomic orbitals.
                mask: mask indicating whether a shell in the basis set is zero at a grid
                point. Shape (N', nbas).
                weight: weight associated with each grid point. Shape (N').
                coords: coordinates of the grid. Shape (N', 3).
        """
        # Wrap block_loop so we can store internal variables required to evaluate
        # the contribution to the XC potential from local Hartree-Fock features.
        for ao, mask, weight, coords in super().block_loop(
            mol=mol,
            grids=grids,
            nao=nao,
            deriv=deriv,
            max_memory=max_memory,
            non0tab=non0tab,
            blksize=blksize,
            buf=buf,
        ):
            # Cache the curent block so we can access it in eval_xc.
            self._grid_state = _GridState(
                ao=ao, mask=mask, weight=weight, coords=coords
            )
            yield ao, mask, weight, coords

    def construct_functional_inputs(
        self,
        mol: gto.Mole,
        dms: Union[np.ndarray, Sequence[np.ndarray]],
        spin: int,
        coords: np.ndarray,
        weights: np.ndarray,
        rho: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        ao: Optional[np.ndarray] = None,
    ) -> Tuple[FunctionalInputs, Tuple[np.ndarray, np.ndarray]]:
        """Constructs the input features required for the functional.

        Args:
            mol: PySCF molecule.
            dms: density matrix of shape (nao, nao) (restricted calculations) or of
                shape (2, nao, nao) (unrestricted calculations) or tuple of density
                matrices for each spin channel, each of shape (nao, nao) (unrestricted
                calculations).
            spin: 0 for a spin-unpolarized (restricted Kohn-Sham) calculation, and
                spin-polarized (unrestricted) otherwise.
            coords: coordinates of the grid. Shape (N, 3), where N is the number of
                grid points.
            weights: weight associated with each grid point. Shape (N).
            rho: density and density derivatives at each grid point. Single array
                containing the total density for restricted calculations, tuple of
                arrays for each spin channel for unrestricted calculations. Each array
                has shape (6, N). See pyscf.dft.numint.eval_rho and comments in
                FunctionalInputs for more details.
            ao: The atomic orbitals evaluated on the grid, shape (N, nao). Computed if
                not supplied.

        Returns:
            inputs, fxx, where
                inputs: FunctionalInputs object containing the inputs (as np.ndarrays)
                for the functional.
                fxx: intermediates, shape (N, nao) for the alpha- and beta-spin
                channels, required for computing the first derivative of the local
                Hartree-Fock density with respect to the density matrices. See
                compute_hfx_density for more details.
        """
        if spin == 0:
            # RKS
            rhoa = rho / 2
            rhob = rho / 2
        else:
            # UKS
            rhoa, rhob = rho

        # Local HF features.
        exxa, exxb = [], []
        fxxa, fxxb = [], []
        for omega in sorted(self._omega_values):
            hfx_results = compute_hfx_density.get_hf_density(
                mol, dms, coords=coords, omega=omega, deriv=1, ao=ao
            )
            exxa.append(hfx_results.exx[0])
            exxb.append(hfx_results.exx[1])
            fxxa.append(hfx_results.fxx[0])
            fxxb.append(hfx_results.fxx[1])
        exxa = np.stack(exxa, axis=-1)
        fxxa = np.stack(fxxa, axis=-1)
        if spin == 0:
            exx = (exxa, exxa)
            fxx = (fxxa, fxxa)
        else:
            exxb = np.stack(exxb, axis=-1)
            fxxb = np.stack(fxxb, axis=-1)
            exx = (exxa, exxb)
            fxx = (fxxa, fxxb)

        return FunctionalInputs(
            rho_a=rhoa,
            rho_b=rhob,
            hfx_a=exx[0],
            hfx_b=exx[1],
            grid_coords=coords,
            grid_weights=weights,
        ), fxx

    def eval_xc_eff(self, xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None):
        r"""Returns the derivative tensor against the density parameters
        [density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a]

        or spin-polarized density parameters

        [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
          [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].

        Copied from PySCF 2.4.0, which had a working version (without calling libxc)
        https://github.com/pyscf/pyscf/blob/1007524cd4c61c15a0767b70004d93fcf3bc25a5/pyscf/dft/numint.py#L2656

        It differs from the eval_xc method in the derivatives of non-local part.
        The eval_xc method returns the XC functional derivatives to sigma
        (|\nabla \rho|^2)

        Args:
            rho: 2-dimensional or 3-dimensional array
                Total density or (spin-up, spin-down) densities (and their
                derivatives if GGA or MGGA functionals) on grids

        Kwargs:
            deriv: int
                derivative orders
            omega: float
                define the exponent in the attenuated Coulomb for RSH functional
        """
        del xc_code, verbose  # unused

        if omega is not None:
            raise NotImplementedError(
                "User-specifed range seperation parameters are "
                "not implemented for DM21 functionals."
            )
        rhop = np.asarray(rho)

        spin_polarized = rhop.ndim == 3

        if spin_polarized:
            assert rhop.shape[0] == 2
            spin = 1
            if rhop.shape[1] == 5:
                ngrids = rhop.shape[2]
                rhop = np.empty((2, 6, ngrids))
                rhop[0, :4] = rho[0][:4]
                rhop[1, :4] = rho[1][:4]
                rhop[:, 4] = 0
                rhop[0, 5] = rho[0][4]
                rhop[1, 5] = rho[1][4]
        else:
            spin = 0
            if rhop.shape[0] == 5:
                ngrids = rho.shape[1]
                rhop = np.empty((6, ngrids))
                rhop[:4] = rho[:4]
                rhop[4] = 0
                rhop[5] = rho[4]

        exc, vxc, fxc, kxc = self.eval_xc(None, rhop, spin, 0, deriv, omega, None)
        if deriv > 2:
            kxc = xc_deriv.transform_kxc(rhop, fxc, kxc, xctype, spin)
        if deriv > 1:
            fxc = xc_deriv.transform_fxc(rhop, vxc, fxc, xctype, spin)
        if deriv > 0:
            vxc = xc_deriv.transform_vxc(rhop, vxc, xctype, spin)
        return exc, vxc, fxc, kxc

    def eval_xc(
        self,
        xc_code: str,
        rho: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        spin: int = 0,
        relativity: int = 0,
        deriv: int = 1,
        omega: Optional[float] = None,
        verbose=None,
    ) -> Tuple[
        np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None
    ]:
        """Evaluates the XC energy and functional derivatives.

        See pyscf.dft.libxc.eval_xc for more details on the interface.

        Note: this also sets self._vmat_extra, which contains the contribution the
        the potential matrix from the local Hartree-Fock terms in the functional.

        Args:
            xc_code: unused.
            rho: density and density derivatives at each grid point. Single array
                containing the total density for restricted calculations, tuple of
                arrays for each spin channel for unrestricted calculations. Each array
                has shape (6, N), where N is the number of grid points. See
                pyscf.dft.numint.eval_rho and comments in FunctionalInputs for more
                details.
            spin: 0 for a spin-unpolarized (restricted Kohn-Sham) calculation, and
                spin-polarized (unrestricted) otherwise.
            relativity: Not supported.
            deriv: unused. The first functional derivatives are always computed.
            omega: RSH parameter. Not supported.
            verbose: unused.

        Returns:
            exc, vxc, fxc, kxc, where:
                exc is the exchange-correlation potential matrix evaluated at each grid
                point, shape (N).
                vxc is (vrho, vgamma, vlapl, vtau), the first-order functional
                derivatives evaluated at each grid point, each shape (N).
                fxc is set to None. (The second-order functional derivatives are not
                computed.)
                kxc is set to None. (The third-order functional derivatives are not
                computed.)
        """
        del xc_code, verbose, deriv  # unused

        if relativity != 0:
            raise NotImplementedError(
                "Relatistic calculations are not implemented for DM21 functionals."
            )
        if omega is not None:
            raise NotImplementedError(
                "User-specifed range seperation parameters are "
                "not implemented for DM21 functionals."
            )

        # Retrieve cached state.
        ao = self._grid_state.ao
        if ao.ndim == 3:
            # Just need the AO values, not the gradients.
            ao = ao[0]
        if self._grid_state.weight is None:
            weights = np.array([1.0])
        else:
            weights = self._grid_state.weight
        mask = self._grid_state.mask

        inputs, (fxxa, fxxb) = self.construct_functional_inputs(
            mol=self._system_state.mol,
            dms=self._system_state.dms,
            spin=spin,
            rho=rho,
            weights=weights,
            coords=self._grid_state.coords,
            ao=ao,
        )

        exc, vrho, vsigma, vtau, vhf = self._model(inputs)

        mol = self._system_state.mol
        shls_slice = (0, mol.nbas)
        ao_loc_nr = mol.ao_loc_nr()
        if spin == 0:
            vxc_0 = (vrho[0] + vrho[1]) / 2.0
            # pyscf expects derivatives with respect to:
            # grad_rho . grad_rho.
            # The functional uses the first and last as inputs, but then has
            # grad_(rho_a + rho_b) . grad_(rho_a + rho_b)
            # as input. The following computes the correct total derivatives.
            vxc_1 = vsigma[0] / 4.0 + vsigma[1] / 4.0 + vsigma[2]
            vxc_3 = (vtau[0] + vtau[1]) / 2.0
            vxc_2 = np.zeros_like(vxc_3)
            vhfs = (vhf[0] + vhf[1]) / 2.0
            # Local Hartree-Fock terms
            for i in range(len(self._omega_values)):
                # Factor of 1/2 is to account for adding vmat_hf + vmat_hf.T to vmat,
                # which we do to match existing PySCF style. Unlike other terms, vmat_hf
                # is already symmetric though.
                aow = np.einsum("pi,p->pi", fxxa[:, :, i], -0.5 * vhfs[:, i])
                self._vmat_hf += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc_nr)
        else:
            vxc_0 = np.stack([vrho[0], vrho[1]], axis=1)
            # pyscf expects derivatives with respect to:
            # grad_rho_a . grad_rho_a
            # grad_rho_a . grad_rho_b
            # grad_rho_b . grad_rho_b
            # The functional uses the first and last as inputs, but then has
            # grad_(rho_a + rho_b) . grad_(rho_a + rho_b)
            # as input. The following computes the correct total derivatives.
            vxc_1 = np.stack(
                [
                    vsigma[0] + vsigma[2],
                    2.0 * vsigma[2],
                    vsigma[1] + vsigma[2],
                ],
                axis=1,
            )
            vxc_3 = np.stack([vtau[0], vtau[1]], axis=1)
            vxc_2 = np.zeros_like(vxc_3)
            vhfs = np.stack([vhf[0], vhf[1]], axis=2)
            for i in range(len(self._omega_values)):
                # Factors of 1/2 are due to the same reason as in the spin=0 case.
                aow = np.einsum("pi,p->pi", fxxa[:, :, i], -0.5 * vhfs[:, i, 0])
                self._vmat_hf[0] += _dot_ao_ao(
                    mol, ao, aow, mask, shls_slice, ao_loc_nr
                )
                aow = np.einsum("pi,p->pi", fxxb[:, :, i], -0.5 * vhfs[:, i, 1])
                self._vmat_hf[1] += _dot_ao_ao(
                    mol, ao, aow, mask, shls_slice, ao_loc_nr
                )

        fxc = None  # Second derivative not implemented
        kxc = None  # Second derivative not implemented
        # PySCF C routines expect float64.
        exc = np.array(exc).astype(np.float64)
        vxc = tuple(
            np.array(v).astype(np.float64) for v in (vxc_0, vxc_1, vxc_2, vxc_3)
        )
        return exc, vxc, fxc, kxc
