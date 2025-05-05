# Copyright 2021 DeepMind Technologies Limited.
# Copyright 2025 Teddy Koker. Modifications:
# - removed tensorflow dependency
# - removed export test.
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
"""Tests for neural_numint."""

from absl.testing import absltest
from absl.testing import parameterized
from pyscf import dft
from pyscf import gto
from pyscf import lib

from dm21_jax import neural_numint


class NeuralNumintTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        lib.param.TMPDIR = None
        lib.num_threads(1)

    # Golden values were obtained using the version of PySCF (including integral
    # generation) reported in the DM21 paper.
    @parameterized.parameters(
        {"functional": neural_numint.Functional.DM21, "expected_energy": -126.898521},
        {"functional": neural_numint.Functional.DM21m, "expected_energy": -126.907332},
        {"functional": neural_numint.Functional.DM21mc, "expected_energy": -126.922127},
        {"functional": neural_numint.Functional.DM21mu, "expected_energy": -126.898178},
    )
    def test_rks(self, functional, expected_energy):
        ni = neural_numint.NeuralNumInt(functional)

        mol = gto.Mole()
        mol.atom = [["Ne", 0.0, 0.0, 0.0]]
        mol.basis = "sto-3g"
        mol.build()

        mf = dft.RKS(mol)
        mf.small_rho_cutoff = 1.0e-20
        mf._numint = ni
        mf.run()
        self.assertAlmostEqual(mf.e_tot, expected_energy, delta=2.0e-4)

    @parameterized.parameters(
        {"functional": neural_numint.Functional.DM21, "expected_energy": -37.34184876},
        {"functional": neural_numint.Functional.DM21m, "expected_energy": -37.3377766},
        {
            "functional": neural_numint.Functional.DM21mc,
            "expected_energy": -37.33489173,
        },
        {
            "functional": neural_numint.Functional.DM21mu,
            "expected_energy": -37.34015315,
        },
    )
    def test_uks(self, functional, expected_energy):
        ni = neural_numint.NeuralNumInt(functional)

        mol = gto.Mole()
        mol.atom = [["C", 0.0, 0.0, 0.0]]
        mol.spin = 2
        mol.basis = "sto-3g"
        mol.build()

        mf = dft.UKS(mol)
        mf.small_rho_cutoff = 1.0e-20
        mf._numint = ni
        mf.run()
        self.assertAlmostEqual(mf.e_tot, expected_energy, delta=2.0e-4)


if __name__ == "__main__":
    absltest.main()
