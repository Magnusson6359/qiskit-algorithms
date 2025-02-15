# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for an Imaginary McLachlan's Variational Principle."""
from __future__ import annotations

import warnings

from collections.abc import Sequence

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .imaginary_variational_principle import ImaginaryVariationalPrinciple

from ....exceptions import AlgorithmError
from ....gradients import (
    BaseEstimatorGradient,
    BaseQGT,
    DerivativeType,
    LinCombQGT,
    LinCombEstimatorGradient,
)


class ImaginaryMcLachlanPrinciple(ImaginaryVariationalPrinciple):
    """Class for an Imaginary McLachlan's Variational Principle. It aims to minimize the distance
    between both sides of the Wick-rotated Schrödinger equation with a quantum state given as a
    parametrized trial state. The principle leads to a system of linear equations handled by a
    linear solver. The imaginary variant means that we consider imaginary time dynamics.
    """

    def __init__(
        self,
        qgt: BaseQGT | None = None,
        gradient: BaseEstimatorGradient | None = None,
        non_hermitian: bool = False,
    ) -> None:
        """
        Args:
            qgt: Instance of a the GQT class used to compute the QFI.
                If ``None`` provided, ``LinCombQGT`` is used.
            gradient: Instance of a class used to compute the state gradient.
                If ``None`` provided, ``LinCombEstimatorGradient`` is used.
            non_hermitian: If True, the Hamiltonian is split into Hermitian and anti-Hermitian parts
                and the gradient is computed for each part separately. The final gradient is the sum
                of the gradients of the Hermitian and anti-Hermitian parts. This computes the gradient
                even if the Hamiltonian is non-Hermitian, but requires more evaluations. Default is False.
                As in https://doi.org/10.48550/arXiv.2201.03049


        Raises:
            AlgorithmError: If the gradient instance does not contain an estimator.
        """

        self._validate_grad_settings(gradient)

        if gradient is not None:
            try:
                estimator = gradient._estimator
            except Exception as exc:
                raise AlgorithmError(
                    "The provided gradient instance does not contain an estimator primitive."
                ) from exc
        else:
            estimator = Estimator()
            gradient = LinCombEstimatorGradient(estimator)

        if qgt is None:
            qgt = LinCombQGT(estimator)

        super().__init__(qgt, gradient)
        self.non_hermitian = non_hermitian

    def evolution_gradient(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: Sequence[float],
        gradient_params: Sequence[Parameter] | None = None,
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of parameters to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
                If ``None`` given, gradients w.r.t. all parameters will be computed.

        Returns:
            An evolution gradient.

        Raises:
            AlgorithmError: If a gradient job fails.
        """
        if self.non_hermitian:
            # 1: Split Hamiltonian into Hermitian and anti-Hermitian parts by H^+ = H + H^\dagger, H^- = H - H^\dagger
            h_plus = (hamiltonian + hamiltonian.adjoint())/2.0
            h_plus = h_plus.simplify()
            h_minus = (hamiltonian - hamiltonian.adjoint())/2.0
            h_minus = h_minus.simplify()  
            h_minus.coeffs = np.imag(h_minus.coeffs)

            # 2: Compute the gradient of each part
            try:
                evolution_grad_lse_rhs_plus = (
                    self.gradient.run([ansatz], [h_plus], [param_values], [gradient_params])
                    .result()
                    .gradients[0]
                )
                evolution_grad_lse_rhs_minus = (
                    self.gradient.run([ansatz], [h_minus], [param_values], [gradient_params], anti_hermitian=True)
                    .result()
                    .gradients[0]
                )
            except Exception as exc:
                raise AlgorithmError("The gradient primitive job failed!") from exc
            return -0.5 * np.real(evolution_grad_lse_rhs_plus + evolution_grad_lse_rhs_minus)
        else:
            try:
                evolution_grad_lse_rhs = (
                self.gradient.run([ansatz], [hamiltonian], [param_values], [gradient_params])
                .result()
                .gradients[0]
            )

            except Exception as exc:
                raise AlgorithmError("The gradient primitive job failed!") from exc
            return -0.5 * evolution_grad_lse_rhs

    @staticmethod
    def _validate_grad_settings(gradient):
        if (
            gradient is not None
            and hasattr(gradient, "_derivative_type")
            and gradient._derivative_type != DerivativeType.REAL
        ):
            warnings.warn(
                "A gradient instance with a setting for calculating imaginary part of "
                "the gradient was provided. This variational principle requires the"
                "real part. The setting to real was changed automatically."
            )
            gradient._derivative_type = DerivativeType.REAL
