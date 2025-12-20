# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
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
"""
Nelder-Mead Optimizer
=====================
The Nelder-Mead algorithm performs unconstrained optimization in
multidimensional spaces. It is based on the Simplex method and is
particularly useful when the (first and second) derivatives of the
objective function are unknown or unreliable. Unlike some other methods,
it does not take into account bounds or constraints on the variables.

Although the Nelder-Mead algorithm is generally robust and widely
applicable, it has some limitations. When derivatives can be accurately
computed, alternative algorithms that utilize this information may offer
better performance. These methods are often preferred due to their
ability to handle a wider range of scenarios and their tendency to
converge to more optimal solutions.

Nelder-Mead technique is a heuristic search approach, so it may converge
to non-stationary points or sub-optimal solutions.


"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Set
import warnings

import numpy as np
from numpy.typing import ArrayLike
from sympy import Symbol as SympySymbol

from lambeq.backend.symbol import Symbol
from lambeq.backend.tensor import Diagram
from lambeq.training.dataset import flatten
from lambeq.training.optimizer import Optimizer
from lambeq.training.quantum_model import QuantumModel


class NelderMeadOptimizer(Optimizer):
    """An optimizer based on the Nelder-Mead algorithm.

    This implementation is based heavily on SciPy's `optimize.minimize
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

    """
    model: QuantumModel
    bounds: np.ndarray | None

    def __init__(self,
                 *,
                 model: QuantumModel,
                 loss_fn: Callable[[Any, Any], float],
                 hyperparams: dict[str, float] | None = None,
                 bounds: ArrayLike | None = None) -> None:
        """Initialise the Nelder-Mead optimizer.

        The hyperparameters may contain the following key-value pairs:

        - `adaptive`: bool, default: False
            Adjust the algorithm's parameters based on the
            dimensionality of the problem. This is particularly helpful
            when minimizing functions in high-dimensional spaces.

        - `maxfev`: int, default: 1000
            Maximum number of function evaluations allowed.

        - `initial_simplex`: ArrayLike (N+1, N), default: None
            If provided, replaces the initial model weights. Each row
            should contain the coordinates of the `i`th vertex of the
            `N+1` vertices in the simplex, where `N` is the dimension.

        - `xatol`: float, default: 1e-4
            The acceptable level of absolute error in the optimal model
            weights (optimal solution) between iterations that indicates
            convergence.

        - `fatol`: float, default: 1e-4
            The acceptable level of absolute error in the loss value
            between iterations that indicates convergence.

        Parameters
        ----------
        model : :py:class:`.QuantumModel`
            A lambeq quantum model.
        hyperparams : dict of str to float
            A dictionary containing the models hyperparameters.
        loss_fn : Callable[[ArrayLike, ArrayLike], float]]
            A loss function of form `loss(prediction, labels)`.
        bounds : ArrayLike, optional
            The range of each of the model parameters.

        Raises
        ------
        ValueError
            - If the hyperparameters are not set correctly, or if the
              length of `bounds` does not match the number of the model
              parameters.
            - If the lower bounds are greater than the upper bounds.
            - If the initial simplex is not a 2D array.
            - If the initial simplex does not have N+1 rows, where N is
              the number of model parameters.

        Warning
            - If the initial model weights are not within the bounds.

        References
        ----------
        Gao, Fuchang & Han, Lixing. (2012). Implementing the Nelder-Mead
        Simplex Algorithm with Adaptive Parameters.
        `Computational Optimization and Applications`, 51. 259-277.
        10.1007/s10589-010-9329-3.

        """
        if hyperparams is None:
            hyperparams = {}

        super().__init__(model=model,
                         hyperparams=hyperparams,
                         loss_fn=loss_fn,
                         bounds=bounds)
        self.ncalls = 0

        self.current_sweep = 1
        self.adaptive = hyperparams.get('adaptive', False)
        self.maxfev = hyperparams.get('maxfev', 1000)
        self.initial_simplex = hyperparams.get('initial_simplex', None)
        self.xatol = hyperparams.get('xatol', 1e-4)
        self.fatol = hyperparams.get('fatol', 1e-4)

        if self.adaptive:
            self.dim = float(len(self.model.weights))
            self.rho = 1
            self.chi = 1 + 2 / self.dim
            self.psi = 0.75 - 1 / (2 * self.dim)
            self.sigma = 1 - 1 / self.dim
        else:
            self.rho = 1
            self.chi = 2
            self.psi = 0.5
            self.sigma = 0.5

        self.nonzdelt = 0.05
        self.zdelt = 0.00025

        if bounds is None:
            self.bounds = None
        else:
            self.bounds = np.asarray(bounds)
            if len(self.bounds) != len(self.model.weights):
                raise ValueError(
                    'Length of `bounds` must be the same as the '
                    'number of the model parameters'
                )

            lower_bound = self.bounds[:, 0]
            upper_bound = self.bounds[:, 1]

            # check bounds
            if (lower_bound > upper_bound).any():
                raise ValueError(
                    'Nelder-Mead lower bounds must be less than upper bounds.'
                )

            if (
                np.any(lower_bound > self.model.weights)
                or np.any(self.model.weights > upper_bound)
            ):
                warnings.warn(
                    'Initial value of model weights is not within the bounds.',
                    stacklevel=2,
                )

            self.model.weights = self.project(self.model.weights)

        self.N = len(self.model.weights)
        if self.initial_simplex is None:
            self.sim = np.empty((self.N + 1, self.N),
                                dtype=self.model.weights.dtype)
            self.sim[0] = model.weights
            for k in range(self.N):
                y = np.array(self.model.weights, copy=True)
                if y[k] != 0:
                    y[k] = (1 + self.nonzdelt) * y[k]
                else:
                    y[k] = self.zdelt
                self.sim[k + 1] = y
        else:
            self.sim = np.asfarray(self.initial_simplex).copy()
            if (
                self.sim.ndim != 2
                or self.sim.shape[0] != self.sim.shape[1] + 1
            ):
                raise ValueError(
                    '`initial_simplex` should be an array of shape '
                    f'({self.N + 1}, {self.N})'
                )
            if len(self.model.weights) != self.sim.shape[1]:
                raise ValueError(
                    'Size of `initial_simplex` is not consistent with '
                    '`model.weights`'
                )
            self.N = self.sim.shape[1]

        self.sim = self.project(self.sim)
        self.fsim = np.full((self.N + 1,), np.inf, dtype=float)
        self.first_iter = True  # flag for first iteration

    def project(self, x: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return x
        else:
            return x.clip(self.bounds[:, 0], self.bounds[:, 1])

    def objective(self, x: Iterable[Any], y: ArrayLike, w: ArrayLike) -> float:
        """The objective function to be minimized.

        Parameters
        ----------
        x : ArrayLike
            The input data.
        y : ArrayLike
            The labels.
        w : ArrayLike
            The model parameters.

        Returns
        -------
        result: float
            The result of the objective function.

        Raises
        ------
        ValueError
            If the objective function does not return a scalar value.

        """
        self.ncalls += 1
        self.model.weights = np.copy(w)
        result = self.loss_fn(self.model(x), y)

        # `result` must be a scalar
        if not np.isscalar(result):
            try:
                result = np.asarray(result).item()
            except (TypeError, ValueError) as e:
                raise ValueError(
                    'Objective function must return a scalar'
                ) from e
        return result

    def backward(self, batch: tuple[Iterable[Any], np.ndarray]) -> float:
        """Calculate the gradients of the loss function.

        The gradients are calculated with respect to the model
        parameters.

        Parameters
        ----------
        batch : tuple of Iterable and numpy.ndarray
            Current batch. Contains an Iterable of diagrams in index 0,
            and the targets in index 1.

        Returns
        -------
        float
            The calculated loss.

        """
        diagrams, targets = batch
        diags_gen = flatten(diagrams)

        # the symbolic parameters
        parameters = self.model.symbols

        # try to extract the relevant parameters from the diagrams
        diags = [diag.free_symbols
                 for diag in diags_gen if isinstance(diag, Diagram)]
        relevant_params: Set[SympySymbol] | Set[Symbol] = set.union(*diags)
        if not relevant_params:
            relevant_params = set(parameters)
        mask = np.array([int(sym in relevant_params) for sym in parameters])

        # Initialize ind, sim and fsim
        if self.first_iter:
            for k in range(self.N + 1):
                self.fsim[k] = self.objective(diagrams, targets, self.sim[k])

            self.ind = np.argsort(self.fsim)
            self.sim = self.sim.take(self.ind, 0)
            self.fsim = self.fsim.take(self.ind, 0)
            self.first_iter = False

        if not (
            np.abs(self.sim[1:] - self.sim[0]).max() <= self.xatol
            and np.abs(self.fsim[0] - self.fsim[1:]).max() <= self.fatol
        ):
            xbar = self.sim[:-1].sum(0) / self.N
            xr = self.project((1 + self.rho) * xbar - self.rho * self.sim[-1])
            fxr = self.objective(diagrams, targets, xr)
            shrink = False

            if fxr < self.fsim[0]:
                xe = self.project(
                    xbar + self.rho * self.chi * (xbar - self.sim[-1])
                )
                fxe = self.objective(diagrams, targets, xe)
                if fxe < fxr:
                    self.sim[-1] = xe
                    self.fsim[-1] = fxe
                else:
                    self.sim[-1] = xr
                    self.fsim[-1] = fxr
            elif fxr < self.fsim[-2]:  # and fsim[0] <= fxr
                self.sim[-1] = xr
                self.fsim[-1] = fxr
            elif fxr < self.fsim[-1]:  # and fxr >= fsim[-2] and fsim[0] <= fxr
                # Perform contraction
                xc = self.project(
                    xbar + self.psi * self.rho * (xbar - self.sim[-1])
                )
                fxc = self.objective(diagrams, targets, xc)

                if fxc <= fxr:
                    self.sim[-1] = xc
                    self.fsim[-1] = fxc
                else:
                    shrink = True
            else:  # Perform an inside contraction
                xcc = self.project(xbar + self.psi * (self.sim[-1] - xbar))
                fxcc = self.objective(diagrams, targets, xcc)

                if fxcc < self.fsim[-1]:
                    self.sim[-1] = xcc
                    self.fsim[-1] = fxcc
                else:
                    shrink = True

            if shrink:
                for j in range(1, self.N + 1):
                    self.sim[j] = self.project(
                        self.sim[0]
                        + self.sigma * (self.sim[j] - self.sim[0])
                    )
                    self.fsim[j] = self.objective(diagrams,
                                                  targets,
                                                  self.sim[j])

        self.ind = np.argsort(self.fsim)
        self.sim = self.sim.take(self.ind, 0)
        self.fsim = self.fsim.take(self.ind, 0)

        loss = float(np.min(self.fsim))
        self.gradient = self.sim[0] * mask

        if self.ncalls >= self.maxfev:
            warnings.warn(
                'Maximum number of function evaluations exceeded.',
                stacklevel=3
            )

        return loss

    def step(self) -> None:
        """Perform optimisation step."""
        self.model.weights = np.copy(self.gradient)
        self.model.weights = self.project(self.model.weights)
        self.update_hyper_params()
        self.zero_grad()

    def update_hyper_params(self) -> None:
        """Update the hyperparameters of the Nelder-Mead algorithm."""
        self.current_sweep += 1

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer states as dictionary.

        Returns
        -------
        dict
            A dictionary containing the current state of the optimizer.

        """
        return {
            'adaptive': self.adaptive,
            'initial_simplex': self.initial_simplex,
            'xatol': self.xatol,
            'fatol': self.fatol,
            'sim': self.sim,
            'fsim': self.fsim,
            'ncalls': self.ncalls,
            'first_iter': self.first_iter,
            'current_sweep': self.current_sweep,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load state of the optimizer from the state dictionary.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing a snapshot of the optimizer state.

        """
        self.adaptive = state_dict['adaptive']
        self.initial_simplex = state_dict['initial_simplex']
        self.xatol = state_dict['xatol']
        self.fatol = state_dict['fatol']
        self.sim = state_dict['sim']
        self.fsim = state_dict['fsim']
        self.ncalls = state_dict['ncalls']
        self.first_iter = state_dict['first_iter']
        self.current_sweep = state_dict['current_sweep']
