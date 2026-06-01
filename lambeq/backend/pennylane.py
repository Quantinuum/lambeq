# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
PennyLane interface
===================
Lambeq's interface with Pennylane circuits. This work is
based on DisCoPy (https://discopy.org/) which is released under the
BSD 3-Clause 'New' or 'Revised' License.

Notes
-----

If `probabilities` is set to False, the output states of the PennyLane
circuit will be exactly equivalent to those of the lambeq circuit
(for the same parameters).

If `probabilities` is set to True, the output states of the PennyLane
circuit will be the probabilities of the output states, equivalent
to appending :class:`lambeq.backend.quantum.Measure` to all the
open wires in the lambeq circuit.

Once a :class:`PennyLaneCircuit` has been constructed, it
can be evaluated with :func:`.eval()`. If the circuit contains only
concrete parameters (i.e. no symbolic parameters), no arguments
should be passed to `eval()`. If the circuit contains symbolic
parameters, a list of the symbolic parameters and a list of their
associated weights should be passed to `eval()` as `symbols=` and
`weights=`.
"""

from __future__ import annotations

from itertools import product
import sys
from typing import List, Set, Tuple, TYPE_CHECKING, Union

import pennylane as qml
import torch
from typing_extensions import Never

from lambeq.backend.quantum import (Gate, Measure,
                                    readoff_circuital,
                                    to_circuital)
from lambeq.backend.symbol import lambdify, Symbol

if TYPE_CHECKING:
    from lambeq.backend.quantum import Diagram


OP_MAP = {
    'H': qml.Hadamard,
    'X': qml.PauliX,
    'Y': qml.PauliY,
    'Z': qml.PauliZ,
    'S': qml.S,
    'Sdg': lambda wires: qml.S(wires=wires).inv(),
    'T': qml.T,
    'Tdg': lambda wires: qml.T(wires=wires).inv(),
    'Rx': qml.RX,
    'Ry': qml.RY,
    'Rz': qml.RZ,
    'CX': qml.CNOT,
    'CY': qml.CY,
    'CZ': qml.CZ,
    'CRx': qml.CRX,
    'CRy': qml.CRY,
    'CRz': qml.CRZ,
    'CU1': lambda a, wires: qml.ctrl(qml.U1(a,
                                            wires=wires[1]),
                                     control=wires[0]),
    'SWAP': qml.SWAP,
    'noop': qml.Identity,
}


def extract_ops_from_circuital(
    gates: List['Gate']
) -> Tuple[
        List[qml.operation.Operation],
        List[List[Union[torch.Tensor, Symbol, Never]]],
        Set[Symbol],
        List[List[int]]
]:
    """
    Extract the operation, parameters and wires from
    a circuital diagram dictionary, and return the corresponding PennyLane
    operation.

    Parameters
    ----------
    circuit_dict : :class:`Dict`
        The circuital dictionary to convert.

    Returns
    -------
    list of :class:`qml.operation.Operation`
        The PennyLane operation equivalent to the input pytket Op.
    list of (:class:`torch.FloatTensor` or
             :class:`lambeq.backend.symbol.Symbol`)
        The parameters of the operation.
    list of :class:`lambeq.backend.symbol.Symbol`
        The free symbols in the parameters of the operation.
    list of lists of int
        The wires/qubits to apply the operation to.

    """
    ops = [OP_MAP[x.gtype] for x in gates]
    qubits = [x.qubits for x in gates]
    params: list[Union[Symbol, float, int,
                       list, torch.Tensor]] = [x.phase
                                               if x.phase
                                               else []
                                               for x in gates]

    symbols = set()

    remapped_params: list[list[Union[Symbol, torch.Tensor, Never]]] = []
    for param in params:

        # Check if the param contains a symbol
        if isinstance(param, list) and len(param) == 0:
            remapped_params.append([])
            continue
        elif not isinstance(param, Symbol):
            param = torch.tensor(param)
        else:
            symbols.add(param)

        remapped_params.append([param])

    return ops, remapped_params, symbols, qubits


def to_pennylane(diagram: Diagram,
                 probabilities=False,
                 backend_config=None,
                 diff_method='best') -> PennyLaneCircuit:
    """
    Return a PennyLaneCircuit equivalent to the input lambeq
    circuit. `probabilities` determines whether the PennyLaneCircuit
    returns states (as in lambeq), or probabilities (to be more
    compatible with automatic differentiation in PennyLane).

    Parameters
    ----------
    diagram : :class:`lambeq.backend.quantum.Diagram`
        The lambeq circuit to convert to PennyLane.
    probabilities : bool, default: False
        Determines whether the PennyLane
        circuit outputs states or un-normalized probabilities.
        Probabilities can be used with more PennyLane backpropagation
        methods.
    backend_config : dict, default: None
        A dictionary of PennyLane backend configuration options,
        including the provider (e.g. IBM or Honeywell), the device,
        the number of shots, etc. See the `PennyLane plugin
        documentation <https://pennylane.ai/plugins/>`_
        for more details.
    diff_method : str, default: "best"
        The differentiation method to use to obtain gradients for the
        PennyLane circuit. Some gradient methods are only compatible
        with simulated circuits. See the `PennyLane documentation
        <https://docs.pennylane.ai/en/stable/introduction/interfaces.html>`_
        for more details.

    Returns
    -------
    :class:`PennyLaneCircuit`
        The PennyLane circuit equivalent to the input lambeq circuit.

    """
    if any(isinstance(box, Measure) for box in diagram.boxes):
        raise ValueError('Only pure circuits, or circuits with discards'
                         ' are currently supported.')

    if diagram.is_mixed and diagram.cod:
        # Some qubits discarded, some left open
        print('Warning: Circuit includes both discards and open codomain'
              ' wires. All open wires will be discarded during conversion',
              file=sys.stderr)

    is_mixed = diagram.is_mixed

    if not diagram.is_circuital:
        diagram = to_circuital(diagram)

    circuit_info = readoff_circuital(diagram)

    scalar = 1.0
    for gate in circuit_info.gates:
        if gate.gtype == 'Scalar' and gate.phase is not None:
            scalar *= gate.phase
            circuit_info.gates.remove(gate)

    ex_ops = extract_ops_from_circuital(circuit_info.gates)
    op_list, params_list, symbols_set, wires_list = ex_ops

    # Get post selection bits
    post_selection = circuit_info.postmap

    return PennyLaneCircuit(op_list,
                            list(symbols_set),
                            params_list,
                            wires_list,
                            probabilities,
                            post_selection,
                            is_mixed,
                            scalar,
                            circuit_info.total_qubits,
                            backend_config,
                            diff_method)


STATE_BACKENDS = ['default.qubit', 'lightning.qubit', 'qiskit.aer']
STATE_DEVICES = ['aer_simulator_statevector', 'statevector_simulator']


class PennyLaneCircuit:
    """Implement a pennylane circuit with post-selection."""

    def __init__(self, ops, symbols, params, wires, probabilities,
                 post_selection, mixed, scale, n_qubits, backend_config,
                 diff_method):
        self._ops = ops
        self._symbols = symbols
        self._params = params
        self._wires = wires
        self._probabilities = probabilities
        self._post_selection = post_selection
        self._mixed = mixed
        self._scale = scale
        self._n_qubits = n_qubits
        self._backend_config = backend_config
        self.diff_method = diff_method

        self._contains_symbols = self.contains_symbols()
        if self._contains_symbols:
            self._concrete_params = None
        else:
            self._concrete_params = params
        self.initialise_device_and_circuit()
        self._valid_states = self.get_valid_states()

    def get_device(self, backend_config):
        """
        Return a PennyLane device with the specified backend
        configuration.
        """
        if backend_config is None:
            backend = 'default.qubit'
            backend_config = {}
        else:
            backend = backend_config.pop('backend')

        if backend == 'honeywell.hqs':
            try:
                backend_config['machine'] = backend_config.pop('device')
            except KeyError:
                raise ValueError('When using the honeywell.hqs provider, '
                                 'a device must be specified.')
        elif 'device' in backend_config:
            backend_config['backend'] = backend_config.pop('device')

        if not self._probabilities:
            if backend not in STATE_BACKENDS:
                raise ValueError(f'The {backend} backend is not '
                                 'compatible with state outputs.')
            elif ('backend' in backend_config
                  and backend_config['backend'] not in STATE_DEVICES):
                raise ValueError(f'The {backend_config["backend"]} '
                                 'device is not compatible with state '
                                 'outputs.')

        return qml.device(backend, wires=self._n_qubits, **backend_config)

    def initialise_device_and_circuit(self):
        """
        Initialise the PennyLane device and circuit when instantiating the
        PennyLaneCirucit, or loading from disk.
        """
        self._device = self.get_device(None if self._backend_config is None
                                       else {**self._backend_config})
        self._circuit = self.make_circuit()

    def contains_symbols(self):
        """
        Determine if the circuit parameters are
        concrete or contain SymPy symbols.

        Returns
        -------
        bool
            Whether the circuit parameters contain SymPy symbols.
        """
        return any(isinstance(expr, Symbol) for expr_list in
                   self._params for expr in expr_list)

    def initialise_concrete_params(self, symbol_weight_map):
        """
        Given concrete values for each of the SymPy symbols, substitute
        the symbols for the values to obtain concrete parameters, via
        the `param_substitution` method.
        """
        if self._contains_symbols:
            weights = [symbol_weight_map[symbol] for symbol in self._symbols]
            self._concrete_params = self.param_substitution(weights)

    def draw(self):
        """
        Print a string representation of the circuit
        similar to `qml.draw`, but including post-selection.

        Parameters
        ----------
        symbols : list of :class:`lambeq.Symbol`, default: None
            The symbols from the original lambeq circuit.
        weights : list of :class:`torch.FloatTensor`, default: None
            The weights to substitute for the symbols.
        """
        if self._concrete_params is None:
            raise ValueError('Cannot draw circuit with symbolic parameters. '
                             'Initialise concrete parameters first.')

        wires = (qml.draw(self._circuit)
                 (self._concrete_params).split('\n'))
        for k, v in self._post_selection.items():
            wires[k] = wires[k].split('┤')[0] + '┤' + str(v) + '>'

        print('\n'.join(wires))

    def get_valid_states(self):
        """
        Determine which of the output states of the circuit are
        compatible with the post-selections.

        Returns
        -------
        list of int
            The indices of the circuit output that are
            compatible with the post-selections.
        """
        keep_indices = []
        fixed = ['0' if self._post_selection.get(i, 0) == 0 else
                 '1' for i in range(self._n_qubits)]
        open_wires = set(range(self._n_qubits)) - self._post_selection.keys()
        permutations = [''.join(s) for s in product('01',
                                                    repeat=len(open_wires))]
        for perm in permutations:
            new = fixed.copy()
            for i, open in enumerate(open_wires):
                new[open] = perm[i]
            keep_indices.append(int(''.join(new), 2))
        return keep_indices

    def make_circuit(self):
        """
        Construct the :class:`qml.Qnode`, a circuit that can be used with
        autograd to construct hybrid models.

        Returns
        -------
        :class:`qml.Qnode`
            A Pennylane circuit without post-selection.
        """
        @qml.qnode(self._device, interface='torch',
                   diff_method=self.diff_method)
        def circuit(circ_params):
            for op, params, wires in zip(self._ops, circ_params, self._wires):
                op(*[2 * torch.pi * p for p in params], wires=wires)

            if self._mixed:
                return qml.density_matrix(self._post_selection.keys())
            if self._probabilities:
                return qml.probs(wires=range(self._n_qubits))
            else:
                return qml.state()

        return circuit

    def post_selected_circuit(self, params):
        """
        Run the circuit with the given parameters and return
        the post-selected output.

        Parameters
        ----------
        params : :class:`torch.FloatTensor`
            The concrete parameters for the gates in the circuit.

        Returns
        -------
        :class:`torch.Tensor`
            The post-selected output of the circuit.
        """
        states = self._circuit(params)

        if self._mixed:
            # Select the all-zeros subsystem
            return states[0][0]

        open_wires = self._n_qubits - len(self._post_selection)
        post_selected_states = states[self._valid_states]
        post_selected_states *= (self._scale ** 2 if self._probabilities
                                 else self._scale)

        if post_selected_states.shape[0] == 1:
            return post_selected_states
        else:
            return torch.reshape(post_selected_states, (2,) * open_wires)

    def param_substitution(self, weights):
        """
        Substitute symbolic parameters (`lambeq.Symbol`s) with floats.

        Parameters
        ----------
        weights : list of :class:`torch.FloatTensor`
            The weights to substitute for the symbols.

        Returns
        -------
        :class:`torch.FloatTensor`
            The concrete (non-symbolic) parameters for the
            circuit.
        """
        concrete_params = []
        for expr_list in self._params:
            concrete_list = []
            for expr in expr_list:
                if isinstance(expr, Symbol):
                    f_expr = lambdify(self._symbols, expr)
                    expr = f_expr(*weights)
                concrete_list.append(expr)
            concrete_params.append(concrete_list)

        return concrete_params

    def eval(self):
        """
        Evaluate the circuit. The symbols should be those
        from the original lambeq diagram, which will be substituted
        for the concrete parameters in weights.

        Parameters
        ----------
        symbols : list of :class:`lambeq.Symbol`, default: None
            The symbols from the original lambeq circuit.
        weights : list of :class:`torch.FloatTensor`, default: None
            The weights to substitute for the symbols.

        Returns
        -------
        :class:`torch.Tensor`
            The post-selected output of the circuit.
        """
        if self._concrete_params is None:
            raise ValueError('Initialise concrete parameters first.')

        return self.post_selected_circuit(self._concrete_params)
