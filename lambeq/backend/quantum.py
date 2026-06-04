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
Quantum category
================
Lambeq's internal representation of the quantum category. This work is
based on DisCoPy (https://discopy.org/) which is released under the
BSD 3-Clause 'New' or 'Revised' License.

Notes
-----

In lambeq, gates are represented as the transpose of their matrix
according to the standard convention in quantum computing. This makes
composition of gates using the tensornetwork library easier.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import cast, Dict, Optional, Tuple, Union

import numpy as np
import tensornetwork as tn
from typing_extensions import Any, Self

from lambeq.backend import Functor, grammar, Symbol, tensor
from lambeq.backend.numerical_backend import backend, get_backend
from lambeq.backend.symbol import lambdify
from lambeq.core.utils import fast_deepcopy


quantum = grammar.Category('quantum')


@quantum
class Ty(tensor.Dim):
    """A type in the quantum category."""

    def __init__(self,
                 name: str | None = None,
                 objects: list[Self] | None = None):
        """Initialise a type in the quantum category.

        Parameters
        ----------
        name : str, optional
            The name of the type, by default None
        objects : list[Ty], optional
            The objects defining a complex type, by default None

        """

        if objects:
            super().__init__(objects=objects)
            self.name = None
            self.label = None
        else:
            if name is None:
                super().__init__()
            else:
                super().__init__(2)
            self.label = name

    def _repr_rec(self) -> str:
        if self.is_empty:
            return ''
        elif self.is_atomic:
            return f'{self.label}'
        else:
            return ' @ '.join(d._repr_rec() for d in self.objects)

    def __str__(self) -> str:
        return self.label if self.label else ''

    def __repr__(self) -> str:
        return f'Ty({self._repr_rec()})'

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other):
        return (self.label == other.label
                and self.name == other.name
                and self.objects == other.objects)


qubit = Ty('qubit')
bit = Ty('bit')


@quantum
class Box(tensor.Box):
    """A box in the quantum category."""
    name: str
    dom: Ty
    cod: Ty
    data: float | np.ndarray | None
    z: int
    is_mixed: bool
    self_adjoint: bool

    def __init__(self,
                 name: str,
                 dom: Ty,
                 cod: Ty,
                 data: float | np.ndarray | None = None,
                 z: int = 0,
                 is_mixed: bool = False,
                 self_adjoint: bool = False):
        """Initialise a box in the quantum category.

        Parameters
        ----------
        name : str
            Name of the box.
        dom : Ty
            Domain of the box.
        cod : Ty
            Codomain of the box.
        data : float | np.ndarray, optional
            Array defining the tensor of the box, by default None
        z : int, optional
            The winding number, by default 0
        is_mixed : bool, optional
            Whether the box is mixed, by default False
        self_adjoint : bool, optional
            Whether the box is self-adjoint, by default False

        """
        self.name = name
        self.dom = dom
        self.cod = cod
        self.data = data
        self.z = z
        self.is_mixed = is_mixed
        self.self_adjoint = self_adjoint

    @property
    def is_classical(self) -> bool:
        return set(self.dom @ self.cod) == {bit}

    def dagger(self) -> Daggered | Box:
        """Return the dagger of the box."""
        if self.self_adjoint:
            return self

        return Daggered(self)

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
@quantum
class Layer(tensor.Layer):
    """A Layer in a quantum Diagram.

    Parameters
    ----------
    box : Box
        The box of the layer.
    left : Ty
        The wire type to the left of the box.
    right : Ty
        The wire type to the right of the box.

    """

    left: Ty
    box: Box
    right: Ty


@dataclass
@quantum
class Diagram(tensor.Diagram):
    """A diagram in the quantum category.

    Parameters
    ----------
    dom : Ty
        The type of the input wires.
    cod : Ty
        The type of the output wires.
    layers : list[Layer]
        The layers of the diagram.

    """

    dom: Ty
    cod: Ty
    layers: list[Layer]  # type: ignore[assignment]

    def __getattr__(self, name: str) -> Any:
        try:
            gate = GATES[name]
            if callable(gate):
                return partial(self.apply_parametrized_gate, gate)
            return partial(self.apply_gate, gate)
        except KeyError:
            return super().__getattr__(name)  # type: ignore[misc]

    def apply_parametrized_gate(self,
                                gate: Callable[[float], Parametrized],
                                param: float,
                                *qubits: int) -> Self:
        return self.apply_gate(gate(param), *qubits)

    def apply_gate(self, gate: Box, *qubits: int) -> Self:
        if isinstance(gate, Controlled):
            min_idx = min(qubits)
            final_gate: Box

            if isinstance(gate.controlled, Controlled):
                assert len(qubits) == 3
                atomic = gate.controlled.controlled
                dist1 = qubits[2] - qubits[0]
                dist2 = qubits[2] - qubits[1]

                if dist1 * dist2 < 0:  # sign flip
                    final_gate = Controlled(Controlled(atomic, dist1), dist2)
                else:
                    dists = np.array([dist1, dist2])
                    idx = np.argmin(np.abs(dists)), np.argmax(np.abs(dists))
                    final_gate = Controlled(Controlled(atomic, dists[idx[0]]),
                                            dists[idx[1]]-dists[idx[0]])

            else:
                # Singly controlled
                assert len(qubits) == 2
                dist = qubits[1] - qubits[0]

                final_gate = Controlled(gate.controlled, dist)

            return self.then_at(final_gate, min_idx)

        else:
            assert len(qubits) == len(gate.dom)
            return self.then_at(gate, min(qubits))

    @property
    def is_mixed(self) -> bool:
        """Whether the diagram is mixed.

        A diagram is mixed if it contains a mixed box or if it has both
        classical and quantum wires.

        """

        dom_n_cod = self.dom @ self.cod
        mixed_boundary = bit in dom_n_cod and qubit in dom_n_cod

        return mixed_boundary or any(box.is_mixed for box in self.boxes)

    @property
    def is_circuital(self) -> bool:
        """Checks if this diagram is a 'circuital' quantum diagram.

        Circuital means:
            1. All initial layers are qubits
            2. All post selections are at the end

        Allows for mixed_circuit measurements.

        Returns
        -------
        bool
            Whether this diagram is a circuital diagram.

        """

        if self.dom:
            return False

        layers = self.layers

        num_qubits = sum([1 for layer in layers
                          if isinstance(layer.box, Ket)])

        qubit_layers = layers[:num_qubits]

        if not all([isinstance(layer.box, Ket) for layer in qubit_layers]):
            return False

        for qubit_layer in qubit_layers:
            if len(qubit_layer.right):
                return False

        # Check there are no gates in between post-selections.
        measure_idx = [i for i, layer in enumerate(layers[num_qubits:])
                       if isinstance(layer.box, (Discard, Bra))]
        if not measure_idx:
            return True
        mmax = max(measure_idx)
        mmin = min(measure_idx)
        for i, gate in enumerate(layers[num_qubits:]):
            if not isinstance(gate.box, (Discard, Bra, Measure)):
                if i > mmin and i < mmax:
                    return False

        return True

    def eval(self,
             *others,
             backend=None,
             mixed=False,
             contractor=tn.contractors.auto,
             **params):
        """Evaluate the circuit represented by the diagram.

        Be aware that this method is only suitable for small circuits with
        a small number of qubits (depending on hardware resources).

        Parameters
        ----------
        others : :class:`lambeq.backend.quantum.Diagram`
            Other circuits to process in batch if backend is set to tket.
        backend : pytket.Backend, optional
            Backend on which to run the circuit, if none then we apply
            tensor contraction.
        mixed : bool, optional
            Whether the circuit is mixed, by default False
        contractor : Callable, optional
            The contractor to use, by default tn.contractors.auto

        Returns
        -------
        np.ndarray or list of np.ndarray
            The result of the circuit simulation.

        """
        if backend is None:
            return contractor(*self.to_tn(mixed=mixed)).tensor

        circuits = [circuit.to_tk() for circuit in (self, ) + others]
        results, counts = [], circuits[0].get_counts(
            *circuits[1:], backend=backend, **params)

        for i, circuit in enumerate(circuits):
            n_bits = len(circuit.post_processing.dom)
            result = np.zeros((n_bits * (2, )))
            for bitstring, count in counts[i].items():
                result[bitstring] = count
            if circuit.post_processing:
                post_result = circuit.post_processing.eval().astype(float)

                if result.shape and post_result.shape:
                    result = np.tensordot(result, post_result, -1)
                else:
                    result * post_result
            results.append(result)
        return results if len(results) > 1 else results[0]

    def init_and_discard(self):
        """Return circuit with empty domain and only bits as codomain. """
        circuit = self
        if circuit.dom:
            init = Id().tensor(*(Ket(0) if x == qubit else Bit(0)
                                 for x in circuit.dom))
            circuit = init >> circuit
        if circuit.cod != bit ** len(circuit.cod):
            discards = Id().tensor(*(
                Discard() if x == qubit
                else Id(bit) for x in circuit.cod))
            circuit = circuit >> discards
        return circuit

    def to_tk(self):
        """Export to t|ket>.

        Returns
        -------
        tk_circuit : lambeq.backend.converters.tk.Circuit
            A :class:`lambeq.backend.converters.tk.Circuit`.

        Notes
        -----
        * No measurements are performed.
        * SWAP gates are treated as logical swaps.
        * If the circuit contains scalars or a :class:`Bra`,
          then :code:`tk_circuit` will hold attributes
          :code:`post_selection` and :code:`scalar`.

        Examples
        --------
        >>> from lambeq.backend.quantum import *

        >>> bell_test = H @ Id(qubit) >> CX >> Measure() @ Measure()
        >>> bell_test.to_tk()
        tk.Circuit(2, 2).H(0).CX(0, 1).Measure(0, 0).Measure(1, 1)

        >>> circuit0 = (Sqrt(2) @ H @ Rx(0.5) >> CX >>
        ...             Measure() @ Discard())
        >>> circuit0.to_tk()
        tk.Circuit(2, 1).H(0).Rx(1.0, 1).CX(0, 1).Measure(0, 0).scale(2)

        >>> circuit1 = Ket(1, 0) >> CX >> Id(qubit) @ Ket(0) @ Id(qubit)
        >>> circuit1.to_tk()
        tk.Circuit(3).X(0).CX(0, 2)

        >>> circuit2 = X @ Id(qubit ** 2) \\
        ...     >> Id(qubit) @ SWAP >> CX @ Id(qubit) >> Id(qubit) @ SWAP
        >>> circuit2.to_tk()
        tk.Circuit(3).X(0).SWAP(1, 2).CX(0, 1).SWAP(1, 2)

        >>> circuit3 = Ket(0, 0)\\
        ...     >> H @ Id(qubit)\\
        ...     >> CX\\
        ...     >> Id(qubit) @ Bra(0)
        >>> circuit3.to_tk()
        tk.Circuit(2, 1).H(0).CX(0, 1).Measure(1, 0).post_select({1: 0})
        """
        from lambeq.backend.converters.tk import to_tk
        return to_tk(self)

    def to_pennylane(self, probabilities=False, backend_config=None,
                     diff_method='best'):
        """
        Export lambeq circuit to PennylaneCircuit.

        Parameters
        ----------
        probabilities : bool, default: False
            If True, the PennylaneCircuit will return the normalized
            probabilities of measuring the computational basis states
            when run. If False, it returns the unnormalized quantum
            states in the computational basis.
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
        :class:`lambeq.backend.pennylane.PennylaneCircuit`

        """
        from lambeq.backend.pennylane import to_pennylane
        return to_pennylane(self, probabilities=probabilities,
                            backend_config=backend_config,
                            diff_method=diff_method)

    def to_tn(self, mixed=False):
        """Send a diagram to a mixed :code:`tensornetwork`.

        Parameters
        ----------
        mixed : bool, default: False
            Whether to perform mixed (also known as density matrix)
            evaluation of the circuit.

        Returns
        -------
        nodes : :class:`tensornetwork.Node`
            Nodes of the network.

        output_edge_order : list of :class:`tensornetwork.Edge`
            Output edges of the network.

        """

        if not mixed and not self.is_mixed:
            return super().to_tn(dtype=complex)

        diag = Id(self.dom)

        for left, box, right in self.layers:
            subdiag = box

            if hasattr(box, 'decompose'):
                subdiag = box.decompose()

            diag >>= Id(left) @ subdiag @ Id(right)

        c_nodes = [tn.CopyNode(2, 2, f'c_input_{i}', dtype=complex)
                   for i in range(list(diag.dom).count(bit))]
        q_nodes1 = [tn.CopyNode(2, 2, f'q1_input_{i}', dtype=complex)
                    for i in range(list(diag.dom).count(qubit))]
        q_nodes2 = [tn.CopyNode(2, 2, f'q2_input_{i}', dtype=complex)
                    for i in range(list(diag.dom).count(qubit))]

        inputs = [n[0] for n in c_nodes + q_nodes1 + q_nodes2]

        c_scan = [n[1] for n in c_nodes]
        q_scan1 = [n[1] for n in q_nodes1]
        q_scan2 = [n[1] for n in q_nodes2]

        nodes = c_nodes + q_nodes1 + q_nodes2

        for left, box, _ in diag.layers:
            c_offset = list(left).count(bit)
            q_offset = list(left).count(qubit)

            if isinstance(box, Swap) and box.is_classical:
                c_scan[q_offset], c_scan[q_offset + 1] = (c_scan[q_offset + 1],
                                                          c_scan[q_offset])

            elif isinstance(box, Discard):
                tn.connect(q_scan1[q_offset], q_scan2[q_offset])
                del q_scan1[q_offset]
                del q_scan2[q_offset]

            elif box.is_mixed:
                if isinstance(box, (Measure, Encode)):
                    node = tn.CopyNode(3, 2, 'cq_' + str(box), dtype=complex)

                elif isinstance(box, (MixedState)):
                    node = tn.CopyNode(2, 2, 'cq_' + str(box), dtype=complex)
                else:
                    node = tn.Node(box.data + 0j, 'cq_' + str(box))

                c_dom = list(box.dom).count(bit)
                q_dom = list(box.dom).count(qubit)
                c_cod = list(box.cod).count(bit)
                q_cod = list(box.cod).count(qubit)

                for i in range(c_dom):
                    tn.connect(c_scan[c_offset + i], node[i])

                for i in range(q_dom):
                    tn.connect(q_scan1[q_offset + i], node[c_dom + i])
                    tn.connect(q_scan2[q_offset + i], node[c_dom + q_dom + i])

                cq_dom = c_dom + 2 * q_dom

                c_edges = node[cq_dom: cq_dom + c_cod]
                q_edges1 = node[cq_dom + c_cod: cq_dom + c_cod + q_cod]
                q_edges2 = node[cq_dom + c_cod + q_cod:]

                c_scan[c_offset:c_offset + c_dom] = c_edges
                q_scan1[q_offset:q_offset + q_dom] = q_edges1
                q_scan2[q_offset:q_offset + q_dom] = q_edges2

                nodes.append(node)
            else:
                # Purely quantum box

                if isinstance(box, Swap):
                    for scan in (q_scan1, q_scan2):
                        (scan[q_offset],
                         scan[q_offset + 1]) = (scan[q_offset + 1],
                                                scan[q_offset])
                else:
                    utensor = box.array
                    node1 = tn.Node(utensor + 0j, 'q1_' + str(box))
                    with backend() as np:
                        node2 = tn.Node(np.conj(utensor) + 0j,
                                        'q2_' + str(box))

                    for i in range(len(box.dom)):
                        tn.connect(q_scan1[q_offset + i], node1[i])
                        tn.connect(q_scan2[q_offset + i], node2[i])

                    q_scan1[q_offset:q_offset
                            + len(box.dom)] = node1[len(box.dom):]
                    q_scan2[q_offset:q_offset
                            + len(box.dom)] = node2[len(box.dom):]

                    nodes.extend([node1, node2])

        outputs = c_scan + q_scan1 + q_scan2

        return nodes, inputs + outputs

    __hash__: Callable[[], int] = tensor.Diagram.__hash__


class SelfConjugate(Box):
    """A self-conjugate box is equal to its own conjugate."""

    def rotate(self, z):
        return self


class AntiConjugate(Box):
    """An anti-conjugate box is equal to the conjugate of its conjugate.
    """

    def rotate(self, z):
        if z % 2 == 0:
            return self
        return self.dagger()


@Diagram.register_special_box('cap')
def generate_cap(left: Ty, right: Ty, is_reversed=False) -> Diagram:
    """Generate a cap diagram.

    Parameters
    ----------
    left : Ty
        The left type of the cap.
    right : Ty
        The right type of the cap.
    is_reversed : bool, optional
        Unused, by default False

    Returns
    -------
    Diagram
        The cap diagram.

    """

    assert left == right

    atomic_cap = Ket(0) @ Ket(0) >> H @ Sqrt(2) @ qubit >> Controlled(X)

    d = Id()

    for i in range(len(left)):
        d = d.then_at(atomic_cap, i)

    return d


@Diagram.register_special_box('cup')
def generate_cup(left: Ty, right: Ty, is_reversed=False) -> Diagram:
    """Generate a cup diagram.

    Parameters
    ----------
    left : Ty
        The left type of the cup.
    right : Ty
        The right type of the cup.
    is_reversed : bool, optional
        Unused, by default False

    Returns
    -------
    Diagram
        The cup diagram.

    """

    assert left == right

    atomic_cup = Controlled(X) >> H @ Sqrt(2) @ qubit >> Bra(0) @ Bra(0)

    d = Id(left @ right)

    for i in range(len(left)):
        d = d.then_at(atomic_cup, len(left) - i - 1)

    return d


@Diagram.register_special_box('spider')
def generate_spider(type: Ty, n_legs_in: int, n_legs_out: int) -> Diagram:

    i, o = n_legs_in, n_legs_out
    if i == o == 1:
        return Id(type)

    if type == Ty():
        return Id()

    if type != qubit:
        raise NotImplementedError('Multi-qubit spiders are not presently'
                                  ' supported.')

    if (i, o) == (1, 0):
        return cast(Diagram, Sqrt(2) @ H >> Bra(0))
    if (i, o) == (2, 1):
        return cast(Diagram, CX >> Id(qubit) @ Bra(0))
    if o > i:
        return generate_spider(type, o, i).dagger()

    if o != 1:
        return generate_spider(type, i, 1) >> generate_spider(type, 1, o)
    if i % 2:
        return (generate_spider(type, i - 1, 1) @ Id(type)
                >> generate_spider(type, 2, 1))

    half_spiders = generate_spider(type, i // 2, 1)
    return half_spiders @ half_spiders >> generate_spider(type, 2, 1)


@Diagram.register_special_box('swap')
class Swap(tensor.Swap, SelfConjugate, Box):
    """A swap box in the quantum category."""
    type: Ty
    n_legs_in: int
    n_legs_out: int
    name: str
    dom: Ty
    cod: Ty
    z: int = 0

    def __init__(self, left: Ty, right: Ty):
        """Initialise a swap box.

        Parameters
        ----------
        left : Ty
            The left type of the swap.
        right : Ty
            The right type of the swap.

        """

        Box.__init__(self,
                     'SWAP',
                     left @ right,
                     right @ left,
                     np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]]))
        tensor.Swap.__init__(self, left, right)

    __hash__: Callable[[], int] = tensor.Swap.__hash__
    __repr__: Callable[[], str] = tensor.Swap.__repr__
    dagger = tensor.Swap.dagger


def Id(ty: Ty | int | None = None) -> Diagram:
    if isinstance(ty, int):
        ty = qubit ** ty
    return Diagram.id(ty)


class Ket(SelfConjugate, Box):
    """A ket in the quantum category.

    A ket is a box that initializes a qubit to a given state.

    """

    def __new__(cls, *bitstring: int):
        if len(bitstring) <= 1:
            return super(Ket, cls).__new__(cls)

        return Id().tensor(* [cls(bit) for bit in bitstring])

    def __init__(self, bit: int) -> None:
        """Initialise a ket box.

        Parameters
        ----------
        bit : int
            The state of the qubit (either 0 or 1).

        """

        assert bit in {0, 1}
        self.bit = bit

        super().__init__(str(bit), Ty(), qubit, np.eye(2)[bit].T)

    def dagger(self) -> Self:
        return Bra(self.bit)  # type: ignore[return-value]


class Bra(SelfConjugate, Box):
    """A bra in the quantum category.

    A bra is a box that measures a qubit in the computational basis and
    post-selects on a given state.

    """

    def __new__(cls, *bitstring: int):
        if len(bitstring) <= 1:
            return super(Bra, cls).__new__(cls)

        return Id().tensor(* [cls(bit) for bit in bitstring])

    def __init__(self, bit: int):
        """Initialise a bra box.

        Parameters
        ----------
        bit : int
            The state of the qubit to post-select on (either 0 or 1).

        """

        assert bit in {0, 1}
        self.bit = bit

        super().__init__(str(bit), qubit, Ty(), np.eye(2)[bit])

    def dagger(self) -> Self:
        return Ket(self.bit)  # type: ignore[return-value]


class Parametrized(Box):
    """A parametrized box in the quantum category.

    A parametrized box is a unitary gate that can be parametrized by a
    real number.

    Parameters
    ----------
    name : str
        The name of the box.
    dom : Ty
        The domain of the box.
    cod : Ty
        The codomain of the box.
    data : float
        The parameterised unitary of the box.
    is_mixed : bool, default: False
        Whether the box is mixed
    self_adjoint : bool, default: False
        Whether the box is self-adjoint

    """

    name: str
    dom: Ty
    cod: Ty
    data: float
    is_mixed: bool = False
    self_adjoint: bool = False

    def lambdify(self, *symbols, **kwargs):
        """Return a lambda function that evaluates the box."""

        return lambda *xs: type(self)(lambdify(
                                        symbols, self.data)(*xs))

    @property
    def modules(self):
        if self.free_symbols:
            raise RuntimeError(
                'Attempting to access modules for a symbolic expression. '
                + 'Eval of a symbolic expression is not supported.')
        else:
            return get_backend()


class Rotation(Parametrized):
    """Single qubit gate defining a rotation around the bloch sphere."""

    def __init__(self, phase):

        super().__init__(
            f'{type(self).__name__}({phase})', qubit, qubit, phase)

    @property
    def phase(self) -> float:
        return self.data

    def dagger(self) -> Self:
        return type(self)(-self.data)


class Rx(AntiConjugate, Rotation):
    """Single qubit gate defining a rotation aound the x-axis."""

    @property
    def array(self):
        with backend() as np:
            half_theta = np.pi * self.phase
            sin = self.modules.sin(half_theta)
            cos = self.modules.cos(half_theta)

            I_arr = np.eye(2)
            X_arr = np.array([[0, 1], [1, 0]])

            return cos * I_arr - 1j * sin * X_arr


class Ry(SelfConjugate, Rotation):
    """Single qubit gate defining a rotation aound the y-axis."""

    @property
    def array(self):
        with backend() as np:
            half_theta = np.pi * self.phase
            sin = self.modules.sin(half_theta)
            cos = self.modules.cos(half_theta)

            I_arr = np.eye(2)
            Y_arr = np.array([[0, 1j], [-1j, 0]])

            return cos * I_arr - 1j * sin * Y_arr


class Rz(AntiConjugate, Rotation):
    """Single qubit gate defining a rotation aound the z-axis."""

    @property
    def array(self):
        with backend() as np:
            half_theta = self.modules.pi * self.phase
            exp1 = np.e ** (-1j * half_theta)
            exp2 = np.e ** (1j * half_theta)

            P_0 = np.array([[1, 0], [0, 0]])
            P_1 = np.array([[0, 0], [0, 1]])

            return exp1 * P_0 + exp2 * P_1


class Controlled(Parametrized):
    """A gate that applies a unitary controlled by a qubit's state."""

    def __init__(self, controlled: Box, distance=1):
        """Initialise a controlled box.

        Parameters
        ----------
        controlled : Box
            The box to be controlled.
        distance : int, optional
            The distance between the control and the target, by default 1

        """

        assert distance

        self.distance = distance
        self.controlled = controlled

        width = len(controlled.dom) + abs(distance)

        super().__init__(f'C{controlled}',
                         qubit ** width,
                         qubit ** width,
                         controlled.data,
                         controlled.is_mixed)

    def __hash__(self) -> int:
        return hash((self.controlled, self.distance))

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == 'data':
            self.controlled.data = __value
        return super().__setattr__(__name, __value)

    @property
    def phase(self) -> float:
        if isinstance(self.controlled, Rotation):
            return self.controlled.phase
        else:
            raise AttributeError('Controlled gate has no phase.')

    def decompose(self) -> Diagram | Box:
        """Split a box (distance >1) into distance 1 box + swaps."""

        if self.distance == 1:
            return self

        n_qubits = len(self.dom)

        skipped_qbs = n_qubits - (1 + len(self.controlled.dom))

        if self.distance > 0:
            pattern = [0,
                       *range(skipped_qbs + 1, n_qubits),
                       *range(1, skipped_qbs + 1)]
        else:
            pattern = [n_qubits - 1, *range(n_qubits - 1)]

        perm: Diagram = Diagram.permutation(self.dom, pattern)
        diagram = (perm
                   >> type(self)(self.controlled) @ Id(qubit ** skipped_qbs)
                   >> perm.dagger())

        return diagram

    def lambdify(self, *symbols, **kwargs):
        """Return a lambda function that evaluates the box."""
        c_fn = self.controlled.lambdify(*symbols)

        return lambda *xs: type(self)(c_fn(*xs), distance=self.distance)

    @property
    def array(self):
        with backend() as np:
            controlled, distance = self.controlled, self.distance

            n_qubits = len(self.dom)
            if distance == 1:
                d = 1 << n_qubits - 1
                part1 = np.array([[1, 0], [0, 0]])
                part2 = np.array([[0, 0], [0, 1]])
                array = (np.kron(part1, np.eye(d))
                         + np.kron(part2,
                                   np.array(controlled.array.reshape(d, d))))
            else:
                array = self.decompose().eval()
            return array.reshape(*[2] * 2 * n_qubits)

    def dagger(self):
        """Return the dagger of the box."""
        return Controlled(self.controlled.dagger(), self.distance)

    def rotate(self, z):
        """Conjugate the box."""

        if z % 2 == 0:
            return self

        return Controlled(self.controlled.rotate(z), -self.distance)


class MixedState(SelfConjugate):
    """A mixed state is a state with a density matrix proportional to the
    identity matrix."""

    def __init__(self):
        super().__init__('MixedState', Ty(), qubit, is_mixed=True)

    def dagger(self):
        return Discard()


class Discard(SelfConjugate):
    """Discard a qubit. This is a measurement without post-selection."""

    def __init__(self):
        super().__init__('Discard', qubit, Ty(), is_mixed=True)

    def dagger(self):
        return MixedState()


class Measure(SelfConjugate):
    """Measure a qubit and return a classical information bit."""

    def __init__(self):
        super().__init__('Measure', qubit, bit, is_mixed=True)

    def dagger(self):
        return Encode()


class Encode(SelfConjugate):
    """Encode a classical information bit into a qubit."""

    def __init__(self):
        super().__init__('Encode', bit, qubit, is_mixed=True)

    def dagger(self):
        return Measure()


@dataclass
class Scalar(Box):
    """A scalar amplifies a quantum state by a given factor."""
    data: float | np.ndarray

    name: str = field(init=False)
    dom: Ty = field(default=Ty(), init=False)
    cod: Ty = field(default=Ty(), init=False)
    is_mixed: bool = field(default=False, init=False)
    self_adjoint: bool = field(default=False, init=False)
    z: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.name = f'{self.data:.3f}'

    @property
    def array(self):
        with backend() as np:
            return np.array(self.data)

    __hash__: Callable[[Box], int] = Box.__hash__   # type: ignore[assignment]

    def dagger(self):
        return replace(self, data=self.data.conjugate())


@dataclass
class Sqrt(Scalar):
    """A Square root."""
    data: float | np.ndarray

    name: str = field(init=False)
    dom: Ty = field(default=Ty(), init=False)
    cod: Ty = field(default=Ty(), init=False)
    is_mixed: bool = field(default=False, init=False)
    self_adjoint: bool = field(default=False, init=False)
    z: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.name = f'√({self.data})'

    @property
    def array(self):
        with backend() as np:
            return np.array(self.data ** .5)

    __hash__: Callable[[], int] = Scalar.__hash__   # type: ignore[assignment]

    def dagger(self):
        return replace(self, data=np.conjugate(self.data))


@dataclass
class Daggered(tensor.Daggered, Box):
    """A daggered gate reverses the box's effect on a quantum state.

    Parameters
    ----------
    box : Box
        The box to be daggered.

    """
    box: Box
    name: str = field(init=False)
    dom: Ty = field(init=False)
    cod: Ty = field(init=False)
    data: float | np.ndarray | None = field(default=None, init=False)
    is_mixed: bool = field(default=False, init=False)
    self_adjoint: bool = field(default=False, init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.box.name + '†'
        self.dom = self.box.cod
        self.cod = self.box.dom
        self.data = self.box.data
        self.z = 0
        self.is_mixed = self.box.is_mixed

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == 'data':
            self.box.data = __value
        return super().__setattr__(__name, __value)

    def dagger(self) -> Box:
        return self.box

    __hash__: Callable[[Box], int] = Box.__hash__   # type: ignore[assignment]
    __repr__: Callable[[Box], str] = Box.__repr__   # type: ignore[assignment]


class Bit(Box):
    """Classical state for a given bit."""

    def __new__(cls, *bitstring: int):
        if len(bitstring) <= 1:
            return super(Bit, cls).__new__(cls)

        return Id().tensor(* [cls(bit) for bit in bitstring])

    def __init__(self, bit_value: int) -> None:
        """Initialise a ket box.

        Parameters
        ----------
        bit_value : int
            The state of the qubit (either 0 or 1).

        """

        assert bit_value in {0, 1}
        self.bit = bit_value

        super().__init__(str(bit_value), Ty(), bit, np.eye(2)[bit_value].T)


SWAP = Swap(qubit, qubit)
H = SelfConjugate('H', qubit, qubit,
                  (2 ** -0.5) * np.array([[1, 1], [1, -1]]), self_adjoint=True)
S = Box('S', qubit, qubit, np.array([[1, 0], [0, 1j]]))
T = Box('T', qubit, qubit, np.array([[1, 0], [0, np.e ** (1j * np.pi / 4)]]))
X = SelfConjugate('X', qubit, qubit,
                  np.array([[0, 1], [1, 0]]), self_adjoint=True)
Y = Box('Y', qubit, qubit, np.array([[0, 1j], [-1j, 0]]), self_adjoint=True)
Z = SelfConjugate('Z', qubit, qubit,
                  np.array([[1, 0], [0, -1]]), self_adjoint=True)
CX = Controlled(X)
CY = Controlled(Y)
CZ = Controlled(Z)
CCX = Controlled(CX)
CCZ = Controlled(CZ)
def CRx(phi, distance=1): return Controlled(Rx(phi), distance)  # noqa: E731
def CRy(phi, distance=1): return Controlled(Ry(phi), distance)  # noqa: E731
def CRz(phi, distance=1): return Controlled(Rz(phi), distance)  # noqa: E731


GATES: Dict[str, Box | Callable[[Any], Parametrized]] = {
    'SWAP': SWAP,
    'H': H,
    'S': S,
    'T': T,
    'X': X,
    'Y': Y,
    'Z': Z,
    'CZ': CZ,
    'CY': CY,
    'CX': CX,
    'CCX': CCX,
    'CCZ': CCZ,
    'Rx': Rx,
    'Ry': Ry,
    'Rz': Rz,
    'CRx': CRx,
    'CRy': CRy,
    'CRz': CRz,
}


def to_circuital(diagram: Diagram) -> Diagram:
    """Takes a :py:class:`lambeq.quantum.Diagram`, returns
    a modified :py:class:`lambeq.quantum.Diagram` which
    is easier to convert to tket and other circuit simulators

    Parameters
    ----------
    diagram : :py:class:`~lambeq.backend.quantum.Diagram`
        The :py:class:`Circuits <lambeq.backend.quantum.Diagram>`
        to be converted to a tket circuit.

    The returned circuit diagram has all qubits at the top
    with layer depth equal to qubit index,
    followed by gates, and then post-selection
    measurements at the bottom.

    Returns
    -------
    :py:class:`lambeq.quantum.Diagram`
        Circuital diagram compatible with circuital_to_dict.
    """

    # bits and qubits are lists of register indices, at layer i we want
    # len(bits) == circuit[:i].cod.count(bit) and same for qubits
    # Necessary to ensure editing boxes is localized.
    circuit = fast_deepcopy(diagram)

    qubits: list[Layer] = []
    gates: list[Layer] = []
    measures: list[Layer] = []
    postselect: list[Layer] = []
    circuit = circuit.init_and_discard()

    #  Cleans up any '1' kets and converts them to X|0> -> |1>
    def remove_ketbra1(_, box: Box) -> Diagram | Box:
        ob_map: dict[Box, Diagram]
        ob_map = {Ket(1): Ket(0) >> X,  # type: ignore[dict-item]
                  Bra(1): X >> Bra(0)}  # type: ignore[dict-item]
        return ob_map.get(box, box)

    def add_qubit(qubits: list[Layer],
                  layer: Layer,
                  offset: int,
                  gates: list[Layer]) -> Tuple[list[Layer], list[Layer]]:
        """
            Adds a qubit to the qubit list.
            Shifts all the gates to accommodate new qubit.
            Assumes we only add one qubit at a time.
        """

        for qubit_layer in qubits:
            from_left = len(qubit_layer.left)
            if from_left >= offset:
                qubit_layer.left = qubit_layer.left.insert(layer.box.cod,
                                                           offset)

        layer.right = Ty()
        if offset > 0:
            layer.left = qubit ** offset
        else:
            layer.left = Ty()
        qubits.insert(offset, layer)

        return qubits, pull_qubit_through(offset, gates, dom=layer.box.cod)[0]

    def construct_measurements(last_layer: Layer,
                               post_selects: list[Layer]) -> list[Layer]:
        # Change to accommodate measurements before
        total_qubits = (len(last_layer.left)
                        + len(last_layer.box.cod)
                        + len(last_layer.right))

        bit_idx = list(range(total_qubits))
        q_idx = {}
        for layer in post_selects:
            # Find the qubit for each post selection
            q_idx[bit_idx[len(layer.left)]] = layer
            bit_idx.remove(bit_idx[len(layer.left)])

        # Inserting to the left is always trivial
        total_layer = ([*last_layer.left] + [*last_layer.box.cod]
                       + [*last_layer.right])

        new_postselects = []
        for key in sorted(q_idx.keys()):
            bits_left = sum([1 for i in bit_idx if i < key])
            q_idx[key].left = bit ** bits_left
            q_idx[key].right = q_idx[key].right._fromiter(total_layer[key+1:])
            new_postselects.append(q_idx[key])

        return new_postselects

    def pull_bit_through(q_idx: int,
                         gates: list[Layer],
                         layer: Layer) -> tuple[list[Layer], int]:
        """
            Inserts a qubit type into every layer at the appropriate index
            q_idx: idx - index of where to insert the gate.
        """

        for i, gate_layer in enumerate(gates):  # noqa: B007

            l_size = len(gate_layer.left)
            c_size = len(gate_layer.box.cod)
            d_size = len(gate_layer.box.dom)

            # Inserting to the left is always trivial
            if q_idx == l_size:
                break
            elif q_idx < l_size:
                gate_layer.left = gate_layer.left.replace(qubit, q_idx)
            # Qubit on right of gate. Handles 1 qubit gates by l(dom) = 1
            elif q_idx > l_size + len(gate_layer.box.dom) - 1:

                # Index relative to the 1st qubit on right
                r_rel = q_idx - (l_size + len(gate_layer.box.dom))

                # Insert on right. Update relative index from the left
                gate_layer.right = gate_layer.right.replace(qubit, r_rel)

                q_idx = r_rel + l_size + len(gate_layer.box.cod)

            elif c_size == d_size:
                # Initial control qubit box
                box = gate_layer.box
                box.dom = box.dom.replace(qubit, q_idx - l_size)
                box.cod = box.cod.replace(qubit, q_idx - l_size)

            else:
                raise NotImplementedError('Cannot pull bit through '
                                          f'box {gate_layer}')

        # Insert layer back into list and remove from the original
        layer = build_left_right(q_idx, layer, [gates[i-1]])
        gates.insert(i, layer)

        return gates, q_idx

    def pull_qubit_through(q_idx: int,
                           gates: list[Layer],
                           dom: Ty = qubit) -> tuple[list[Layer], int]:  # noqa: E501
        """
            Inserts a qubit type into every layer at the appropriate index
            q_idx: idx - index of where to insert the gate.
        """
        new_gates = []
        for gate_layer in gates:

            l_size = len(gate_layer.left)

            # Inserting to the left is always trivial
            if q_idx <= l_size:
                gate_layer.left = gate_layer.left.insert(dom, q_idx)
                new_gates.append(gate_layer)
            # Qubit on right of gate. Handles 1 qubit gates by l(dom) = 1
            elif q_idx > l_size + len(gate_layer.box.dom) - 1:

                # Index relative to the 1st qubit on right
                r_rel = q_idx - (l_size + len(gate_layer.box.dom))

                # Insert on right. Update relative index from the left
                gate_layer.right = gate_layer.right.insert(dom, r_rel)

                q_idx = r_rel + l_size + len(gate_layer.box.cod)
                new_gates.append(gate_layer)

            else:
                if isinstance(gate_layer.box, Controlled):
                    gate_qubits = [len(gate_layer.left) + j
                                   for j in range(len(gate_layer.box.dom))]

                    # Initial control qubit box
                    dists = [0]
                    curr_box: Box | Controlled = gate_layer.box
                    while isinstance(curr_box, Controlled):
                        # Compute relative index control qubits
                        dists.append(curr_box.distance + sum(dists))
                        curr_box = curr_box.controlled

                    prev_pos = -1 * min(dists) + gate_qubits[0]
                    curr_box = gate_layer.box

                    while isinstance(curr_box, Controlled):
                        curr_pos = prev_pos + curr_box.distance
                        if prev_pos < q_idx and q_idx <= curr_pos:
                            curr_box.distance = curr_box.distance + 1

                        elif q_idx <= prev_pos and q_idx > curr_pos:
                            curr_box.distance = curr_box.distance - 1

                        prev_pos = curr_pos
                        curr_box = curr_box.controlled

                    box = gate_layer.box

                    box.dom = box.dom.insert(dom, q_idx - l_size)
                    box.cod = box.cod.insert(dom, q_idx - l_size)
                    new_gates.append(gate_layer)

                if isinstance(gate_layer.box, Swap):

                    """
                    Replace single swap with a series of swaps
                    Swaps are 2 wide, so if a qubit is pulled through we
                    have to use the pulled qubit as an temp ancillary.
                    """
                    new_gates.append(Layer(gate_layer.left,
                                     Swap(qubit, qubit),
                                     dom >> gate_layer.right))
                    new_gates.append(Layer(dom >> gate_layer.left,
                                     Swap(qubit, qubit),
                                     gate_layer.right))
                    new_gates.append(Layer(gate_layer.left,
                                     Swap(qubit, qubit),
                                     dom >> gate_layer.right))

        return new_gates, q_idx

    def build_left_right(q_idx: int,
                         layer: Layer,
                         layers: list[Layer]) -> Layer:
        """
        We assume that the left and right are constructable
        from the last gate
        and the left position of the bra.
        (We type check at the end.)
        Rebuild left and right based on the last layer
        """
        if len(layers) == 0:
            return layer

        gate_layer = layers[-1]

        total_layer = ([*gate_layer.left] + [*gate_layer.box.cod]
                       + [*gate_layer.right])

        # Assumes you're only inserting one qubit at a time
        total_layer[q_idx] = layer.box.cod

        if q_idx == 0 or not total_layer[:q_idx]:
            layer.left = Ty()
        else:
            layer.left = layer.left._fromiter(total_layer[:q_idx])

        if q_idx == len(total_layer) - 1 or not total_layer[q_idx+1:]:
            layer.right = Ty()
        else:
            layer.right = layer.right._fromiter(total_layer[q_idx+1:])

        return layer

    circuit = Functor(target_category=quantum,
                      ob=lambda _, x: x,
                      ar=remove_ketbra1)(circuit)  # type: ignore [arg-type]

    layers = circuit.layers

    for i, layer in enumerate(layers):

        if isinstance(layer.box, Ket):
            qubits, gates = add_qubit(qubits,
                                      layer,
                                      len(layer.left),
                                      gates)

        elif isinstance(layer.box, (Bra, Discard)):

            q_idx = len(layer.left)
            layers[i+1:], q_idx = pull_qubit_through(q_idx, layers[i+1:])
            layer = build_left_right(q_idx, layer, layers[i+1 :])

            postselect.insert(0, layer)

        else:
            gates.append(layer)

    if gates:
        postselect = construct_measurements(gates[-1], postselect)

    # Rebuild the diagram
    diags = [Diagram(dom=layer.dom, cod=layer.cod, layers=[layer])  # type: ignore [arg-type] # noqa: E501
             for layer in qubits + gates + postselect + measures]

    layerD = diags[0]
    for diagram in diags[1:]:
        layerD = layerD >> diagram

    return layerD


@dataclass
class Gate:
    """Gate information for backend circuit construction.

    Parameters
    ----------
    name : str
        Arbitrary name / id
    gtype : str
        Type for backend conversion, e.g., 'Rx', 'X', etc.
    qubits : list[int]
        List of qubits the gate acts on.
    phase : Union[float, Symbol, None] = 0
        Phase parameter for gate.
    dagger : bool = False
        Whether to dagger the gate.
    control : Optional[list[int]] = None
        For control gates, list of all the control qubits.
    gate_q : Optional[int] = None
        For control gates, the gates being controlled.
    """
    name: str
    gtype: str
    qubits: list[int]
    phase: Union[float, Symbol, None] = 0
    dagger: bool = False
    control: Optional[list[int]] = None
    gate_q: Optional[int] = None

    @classmethod
    def from_box(cls, box: Box, offset: int, use_sympy: bool = False) -> Gate:
        """Constructs Gate for backend circuit construction
        from a Box.

        Parameters
        ----------
        box : Box
            Box to convert to a Gate.
        offset : int
            Qubit index on the leftmost part of the Gate.
        use_sympy : bool
            Use `sympy.Symbol` for the gate params, otherwise use
            `lambeq.backend.Symbol`.
        """
        name = box.name
        gtype = box.name.split('(')[0]
        qubits = [offset + j for j in range(len(box.dom))]
        phase = None
        dagger = False
        control = None
        gate_q = None

        if isinstance(box, Daggered):
            box = box.dagger()
            dagger = True
            gtype = box.name.split('(')[0]

        if isinstance(box, (Rx, Ry, Rz)):
            phase = box.phase
            if use_sympy and isinstance(box.phase, Symbol):
                # Tket uses sympy, lambeq uses custom symbol
                phase = box.phase.to_sympy()
        elif isinstance(box, Controlled):
            # reverse the distance order
            dists = []
            curr_box: Box | Controlled = box

            while isinstance(curr_box, Controlled):
                dists.append(curr_box.distance)
                curr_box = curr_box.controlled
            dists.reverse()

            # Index of the controlled qubit is the last entry in rel_idx
            rel_idx = [0]
            for dist in dists:
                if dist > 0:
                    # Add control to the left, offset by distance
                    rel_idx = [0] + [i + dist for i in rel_idx]
                else:
                    # Add control to the right, don't offset
                    right_most_idx = max(rel_idx)
                    rel_idx.insert(-1, right_most_idx - dist)

            i_qubits = [qubits[i] for i in rel_idx]

            qubits = i_qubits
            control = sorted(qubits[:-1])
            gate_q = qubits[-1]

            if gtype in ('CRx', 'CRz'):
                phase = box.phase
                if use_sympy and isinstance(box.phase, Symbol):
                    # Tket uses sympy, lambeq uses custom symbol
                    phase = box.phase.to_sympy()

        elif isinstance(box, Scalar):
            gtype = 'Scalar'
            phase = box.array

        return Gate(
            name,
            gtype,
            qubits,
            phase,
            dagger,
            control,
            gate_q
        )


@dataclass
class CircuitInfo:
    """Info for constructing circuits with backends.

    Parameters
    ----------
    total_qubits : int
        Total number of qubits in the circuit.
    gates : list[:py:class:`~lambeq.backend.quantum.Gate`]
        List containing gates, in topological ordering.
    bitmap: dict[int, int]
        Dictionary mapping qubit index to bit index for
        measurements, postselection, etc.
    postmap: dict[int, int]
        Dictionary mapping qubit index to post selection value.
    discards: list[int]
        List of discarded qubit indeces.
    """

    total_qubits: int
    gates: list[Gate]
    bitmap: dict[int, int]
    postmap: dict[int, int]
    discards: list[int]


def readoff_circuital(diagram: Diagram,
                      use_sympy: bool = False) -> CircuitInfo:
    """Takes a circuital :py:class:`lambeq.quantum.Diagram`, returns
    a :py:class:`~lambeq.backend.quantum.CircuitInfo` which
    is used by quantum backends to construct circuits. This checks if
    the diagram is circuital before converting.

    Parameters
    ----------
    diagram : :py:class:`~lambeq.backend.quantum.Diagram`
        The :py:class:`Circuits <lambeq.backend.quantum.Diagram>`
        to be converted to dictionary.
    use_sympy : bool, default=False
        Flag to use `sympy.Symbol` instead of `lambeq.backend.Symbol`
        for the parameters.

    Returns
    -------
    :py:class:`~lambeq.backend.quantum.CircuitInfo`
    """

    assert diagram.is_circuital

    layers = diagram.layers

    total_qubits = sum([1 for layer in layers if isinstance(layer.box, Ket)])
    available_qubits = list(range(total_qubits))

    gates: list[Gate] = []
    bitmap: dict = {}
    postmap: dict = {}
    discards: list[int] = []

    for layer in layers:
        if isinstance(layer.box, Ket):
            pass
        elif isinstance(layer.box, Measure):
            qi = available_qubits[layer.left.count(qubit)]
            available_qubits.remove(qi)
            bitmap[qi] = len(bitmap)
        elif isinstance(layer.box, Bra):
            qi = available_qubits[layer.left.count(qubit)]
            available_qubits.remove(qi)
            bitmap[qi] = len(bitmap)
            postmap[qi] = layer.box.bit
        elif isinstance(layer.box, Discard):
            qi = available_qubits[layer.left.count(qubit)]
            available_qubits.remove(qi)
            discards.append(qi)
        else:
            qi = len(layer.left)
            gates.append(Gate.from_box(layer.box, qi,
                                       use_sympy=use_sympy))

    return CircuitInfo(total_qubits,
                       gates,
                       bitmap,
                       postmap,
                       discards)
