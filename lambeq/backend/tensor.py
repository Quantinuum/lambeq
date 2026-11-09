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
Tensor category
===============
Lambeq's internal representation of the tensor category. This work is
based on DisCoPy (https://discopy.org/) which is released under the
BSD 3-Clause 'New' or 'Revised' License.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import cached_property
import math
from typing import Mapping

import numpy as np
import tensornetwork as tn
from typing_extensions import Any, Self

from lambeq.backend import grammar
from lambeq.backend.numerical_backend import backend, get_backend
from lambeq.backend.symbol import lambdify, Symbol


tensor = grammar.Category('tensor')


@tensor('Ty')
@dataclass(init=False)
class Dim(grammar.Ty):
    """Dimension in the tensor category.

    Attributes
    ----------
    dim : tuple of int
        Tuple of dimensions represented by the object.
    product: int
        Product of contained dimensions.

    """
    objects: list[Self]  # type: ignore[misc]

    def __init__(self,
                 *dim: int,
                 objects: list[Self] | None = None) -> None:
        """Initialise a Dim type.

        Parameters
        ----------
        dim : list[int]
            List of dimensions to initialise.
        objects: list[Self] or None, default None
            List of `Dim`s, to prepare a non-atomic `Dim` object.

        """

        if objects:
            assert not len(dim)
            super().__init__(objects=objects)  # type: ignore[arg-type]
        else:
            dims: list[int] = list(filter(lambda x: x > 1, dim))

            if not len(dims):
                super().__init__()

            if len(dims) == 1:
                super().__init__(str(dims[0]))
            else:
                super().__init__(objects=[Dim(d) for d in dims])

    @property
    def dim(self) -> tuple[int, ...]:
        if self.is_atomic:
            return (int(self.name), )  # type: ignore[arg-type]

        return tuple(dim for subdim in self.objects for dim in subdim.dim)

    def rotate(self, z: int) -> Self:
        if self.is_atomic:
            return self

        return super().rotate(z)

    def _repr_rec(self) -> str:
        if self.is_empty:
            return '1'
        elif self.is_atomic:
            return self.name  # type: ignore[return-value]
        else:
            return ', '.join(d._repr_rec() for d in self.objects)

    @property
    def product(self) -> int:
        return math.prod(self.dim)

    def __repr__(self) -> str:
        return f'Dim({self._repr_rec()})'

    def __hash__(self) -> int:
        return hash(repr(self))


@dataclass(init=False)
@tensor
class Box(grammar.Box):
    """Box (tensor) in the the tensor category.

    Attributes
    ----------
    data : np.array or float or None
        Data used to represent the `array` attribute. Typically either
        the array itself, or a symbolic array.
    array : np.array or float
        Tensor which the box represents.
    free_symbols : set of lambeq.backend.symbol.Symbol
        In case of a symbolic tensor, set of symbols in the box's data.

    """
    name: str
    dom: Dim
    cod: Dim
    data: float | np.ndarray | None
    z: int

    def __init__(self,
                 name: str,
                 dom: Dim,
                 cod: Dim,
                 data: float | np.ndarray | None = None,
                 z: int = 0):
        """Initialise a `tensor.Box` type.

        Parameters
        ----------
        name : str
            Name for the box.
        dom : Dim
            Dimension of the box's domain.
        cod : Dim
            Dimension of the box's codomain.
        data : float or np.ndarray, optional
            The concrete tensor the box represents.
        z : int, optional
            Winding number of the box, indicating conjugation. Starts at
            0 if not provided.

        """
        self.name = name
        self.dom = dom
        self.cod = cod
        self.data = data
        self.z = z

    def __eq__(self, other):
        return (self.name == other.name
                and self.dom == other.dom
                and self.cod == other.cod
                and np.equal(self.data, other.data).all())

    @property
    def array(self):
        if self.data is not None:
            if self.z % 2:
                ret_arr = self._conjugate_array()
            else:
                ret_arr = get_backend().array(self.data)

            return ret_arr.reshape(self.dom.dim + self.cod.dim)

    def _adjoint_array(self):
        """Returns the adjoint of the box's data"""

        arr = self.array

        source = range(len(self.dom @ self.cod))
        target = [i + len(self.cod) if i < len(self.dom) else
                  i - len(self.dom) for i in range(len(self.dom @ self.cod))]
        with backend() as np:
            return np.conjugate(np.moveaxis(arr, source, target))

    def _conjugate_array(self):
        """Returns the diagrammtic conjugate of the box's data"""

        dom, cod = self.dom, self.cod
        with backend() as np:
            array = np.moveaxis(self.data,
                                range(len(dom @ cod)),
                                [len(dom) - i - 1
                                    for i in range(len(dom @ cod))])
            return np.conjugate(array)

    def dagger(self):
        """Get the dagger (adjoint) of the box.

        Returns
        -------
        Box
            Dagger of the box.

        """

        return Daggered(self)

    def rotate(self, z: int):
        """Get the result of conjugating the box `z` times.

        Parameters
        ----------
        z : int
            Winding count. The number of conjugations to apply to the box.

        Returns
        -------
        Box
            The box conjugated z times.

        """

        return replace(self,
                       dom=self.dom.rotate(z),
                       cod=self.cod.rotate(z),
                       z=(self.z + z) % 2)

    @cached_property
    def free_symbols(self) -> set[Symbol]:
        def recursive_free_symbols(data) -> set[Symbol]:
            if isinstance(data, Mapping):
                data = data.values()
            if isinstance(data, Iterable):
                if (
                    not hasattr(data, 'shape') or data.shape != ()
                ) and not (
                    not isinstance(data, np.ndarray)
                    or np.issubdtype(data.dtype, np.number)
                ):
                    return set().union(*map(recursive_free_symbols, data))
            # Remove scale before adding to set
            return {data.unscaled} if isinstance(data, Symbol) else set()
        return recursive_free_symbols(self.data)

    def lambdify(self, *symbols: 'Symbol', **kwargs) -> Callable:
        """Get a lambdified version of a symbolic box.

        Returns a function which when provided appropriate parameters,
        initialises a concrete box.

        Parameters
        ----------
        symbols : list of Symbols
            List of symbols in the box in the order in which their
            assigned values will appear in the concretisation call.
        kwargs:
            Additional parameters to pass to `lambdify`.

        Returns
        -------
        Callable[..., Box]:
            A lambda function which when invoked with appropriate
            parameters, returns a concrete version of the box.

        """

        if not any(x in self.free_symbols for x in symbols):
            return lambda *xs: self

        return lambda *xs: type(self)(
            self.name, self.dom, self.cod,
            lambdify(symbols, self.data)(*xs))

    def __repr__(self) -> str:
        return (f'[{self.name}{".l"*(-self.z)}{".r"*self.z}; '
                f'{repr(self.dom)} -> {repr(self.cod)}]')

    def __hash__(self) -> int:
        return hash(repr(self))


@dataclass
@tensor
class Layer(grammar.Layer):
    """Layer in the tensor category."""

    left: Dim
    box: Box
    right: Dim


@dataclass
@tensor
class Diagram(grammar.Diagram):
    """Diagram in the tensor category."""

    dom: Dim
    cod: Dim
    layers: list[Layer]  # type: ignore[assignment]

    def lambdify(self, *symbols, **kwargs):

        lambdified_layers = [(l_,
                              bx.lambdify(*symbols, **kwargs),
                              r_) for l_, bx, r_ in self.layers]

        def lambda_diagram(*xs):
            return type(self)(
                self.dom,
                self.cod,
                [self.category.Layer(l_,
                                     bx_lambda(*xs),
                                     r_) for (l_,
                                              bx_lambda,
                                              r_) in lambdified_layers])

        return lambda_diagram

    @cached_property
    def free_symbols(self) -> set[Symbol]:
        return set().union(*(box.free_symbols for box in self.boxes))

    def eval(self, contractor=tn.contractors.auto, dtype: type | None = None):
        """Evaluate the tensor diagram.

        Parameters
        ----------
        contractor : tn contractor
            `tensornetwork` contractor for chosen contraction algorithm.
        dtype : type, optional
            Data type of the resulting array. Defaults to `np.float32`.

        Returns
        -------
        numpy.ndarray
            n-dimension array representing the contracted tensor.

        """

        return contractor(*self.to_tn(dtype=dtype)).tensor

    def to_tn(self, dtype: type | None = None):
        """Convert the diagram to a `tensornetwork` TN.

        Parameters
        ----------
        dtype : type, optional
            Data type of the resulting array. Defaults to `np.float32`.

        Returns
        -------
        tuple[list[tn.Node], list[tn.Edge]]
            `tensornetwork` representation of the diagram. An edge
            object is returned for each dangling edge in the network.

        """

        if dtype is None:
            dtype = np.float32

        backend = get_backend().name

        nodes = [tn.CopyNode(2, dim, dtype=dtype,
                             backend=backend) for dim in self.dom.dim]
        inputs = [node[0] for node in nodes]
        scan = [node[1] for node in nodes]

        diag = self.category.Diagram.id(self.dom)

        for layer in self.layers:
            left, box, right = layer.unpack()
            subdiag = box

            if hasattr(box, 'decompose'):
                subdiag = box.decompose()

            diag >>= (self.category.Diagram.id(left)
                      @ subdiag
                      @ self.category.Diagram.id(right))

        for lyr in diag.layers:
            l, box, r = lyr.unpack()

            if isinstance(box, Swap):
                scan[len(l)], scan[len(l) + 1] = scan[len(l) + 1], scan[len(l)]
            elif isinstance(box, Cup):
                tn.connect(scan[len(l)], scan[len(l) + 1])
                del scan[len(l): len(l) + 2]
            else:
                if isinstance(box, Spider):
                    node = tn.CopyNode(box.n_legs_in + box.n_legs_out,
                                       box.type.product, dtype=dtype,
                                       backend=backend)
                else:
                    node = tn.Node(box.array,
                                   str(box.name),
                                   backend=backend)

                nodes.append(node)

                for i in range(len(box.dom)):
                    tn.connect(scan[len(l) + i], node[i])

                scan = (scan[:len(l)]
                        + node[len(box.dom):]
                        + scan[len(l) + len(box.dom):])

        # nodes, input_edge_order, output_edge_order
        return nodes, inputs + scan

    __hash__ = grammar.Diagram.__hash__


@Diagram.register_special_box('cap')
class Cap(grammar.Cap, Box):
    """A Cap in the tensor category.

    Its array correlates every matching basis state for arbitrary atomic
    leg dimension.
    """

    left: Dim
    right: Dim
    dom: Dim
    cod: Dim
    z: int = 0
    is_reversed: bool = False

    def __init__(self, left: Dim, right: Dim, is_reversed: bool = False):
        """Initialise a tensor Cap.

        Parameters
        ----------
        left : Dim
            Dimension (type) of the left leg of the cap. Must be the
            conjugate of `right`.
        right : Dim
            Dimension (type) of the right leg of the cap. Must be the
            conjugate of `left`.
        is_reversed : bool, default False
            Ignored parameter, since left and right conjugates are
            equivalent in the tensor category. Necessary to inherit
            from `grammar.Cap` appropriately.

        """

        super().__init__(left, right)

        self.data = np.eye(left.product).ravel()

    __hash__ = Box.__hash__
    __repr__ = Box.__repr__


@Diagram.register_special_box('cup')
class Cup(grammar.Cup, Box):
    """A Cup in the tensor category.

    Its array correlates every matching basis state for arbitrary atomic
    leg dimension.
    """

    left: Dim
    right: Dim
    name: str
    dom: Dim
    cod: Dim
    z: int = 0
    is_reversed: bool = False

    def __init__(self, left: Dim, right: Dim, is_reversed: bool = False):
        """Initialise a tensor Cup.

        Parameters
        ----------
        left : Dim
            Dimension (type) of the left leg of the cup. Must be the
            conjugate of `right`.
        right : Dim
            Dimension (type) of the right leg of the cup. Must be the
            conjugate of `left`.
        is_reversed : bool, default False
            Ignored parameter, since left and right conjugates are
            equivalent in the tensor category. Necessary to inherit
            from `grammar.Cup` appropriately.

        """

        super().__init__(left, right)

        self.data = np.eye(left.product).ravel()

    __hash__ = Box.__hash__
    __repr__ = Box.__repr__


@Diagram.register_special_box('swap')
class Swap(grammar.Swap, Box):
    """A Swap in the tensor category."""

    left: Dim
    right: Dim
    name: str
    dom: Dim
    cod: Dim
    z: int = 0

    def __init__(self, left: Dim, right: Dim):
        """Initialise a tensor Swap.

        Parameters
        ----------
        left : Dim
            Dimension (type) of the left input of the swap.
        right : Dim
            Dimension (type) of the right input of the swap.

        """

        grammar.Swap.__init__(self, left, right)
        Box.__init__(self, 'SWAP', left @ right, right @ left)

    def dagger(self):
        return type(self)(self.right, self.left)

    __hash__ = Box.__hash__
    __repr__ = Box.__repr__


@Diagram.register_special_box('spider')
class Spider(grammar.Spider, Box):
    """A Spider in the tensor category.

    Concretely represented by a copy node.
    """

    type: Dim
    n_legs_in: int
    n_legs_out: int
    name: str
    dom: Dim
    cod: Dim
    z: int = 0

    def __init__(self, type: Dim, n_legs_in: int, n_legs_out: int):
        """Initialise a tensor Spider.

        Parameters
        ----------
        type : Dim
            Dimension (type) of each leg of the spider.
        n_legs_in : int
            Number of input legs of the spider.
        n_legs_out : int
            Number of input legs of the spider.

        """
        Box.__init__(self, 'SPIDER', type ** n_legs_in, type ** n_legs_out)
        grammar.Spider.__init__(self, type, n_legs_in, n_legs_out)

    def dagger(self) -> Self:
        return type(self)(self.type, self.n_legs_out, self.n_legs_in)

    __hash__ = Box.__hash__
    __repr__ = Box.__repr__


Id = Diagram.id


@dataclass
class Daggered(grammar.Daggered, Box):
    """A daggered box.

    Attributes
    ----------
    box : Box
        The box to be daggered.

    """

    box: Box
    name: str = field(init=False)
    dom: Dim = field(init=False)
    cod: Dim = field(init=False)
    data: float | np.ndarray | None = field(default=None, init=False)
    z: int = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.box.name + '†'
        self.dom = self.box.cod
        self.cod = self.box.dom
        self.data = self.box.data
        self.z = self.box.z

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == 'data':
            self.box.data = __value
        return super().__setattr__(__name, __value)

    def lambdify(self, *symbols: 'Symbol', **kwargs) -> Callable:
        b_fn = self.box.lambdify(*symbols, **kwargs)
        return lambda *xs: b_fn(*xs).dagger()

    @property
    def array(self):
        return self.box._adjoint_array()

    __hash__ = Box.__hash__
    __repr__ = Box.__repr__
    __eq__ = grammar.Daggered.__eq__
