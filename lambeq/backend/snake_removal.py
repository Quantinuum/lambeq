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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Snake removal
=============
This module contains a function for removing snakes from diagrams. This
work is based on DisCoPy (https://discopy.org/) which is released under
the BSD 3-Clause "New" or "Revised" License.

"""

from __future__ import annotations

from collections.abc import Iterator

from lambeq.backend.grammar import Box, Cap, Cup, Diagram


class InterchangerError(Exception):
    """ This is raised when we try to interchange conected boxes. """
    def __init__(self, box0: Box, box1: Box) -> None:
        super().__init__(box0, box1)
        self.box0 = box0
        self.box1 = box1

    def __str__(self) -> str:
        return f'Boxes {self.box0} and {self.box1} do not commute.'


def snake_removal(diagram: Diagram, left: bool = False) -> Iterator[Diagram]:
    """
    Returns a generator which yields normalization steps.

    Parameters
    ----------
    left : bool, optional
        Whether to apply left interchangers.

    Yields
    ------
    diagram : :class:`Diagram`
        Rewrite steps.

    Examples
    --------
    >>> from lambeq.backend.grammar import Ty, Box, Cup, Cap, Id
    >>> n, s = Ty('n'), Ty('s')
    >>> cup, cap = Cup(n, n.r), Cap(n.r, n)
    >>> f = Box('f', n, n)
    >>> g = Box('g', s @ n, n)
    >>> h = Box('h', n, n @ s)
    >>> diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h
    >>> for d in snake_removal(diagram):
    ...     print(d)  # doctest: +ELLIPSIS
    |Ty... >> |Ty() @ [CUP; Ty(n) @ Ty(n).r -> Ty()] @ Ty(n)| >>...
    |Ty... >> |Ty(n) @ [CAP; Ty() -> Ty(n).r @ Ty(n)] @ Ty()| >> \
|Ty() @ [CUP; Ty(n) @ Ty(n).r -> Ty()] @ Ty(n)| >>...
    |Ty() @ [g; Ty(s) @ Ty(n) -> Ty(n)] @ Ty()| >> \
|Ty() @ [f†; Ty(n) -> Ty(n)] @ Ty()| >> \
|Ty() @ [f; Ty(n) -> Ty(n)] @ Ty()| >> \
|Ty() @ [h; Ty(n) -> Ty(n) @ Ty(s)] @ Ty()|
    """
    def follow_wire(diagram: Diagram,
                    i: int,
                    j: int) -> tuple[int, int,
                                     tuple[list[int], list[int]]]:
        """
        Given a diagram, the index of a box i and the offset j of an
        output wire, returns (i, j, obstructions) where:
        - i is the index of the box which takes this wire as input,
        or len(diagram) if it is connected to the bottom boundary.
        - j is the offset of the wire at its bottom end.
        - obstructions is a pair of lists of indices for the boxes
        on the left and right of the wire we followed.
        """
        left_obstruction = []  # type: list[int]
        right_obstruction = []  # type: list[int]
        while i < len(diagram) - 1:
            i += 1
            box, off = diagram.boxes[i], diagram.offsets[i]
            if off <= j < off + len(box.dom):
                return i, j, (left_obstruction, right_obstruction)
            if off <= j:
                j += len(box.cod) - len(box.dom)
                left_obstruction.append(i)
            else:
                right_obstruction.append(i)
        return len(diagram), j, (left_obstruction, right_obstruction)

    def find_snake(diagram: Diagram) -> None | tuple[int, int,
                                                     tuple[list[int],
                                                           list[int]],
                                                     bool]:
        """
        Given a diagram, returns (cup, cap, obstructions,
        left_snake) if there is a yankable pair, otherwise returns
        None.
        """
        for cap in range(len(diagram)):
            if not isinstance(diagram.boxes[cap], Cap):
                continue
            for left_snake, wire in [(True, diagram.offsets[cap]),
                                     (False, diagram.offsets[cap] + 1)]:
                cup, wire, obstructions = follow_wire(diagram, cap, wire)
                not_yankable = (cup == len(diagram)
                                or not isinstance(diagram.boxes[cup], Cup)
                                or (left_snake
                                    and diagram.offsets[cup] + 1 != wire)
                                or (not left_snake
                                    and diagram.offsets[cup] != wire))
                if not_yankable:
                    continue
                return cup, cap, obstructions, left_snake
        return None

    def unsnake(diagram: Diagram,
                cup: int,
                cap: int,
                obstructions: tuple[list[int], list[int]],
                left_snake: bool = False) -> Iterator[Diagram]:
        """
        Given a diagram and the indices for a cup and cap pair
        and a pair of lists of obstructions on the left and right,
        returns a new diagram with the snake removed.

        A left snake is one of the form Id @ Cap >> Cup @ Id.
        A right snake is one of the form Cap @ Id >> Id @ Cup.
        """
        left_obstruction, right_obstruction = obstructions
        if left_snake:
            for box in left_obstruction:
                diagram = interchange(diagram, box, cap)
                yield diagram
                for i, right_box in enumerate(right_obstruction):
                    if right_box < box:
                        right_obstruction[i] += 1
                cap += 1
            for box in right_obstruction[::-1]:
                diagram = interchange(diagram, box, cup)
                yield diagram
                cup -= 1
        else:
            for box in left_obstruction[::-1]:
                diagram = interchange(diagram, box, cup)
                yield diagram
                for i, right_box in enumerate(right_obstruction):
                    if right_box > box:
                        right_obstruction[i] -= 1
                cup -= 1
            for box in right_obstruction:
                diagram = interchange(diagram, box, cap)
                yield diagram
                cap += 1
        layers = diagram.layers[:cap] + diagram.layers[cup + 1:]
        yield diagram.category.Diagram(diagram.dom, diagram.cod, layers)

    while True:
        yankable = find_snake(diagram)
        if yankable is None:
            break
        for _diagram in unsnake(diagram, *yankable):
            yield _diagram
            diagram = _diagram


def interchange(diagram: Diagram,
                i: int,
                j: int,
                left: bool = False) -> Diagram:
    """
    Returns a new diagram with boxes i and j interchanged.

    Gets called recursively whenever :code:`i < j + 1 or j < i - 1`.

    Parameters
    ----------
    diagram : :class:`Diagram`
        The diagram to interchange boxes in.
    i : int
        Index of the box to interchange.
    j : int
        Index of the new position for the box.
    left : bool, optional
        Whether to apply left interchangers.

    Notes
    -----
    By default, we apply only right exchange moves::

        top >> Id(left @ box1.dom @ mid) @ box0 @ Id(right)
            >> Id(left) @ box1 @ Id(mid @ box0.cod @ right)
            >> bottom

    gets rewritten to::

        top >> Id(left) @ box1 @ Id(mid @ box0.dom @ right)
            >> Id(left @ box1.cod @ mid) @ box0 @ Id(right)
            >> bottom
    """
    if not 0 <= i < len(diagram) or not 0 <= j < len(diagram):
        raise IndexError
    if i == j:
        return diagram
    if j < i - 1:
        result = diagram
        for k in range(i - j):
            result = interchange(result, i - k, i - k - 1, left=left)
        return result
    if j > i + 1:
        result = diagram
        for k in range(j - i):
            result = interchange(result, i + k, i + k + 1, left=left)
        return result
    if j < i:
        i, j = j, i
    off0, off1 = diagram.offsets[i], diagram.offsets[j]
    left0, box0, right0 = diagram.layers[i].unpack()
    left1, box1, right1 = diagram.layers[j].unpack()
    # By default, we check if box0 is to the right first,
    # then to the left.
    if left and off1 >= off0 + len(box0.cod):  # box0 left of box1
        middle = left1[len(left0 @ box0.cod):]
        layer0 = diagram.category.Layer(left0, box0,
                                        middle @ box1.cod @ right1)
        layer1 = diagram.category.Layer(left0 @ box0.dom @ middle, box1,
                                        right1)
    elif off0 >= off1 + len(box1.dom):  # box0 right of box1
        middle = left0[len(left1 @ box1.dom):]
        layer0 = diagram.category.Layer(left1 @ box1.cod @ middle, box0,
                                        right0)
        layer1 = diagram.category.Layer(left1, box1,
                                        middle @ box0.dom @ right0)
    elif off1 >= off0 + len(box0.cod):  # box0 left of box1
        middle = left1[len(left0 @ box0.cod):]
        layer0 = diagram.category.Layer(left0, box0,
                                        middle @ box1.cod @ right1)
        layer1 = diagram.category.Layer(left0 @ box0.dom @ middle, box1,
                                        right1)
    else:
        raise InterchangerError(box0, box1)
    layers = diagram.layers[:i] + [layer1, layer0] + diagram.layers[i + 2:]
    return diagram.category.Diagram(diagram.dom, diagram.cod, layers=layers)


def normalize(diagram: Diagram, left: bool = False) -> Iterator[Diagram]:
    """
    Implements normalization of diagrams,
    see arXiv:1804.07832.

    Parameters
    ----------
    diagram : :class:`Diagram`
        The diagram to normalize.
    left : bool, optional
        Passed to :func:`interchange`.

    Yields
    ------
    diagram : :class:`Diagram`
        Rewrite steps.

    Examples
    --------
    >>> from lambeq.backend.grammar import Ty, Box
    >>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
    >>> gen = normalize(s0 @ s1)
    >>> for _ in range(3): print(next(gen))
    |Ty() @ [s1; Ty() -> Ty()] @ Ty()| >> \
|Ty() @ [s0; Ty() -> Ty()] @ Ty()|
    |Ty() @ [s0; Ty() -> Ty()] @ Ty()| >> \
|Ty() @ [s1; Ty() -> Ty()] @ Ty()|
    |Ty() @ [s1; Ty() -> Ty()] @ Ty()| >> \
|Ty() @ [s0; Ty() -> Ty()] @ Ty()|
    """
    no_more_moves = False
    while not no_more_moves:
        no_more_moves = True
        for i in range(len(diagram) - 1):
            box0, box1 = diagram.boxes[i], diagram.boxes[i + 1]
            off0, off1 = diagram.offsets[i], diagram.offsets[i + 1]
            if ((left and off1 >= off0 + len(box0.cod))
                    or (not left and off0 >= off1 + len(box1.dom))):
                diagram = interchange(diagram, i, i + 1, left=left)
                yield diagram
                no_more_moves = False
