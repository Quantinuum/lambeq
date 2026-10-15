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

from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

from lambeq.backend.grammar import Diagram, Ty


ROOT_INDEX = -1


class PregroupTreeNodeError(Exception):
    def __init__(self, sentence: str) -> None:
        super().__init__(sentence)
        self.sentence = sentence

    def __str__(self) -> str:
        return f'PregroupTreeNode failed to parse {self.sentence!r}.'


@dataclass
class PregroupTreeNode:
    """
    A node in a pregroup tree.

    A pregroup tree is a compact tree-like representation of a pregroup
    diagram. Each node in the tree represents a token in the sentence,
    annotated with the pregroup type of the outcome wire(s) expected
    from this word, e.g. `s` for a verb, `n` for an adjective, `n.r@s`
    for an adverb and so on. The root of the tree is the head word in
    the sentence (i.e a word with free wires, usually an `s`,
    that deliver the state of the sentence after the composition),
    and the branches of the tree represent cups identifying
    input wires (cups) to the parent node.

    Examples
    --------
    Consider the sentence "John gave Mary a flower", with the
    following pregroup diagram:

    .. code-block:: console

        John       gave      Mary    a    flower
        ────  ─────────────  ────  ─────  ──────
         n    n.r·s·n.l·n.l   n    n·n.l    n
         ╰────╯   │  │   ╰────╯    │  ╰─────╯
                  │  ╰─────────────╯

    The tree for this diagram becomes:

    .. code-block:: console

        gave_1 (s)
        ├ John_0 (n)
        ├ Mary_2 (n)
        └ a_3 (n)
            └ flower_4 (n)

    where the numbers after the underscore indicate the order of each
    word in the sentence. This representation is sufficient for
    reconstructing the original pregroup diagram, since the original
    types of the nodes can be recovered by following the parent-child
    relationships in the tree and adding the necessary type adjoints
    to accomodate the arguments in the type of the parent.

    Notes
    -----
    Since the original pregroup diagram can contain cycles, any nodes
    with more than one parents will be duplicated to allow a tree-like
    representation.

    """
    word: str
    ind: int
    typ: Ty
    typ_indxs: list[int] = field(default_factory=list)
    parent: Optional['PregroupTreeNode'] = None
    children: list['PregroupTreeNode'] = field(default_factory=list)

    _children_words: list[tuple[str, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Create a list of `(word, word index)` tuples which is used
        in several conversion functions."""
        self._children_words = [(child.word, child.ind)
                                for child in self.children]

        for child in self.children:
            child.parent = self

    def __repr__(self) -> str:
        """Return the string representation of the node."""
        return f'{self.word}_{self.ind} ({self.typ})'

    def __lt__(self, other: 'PregroupTreeNode') -> bool:
        return self.ind < other.ind

    def __gt__(self, other: 'PregroupTreeNode') -> bool:
        return self.ind > other.ind

    def __eq__(self, other: object) -> bool:
        """Check if these are the same instances, including
        the children and parent."""
        if not isinstance(other, PregroupTreeNode):
            return NotImplemented
        return (self.word == other.word
                and self.ind == other.ind
                and self.typ == other.typ
                and ((self.parent.ind if self.parent else None)
                     == (other.parent.ind if other.parent else None))
                and sorted(self.children) == sorted(other.children))

    def is_same_word(self, other: object) -> bool:
        """Check if these words are the same words in the sentence.

        This is a relaxed version of the `__eq__` function
        which doesn't check equality of the children - essentially,
        this just checks if `other` is the same token."""
        if not isinstance(other, PregroupTreeNode):
            return NotImplemented   # type: ignore[no-any-return]
        return (self.word == other.word
                and self.ind == other.ind)

    @cached_property
    def _tree_repr(self) -> str:
        """The string representation of the entire tree."""
        if not self.children:
            # Leaf
            return str(self)
        else:
            out = str(self)
            n_children = len(self.children)
            for i, child in enumerate(self.children):
                child_lines = child._tree_repr.split('\n')
                lines = []
                for j, l in enumerate(child_lines):
                    if j == 0:
                        prefix = '└' if i == n_children - 1 else '├'
                    else:
                        prefix = '│' if i != n_children - 1 else ' '
                    lines.append(f'\n{prefix} {l}')
                out += ''.join(lines)

            return out

    def draw(self) -> None:
        """Draw the tree."""
        print(self._tree_repr)

    @cached_property
    def height(self) -> int:
        """The height of the tree."""
        h = 1
        curr_nodes = [self]
        next_nodes = []
        while curr_nodes:
            for node in curr_nodes:
                next_nodes.extend(node.children)

            if len(next_nodes):
                h += 1
            curr_nodes = next_nodes
            next_nodes = []

        return h

    def get_types(self,
                  as_str: bool = True) -> list[list[str]] | list[list[Ty]]:
        """Return the types of each node in the tree.

        Parameters
        ----------
        as_str : bool
            Whether to return the types as str or as Ty

        Returns
        -------
        list[list[str] | list[Ty]]
            List of the string representations of the types or the types
            of each node indexed by the word order.

        """
        nodes_list = self.get_nodes()
        types_list = [
            [n.typ for n in nodes]
            for nodes in nodes_list
        ]

        if as_str:
            return [[str(t) for t in tys] for tys in types_list]

        return types_list

    def get_parents(self) -> list[list[int]]:
        """Return the indices of the parents of each node in the tree.

        Returns
        -------
        list[list[int]]
            List of the indices of the parents of each node,
            in the original sentence. The parent of the root node is
            assigned an index of [-1]. This is indexed by the word order.

        """
        nodes_list = self.get_nodes()
        parents_list = [
            [n.parent.ind if n.parent else -1 for n in nodes]
            for nodes in nodes_list
        ]

        return parents_list

    def get_words(self) -> list[str]:
        """Return the words for each node in the tree.

        Returns
        -------
        list of str
            List of the words corresponding to each node indexed by
            the word order.

        """
        nodes_list = self.get_nodes()
        words_list = [nodes[0].word for nodes in nodes_list]

        return words_list

    def get_word_indices(self) -> list[int]:
        """Return the indices of the word (in the original sentence)
        for each node in the tree.

        Returns
        -------
        list of int
            List of the indices of the words corresponding to each node
            indexed by the word order.

        Notes
        -----
        This is useful when the subtree doesn't form a span of
        the original sentence.

        """
        nodes_list = self.get_nodes()
        word_indices = [nodes[0].ind for nodes in nodes_list]

        return word_indices

    def _get_nodes_flat(self) -> list['PregroupTreeNode']:
        """Return the nodes of the tree following the word order
        in the sentence.

        Returns
        -------
        list of PregroupTreeNode
            List of the nodes corresponding to each word indexed by
            the word order but flattened.

        """
        nodes_list = [self]
        for child in self.children:
            nodes_list.extend(child._get_nodes_flat())

        return sorted(nodes_list, key=lambda n: n.ind)

    def get_nodes(self) -> list[list['PregroupTreeNode']]:
        """Collect nodes from `_get_nodes_flat` into a list for each index
        so we have the same length as the number of words.

        Returns
        -------
        list of list[PregroupTreeNode]
            List of the nodes corresponding to each word indexed by
            the word order. If cycles are present, multiple nodes
            will be assigned to the word.
        """
        flat_nodes_list = self._get_nodes_flat()
        nodes_list: list[list['PregroupTreeNode']] = [[] for _ in range(
            flat_nodes_list[-1].ind + 1
        )]

        # Merge nodes with the same `ind` into a list
        for node in flat_nodes_list:
            nodes_list[node.ind].append(node)

        nodes_list = [n for n in nodes_list if n]

        return nodes_list

    def get_root(self) -> 'PregroupTreeNode':
        """Return the root of the tree where this node belongs to."""
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def get_depth(self, node: Optional['PregroupTreeNode'] = None) -> int:
        """Return the depth of `self` node in the (sub)tree
        with `node` as its root

        Parameters
        ----------
        node : PregroupTreeNode, optional, default is `None`
            The node which we will treat as the root. If not given,
            will try to find the root of the tree where `self`
            belongs to and compute the depth from that node.

        Returns
        -------
        int
            The depth of this node in the tree. This is -1 if `self` is
            not in the tree rooted at `node`.
        """

        if node is None:
            node = self.get_root()

        depth = 0
        curr_node: Optional['PregroupTreeNode'] = self
        not_in_tree = False
        while curr_node != node:
            depth += 1
            if curr_node is None:
                not_in_tree = True
                break
            curr_node = curr_node.parent

        if not_in_tree:
            return -1
        return depth

    def merge(self) -> None:
        """
        If `self` has only one children of the same type, this merges
        the words into one token while preserving the type.
        The minimum index is taken as the index of the new node.
        This modifies the calling node.
        """

        if len(self.children) != 1:
            print('Cannot perform merge on node that '
                  + "doesn't have exactly one child.")
        else:
            child = self.children[0]
            if self.typ == child.typ and abs(self.ind - child.ind) == 1:
                # Perform merge
                if self.ind < child.ind:
                    self.word += f' {child.word}'
                else:
                    self.word = f'{child.word} {self.word}'

                self.ind = min(self.ind, child.ind)
                self.children = child.children

                # Modify parent of child
                child.parent = None
                for c in self.children:
                    c.parent = self
            else:
                print('Cannot perform merge when parent and child '
                      + "types don't match or tokens are not consecutive.")

    def remove_self_cycles(self) -> None:
        """Removes the children of this node that is the same token,
        i.e. self-cycles.

        This is used before breaking cycles.
        """

        new_children = []
        for c in self.children:
            if self.is_same_word(c):
                c.parent = None
            else:
                new_children.append(c)
        self.children = new_children
        for c in self.children:
            c.remove_self_cycles()

    def to_diagram(self, tokens: list[str]) -> Diagram | None:
        from lambeq.text2diagram.pregroup_tree_converter import tree2diagram

        diagram = None
        try:
            diagram = tree2diagram(self, tokens)
        except Exception as e:
            raise PregroupTreeNodeError(
                ' '.join(tokens)
            ) from e

        return diagram
