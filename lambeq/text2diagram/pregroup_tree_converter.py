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

from typing import Any, Optional

import networkx as nx

from lambeq.backend.grammar import Cap, Cup, Diagram, Swap, Ty, Word
from lambeq.core.utils import fast_deepcopy
from lambeq.text2diagram.pregroup_tree import (
    PregroupTreeNode, ROOT_INDEX,
)


WordComponentsDictsType = list[dict[int, list[int]]]
WordComponentsType = list[list[tuple[int, Ty]]]


def diagram2tree(diagram: Diagram,
                 break_cycles: bool = False) -> PregroupTreeNode:
    """Create a pregroup tree from a pregroup diagram.

    Parameters
    ----------
    diagram : `grammar.Diagram`
        The diagram that will be converted into a pregroup tree.
    break_cycles : bool, default: False
        Flag that indicates whether cycles will be broken in
        the output pregroup tree. This is done by removing
        duplicate nodes, keeping the copy of the node that is closest
        to its parent in the original sentence.

    Returns
    -------
    `pregroup_tree.PregroupTreeNode`
        The root node of the pregroup tree representation of
        the diagram.
    """

    word_typ_lists: dict[Optional[tuple[str, int]], list[int]] = {}
    words_by_typ_index: list[tuple[Optional[Word], Optional[int]]] = []
    types: list[Ty] = []
    types_ind: list[int] = []
    boxes_absolute: list[tuple[str, int, int]] = []
    words: list[tuple[Word, int]] = []

    curr_typ_ind = 0
    for li, l in enumerate(diagram):
        if isinstance(l.box, Word):
            words_by_typ_index += [(l.box, li)] * len(l.box.cod)
            types.extend(l.box.cod)
            words.append((l.box, li))
            curr_typ_ind += len(l.box.cod)
        elif isinstance(l.box, (Cup, Swap, Cap)):
            if not types_ind:
                types_ind = list(range(len(types)))

            left_len = len(l.left)
            right_len = len(l.right)
            left_cod = types_ind[:left_len]

            box_ends = []
            if right_len:
                right_cod = types_ind[-right_len:]
                box_ends = types_ind[left_len:-right_len]
            else:
                right_cod = []
                box_ends = types_ind[left_len:]

            if isinstance(l.box, Cap):
                if left_len < len(types_ind):
                    box_left_end = types_ind[left_len]
                else:
                    box_left_end = types_ind[left_len - 1] + 1
                words_by_typ_index = (words_by_typ_index[:box_left_end]
                                      + [(None, None), (None, None)]
                                      + words_by_typ_index[box_left_end:])
                types = (types[:box_left_end]
                         + [l.box.left, l.box.right]
                         + types[box_left_end:])

                box_ends = [box_left_end, box_left_end + 1]
                new_boxes_absolute = []
                for box_abs in boxes_absolute:
                    if box_abs[1] >= box_left_end:
                        new_boxes_absolute.append((
                            box_abs[0],
                            box_abs[1] + 2,
                            box_abs[2] + 2,
                        ))
                        try:
                            left_cod[left_cod.index(box_abs[1])] += 2
                            left_cod[left_cod.index(box_abs[2])] += 2
                        except ValueError:
                            pass
                    elif box_abs[2] >= box_left_end:
                        new_boxes_absolute.append((
                            box_abs[0],
                            box_abs[1],
                            box_abs[2] + 2,
                        ))
                        try:
                            left_cod[left_cod.index(box_abs[2])] += 2
                        except ValueError:
                            pass
                    else:
                        new_boxes_absolute.append(box_abs)
                boxes_absolute = new_boxes_absolute

                # update node type indices
                right_cod = [ind + 2 for ind in right_cod]

            assert len(box_ends) == 2

            boxes_absolute.append(
                (type(l.box).__name__, box_ends[0], box_ends[1])
            )

            box_cod = []
            if isinstance(l.box, Swap):
                box_cod = [box_ends[1], box_ends[0]]
            elif isinstance(l.box, Cap):
                box_cod = box_ends
            types_ind = left_cod + box_cod + right_cod
        else:
            raise Exception(f'Unknown box type: {type(l.box)}')

        assert len(types) == len(words_by_typ_index)

    for i, (word, word_ind) in enumerate(words_by_typ_index):
        k: Optional[tuple[str, int]] = ((word.name, word_ind)
                                        if (word is not None
                                            and word_ind is not None)
                                        else None)
        typ_list = word_typ_lists.get(k, [])
        typ_list.append(i)
        word_typ_lists[k] = typ_list

    assert len(diagram.boxes) == len(words) + len(boxes_absolute)

    if not len(boxes_absolute):
        assert len(words) == 1
        # No morphisms case - just return the node representation
        # of the word
        root_word = words[0]
        root_node = PregroupTreeNode(
            root_word[0].name,
            typ=root_word[0].cod,
            ind=root_word[1],
        )
        return root_node

    # Remove caps
    all_caps = [box for box in boxes_absolute if box[0] == 'Cap']
    for cap in all_caps:
        # find cups connected to this cap
        cups = [box for box in boxes_absolute
                if (box[0] == 'Cup'
                    and (cap[1] in box[1:] or cap[2] in box[1:]))]

        if len(cups) == 2:
            # three cases:
            if max(cups[0][1], cups[1][1]) < cap[1]:
                # cap is to the right of cups
                new_cup_ends = [cups[0][1], cups[1][1]]
            elif min(cups[0][2], cups[1][2]) > cap[2]:
                # cap is to the left of cups
                new_cup_ends = [cups[0][2], cups[1][2]]
            else:
                # cap is in the middle of the cups
                new_cup_ends = [
                    min(cups[0][1], cups[1][1]),
                    max(cups[0][2], cups[1][2]),
                ]

            boxes_absolute.append((
                'Cup', min(new_cup_ends), max(new_cup_ends)
            ))
        elif len(cups) == 1:
            # this cup-cap can be yanked to the free index
            snake_inds = [
                cups[0][1], cups[0][2],
                cap[1], cap[2]
            ]
            assert types_ind[0] in snake_inds
            if types_ind[0] == min(snake_inds):
                types_ind = [max(snake_inds)]
            elif types_ind[0] == max(snake_inds):
                types_ind = [min(snake_inds)]
            else:
                raise Exception('Impossible case when removing caps.')

        for cup in cups:
            boxes_absolute.remove(cup)
        boxes_absolute.remove(cap)

    # Construct the tree from the above data
    root_node = _construct_tree(
        types,
        types_ind,
        words_by_typ_index,
        word_typ_lists,
        boxes_absolute,
    )

    if break_cycles:
        remove_cycles(root_node)

    return root_node


def remove_cycles(root: PregroupTreeNode) -> None:
    """Break cycles by removing duplicate nodes.

    Duplicate nodes are always leaves of the tree and we remove
    the shallowest duplicate of the node.

    Parameters
    ----------
    root : `pregroup_tree.PregroupTreeNode`
        The root node of the tree we're interested in.

    Returns
    -------
    `pregroup_tree.PregroupTreeNode`
        The root node of the new tree without the duplicate nodes.
    """
    root_word_idx = root.ind
    nodes = root.get_nodes()
    assert len(nodes[root_word_idx]) == 1

    root_node = nodes[root_word_idx][0]

    # Remove nodes that cycles to itself
    # (see https://github.com/quantinuum/lambeq/issues/180)
    root_node.remove_self_cycles()

    for _, nodes_for_idx in enumerate(nodes):
        if len(nodes_for_idx) > 1:
            # Retain the deepest copy of the node
            idx_to_retain = 0
            depth = nodes_for_idx[idx_to_retain].get_depth(root_node)

            for j in range(1, len(nodes_for_idx)):
                copy_depth = nodes_for_idx[j].get_depth(root_node)
                if copy_depth > depth:
                    depth = copy_depth
                    idx_to_retain = j

            for j, node in enumerate(nodes_for_idx):
                if j != idx_to_retain:
                    if node.parent:
                        # Remove parent
                        node.parent.children.remove(node)
                        node.parent = None


def _construct_tree(
    types: list[Ty],
    free_strings: list[int],
    words_by_typ_index: list[tuple[Optional[Word], Optional[int]]],
    word_typ_lists: dict[Optional[tuple[str, int]], list[int]],
    boxes_absolute: list[tuple[str, int, int]],
) -> PregroupTreeNode:
    """Helper function for creating the pregroup tree from the passed
    data.

    Parameters
    ----------
    types: list of `Ty`
        The types from the codomain of the words layer as they appear
        in that layer.
    free_strings: list of `int`
        The indices of the codomain wires of the entire diagram,
        following the indexing in `types`.
    words_by_typ_index: list of `(Optional[Word], Optional[int])` tuples
        List of `(word, word_index)` tuples repeated `len(word.cod)`
        times. The tuple `(None, None)` is used to represent
        a `grammar.Cap`.
    word_typ_lists: dict of `Optional[tuple[str, int]]`-`list[int]` KVPs
        Dictionary with key `(word, word_index)` and value the list
        of indices in `types` corresponding to `word.cod`.
    boxes_absolute: list of `(str, int, int)` tuples
        List of `('Cup', cup_left, cup_right)` tuples
        representing the cups in the diagrams.

    Returns
    -------
    `PregroupTreeNode`
        The root node of the resulting tree.
    """
    # NOTE: free_strings does not necessarily contain
    # just a single type here. All remaining types should be considered
    # as the type assigned to the root word.

    # Arbitrary starting node
    root_idx = 0
    root_word_typ = Ty()
    if len(free_strings) > 0:
        # We have free strings - make that the starting node
        root_idx = free_strings[0]
        root_word_typ = types[root_idx]

    # We've added placeholder `(None, None)` values for caps to
    # `words_by_typ_index` when we were tracing earlier.
    # This explains the following type annotation for `root_word`.
    root_word: tuple[
        Optional[Word],
        Optional[int]
    ] = words_by_typ_index[root_idx]

    # NOTE: Free strings might go to different words in some rare cases.
    # For now, raise an exception when that is encountered.
    assert root_word[0] is not None
    assert root_word[1] is not None
    root_word_id = (root_word[0].name, root_word[1])
    root_word_typ_inds = list(word_typ_lists[root_word_id])
    if len(free_strings) > 1:
        for ty_ind in free_strings[1:]:
            if ty_ind in root_word_typ_inds:
                root_word_typ @= types[ty_ind]

    for free_ind in free_strings:
        if free_ind in root_word_typ_inds:
            root_word_typ_inds.remove(free_ind)

    words_with_free_strings = set([
        words_by_typ_index[free_ind] for free_ind in free_strings
    ])
    if len(words_with_free_strings) > 1:
        raise ValueError('Some free strings are not on the root node.')

    root_node = PregroupTreeNode(root_word[0].name,
                                 typ=root_word_typ,
                                 ind=root_word[1],
                                 typ_indxs=root_word_typ_inds)
    all_cups = [box for box in boxes_absolute if box[0] == 'Cup']
    all_nodes: dict[tuple[str, int], PregroupTreeNode] = {
        root_word_id: root_node,
    }

    node_stack = []
    node_stack.append(root_node)

    def _visit(node: PregroupTreeNode):
        # Check all nodes reachable from `node`
        word_typ_indices = list(word_typ_lists[(node.word, node.ind)])

        children_right: list[PregroupTreeNode] = []
        children_words_right: list[tuple[str, int]] = []
        for typ_ind in word_typ_indices:
            if typ_ind not in free_strings:
                for box in all_cups:
                    if box[0] == 'Cup' and typ_ind in box[1:]:
                        if box[2] == typ_ind:
                            # the type (from the parent) is cup.right
                            child_word = words_by_typ_index[box[1]]
                            assert child_word[0] is not None
                            assert child_word[1] is not None

                            key = (child_word[0].name, child_word[1])
                            if key not in node._children_words:
                                child_node = PregroupTreeNode(
                                    child_word[0].name,
                                    typ=types[box[1]],
                                    ind=child_word[1],
                                    parent=node,
                                )
                                all_nodes[key] = child_node
                                child_node.typ_indxs = list(
                                    word_typ_lists[key]
                                )
                                child_node.typ_indxs.remove(box[1])

                                node.children = ([child_node]
                                                 + node.children)
                                node._children_words = (
                                    [key] + node._children_words
                                )
                                node.typ_indxs.remove(box[2])
                            else:
                                # update child node already
                                # in `node.children`
                                child_node = all_nodes[key]
                                child_node.typ = (types[box[1]]
                                                  @ child_node.typ)
                                child_node.typ_indxs.remove(box[1])
                                node.typ_indxs.remove(box[2])
                        elif box[1] == typ_ind:
                            # the type (from the parent) is cup.left
                            child_word = words_by_typ_index[box[2]]
                            assert child_word[0] is not None
                            assert child_word[1] is not None

                            key = (child_word[0].name, child_word[1])
                            if key not in children_words_right:
                                child_node = PregroupTreeNode(
                                    child_word[0].name,
                                    typ=types[box[2]],
                                    ind=child_word[1],
                                    parent=node,
                                )
                                all_nodes[key] = child_node
                                child_node.typ_indxs = list(
                                    word_typ_lists[key]
                                )
                                child_node.typ_indxs.remove(box[2])

                                children_right = ([child_node]
                                                  + children_right)
                                children_words_right = (
                                    [key] + children_words_right
                                )
                                node.typ_indxs.remove(box[1])
                            else:
                                # update child node already
                                # in `node.children`
                                child_node = all_nodes[key]
                                child_node.typ = (types[box[2]]
                                                  @ child_node.typ)
                                child_node.typ_indxs.remove(box[2])
                                node.typ_indxs.remove(box[1])

                        all_cups.remove(box)

        node.children.extend(children_right)
        node._children_words.extend(children_words_right)

    while len(all_cups) > 0 and node_stack:
        node = node_stack.pop()
        _visit(node)
        for child_node in reversed(node.children):
            node_stack.append(child_node)

    return root_node


def has_out_of_bounds_parents(parents: list[list[int]]) -> bool:
    """Check if any node has a parent value outside the allowable
    range `(-1, len(sentence))`."""
    n_tokens = len(parents)
    return any([any([p >= n_tokens or p < -1 for p in parent])
                for parent in parents])


def has_self_as_parents(parents: list[list[int]]) -> bool:
    """Check if any node has its self as one if its parents."""
    return any([i in parent for i, parent in enumerate(parents)])


def has_multiple_roots_assigned(parents: list[list[int]]) -> bool:
    return sum([pp == [ROOT_INDEX] for pp in parents]) > 1


def root_not_assigned(parents: list[list[int]]) -> bool:
    return [ROOT_INDEX] not in parents


def str_to_type(type_str: str) -> Ty:
    """Convert the string representation of an atomic type
    to its `Ty` instance."""
    type_str_split = type_str.split('.')
    typ = Ty(type_str_split[0])
    for r in type_str_split[1:]:
        if r == 'r':
            typ = typ.r
        elif r == 'l':
            typ = typ.l

    return typ


def build_nx_graph(
    parents: list[list[int]],
    graph_cls: type[nx.Graph | nx.DiGraph] = nx.Graph
) -> nx.Graph | nx.DiGraph:
    """Build a `networkx.Graph` or `network.DiGraph` instance using
    the connectivity encoded in the list of parents.

    Parameters
    ----------
    parents: list[list[int]]
        The connectivity information.
    graph_cls: type[nx.Graph]
        The class to be used for constructing the graph object.

    Returns
    -------
    `networkx.Graph` or `networkx.DiGraph`
        The (un)directed graph.

    """
    nx_graph = graph_cls()
    n_tokens = len(parents)
    nx_graph.add_nodes_from(list(range(n_tokens)) + [ROOT_INDEX])
    for child, pars in enumerate(parents):
        for parent_idx, parent in enumerate(pars):
            nx_graph.add_edge(child, parent, parent_idx=parent_idx)

    return nx_graph


def generate_tree(
    tokens: list[str],
    types: list[list[str]],
    parents: list[list[int]],
) -> tuple[list[PregroupTreeNode], list[list[PregroupTreeNode]]]:
    """Generate the pregroup tree from the token, types, and parents data.

    Parameters
    ----------
    tokens: list of str
        The tokens in the sentence.
    types: list of list[str]
        The pregroup tree types assigned to each of the tokens.
    parents: list of list[int]
        The parents assigned to each of the tokens.

    Returns
    -------
    list of `PregroupTreeNode`s
        The root (roots in the case of disjoint subtrees) of
        the generated tree
    list of list of `PregroupTreeNode`s
        The list containing all the nodes in the trees, indexed
        by the token order.

    """
    if has_out_of_bounds_parents(parents):
        err_msg = (
            'Has out-of-bounds parent(s): '
            + f'tokens = {tokens}, types = {types}, parents = {parents}'
        )
        raise ValueError(err_msg)
    elif has_self_as_parents(parents):
        err_msg = (
            'Has self as parent(s): '
            + f'tokens = {tokens}, types = {types}, parents = {parents}'
        )
        raise ValueError(err_msg)

    root_nodes: list[PregroupTreeNode] = []
    nodes: list[list[PregroupTreeNode]] = [[] for _ in range(len(tokens))]

    # Collect nodes
    for i, (token, typs) in enumerate(zip(tokens, types)):
        for typ in typs:
            str_typs = typ.split(' @ ')
            typ_typs = [str_to_type(str_typ) for str_typ in str_typs]
            node = PregroupTreeNode(
                word=token,
                typ=Ty._fromiter(typ_typs),
                ind=i
            )
            nodes[i].append(node)

    # Avoid cycles by reversing one of the edges
    parents_digraph: nx.DiGraph = build_nx_graph(   # type: ignore[assignment]
        parents, nx.DiGraph
    )
    for cycle in nx.simple_cycles(parents_digraph):
        # Order of nodes in `cycle` indicates cycle direction
        # The last index always points to the first index
        start = cycle[0]
        end = cycle[1]
        if len(cycle) == 2:
            if start > end:
                start, end = end, start
            start_old_type = nodes[start][0].typ
            nodes[start][0].typ = Ty()
            parents_digraph.remove_edge(start, end)
            nodes[end][0].typ @= start_old_type.r
        else:
            start_old_type = nodes[start][0].typ
            nodes[start][0].typ = Ty()
            parents_digraph.remove_edge(start, end)
            n_parents = len(nodes[end])
            parents_digraph.add_edge(end, start, parent_idx=n_parents)
            if end < start:
                # Edge is to the left
                new_node_type = start_old_type.l
            else:
                # Edge is to the right
                new_node_type = start_old_type.r
            node = PregroupTreeNode(word=tokens[end],
                                    typ=new_node_type,
                                    ind=end)
            nodes[end].append(node)

    # Build tree structure by defining `children` attributes
    # for each node
    parents_digraph_reverse = parents_digraph.reverse()
    for parent, children_dict in parents_digraph_reverse.adjacency():
        if parent != ROOT_INDEX:
            parent_node = nodes[parent]
            parent_node_inds = set([pn.ind for pn in parent_node])
            assert len(parent_node_inds) == 1
            par_node = parent_node[0]
            child_nodes = []
            for c, edge_attrs in children_dict.items():
                child_nodes.append(nodes[c][edge_attrs['parent_idx']])
            par_node.children = child_nodes
            # Populate parent._children_words attr
            par_node.__post_init__()

    # Get root/start nodes
    # 1. Get connected components
    # 2. For each connected component/subgraph
    #    2.1 If root index (-1) in subgraph,
    #        add node pointing to that to `root_nodes`
    #    2.2 Else, use toposort to get root of component
    parents_graph = parents_digraph.to_undirected()
    for component in nx.connected_components(parents_graph):
        if ROOT_INDEX in component and len(component) > 1:
            # Get node pointing to it
            root_or_start_idx = list(parents_graph[ROOT_INDEX].keys())
        else:
            subgraph = parents_digraph.subgraph(component)
            subgraph_reverse_topo = list(reversed(list(
                nx.topological_sort(subgraph)   # type: ignore[call-overload]
            )))
            root_or_start_idx = [subgraph_reverse_topo[0]]

        for idx in root_or_start_idx:
            if idx != ROOT_INDEX:
                root_nodes.extend(nodes[idx])

    return root_nodes, nodes


def _fix_auxiliary_types(
        word_components_dicts_orig: WordComponentsDictsType,
        word_components_orig: WordComponentsType
        ) -> tuple[WordComponentsDictsType, WordComponentsType]:
    """
    Modifies `word_components_dicts_orig` and `word_components_orig`
    to return the correct type for auxiliary verbs.

    """

    word_components_dicts = fast_deepcopy(word_components_dicts_orig)
    word_components = fast_deepcopy(word_components_orig)

    # Checks for the type `n.r @ s @ ...` for a word and
    # preserves that ordering for the atomic types
    for i, word_component in enumerate(word_components):
        if len(word_component) == 2 and i < len(word_components) - 1:
            # Conditions
            conditions = [
                # 2nd half of compound type is 'n.r @ s'
                str(word_component[1][1] == 'n.r @ s'),
                # 1st half of compound type connects
                # to the right of the word
                word_component[0][0] > i,
                # There's a next word and is connected to this word
                # through the 'n.r @ s' type
                (len(word_components[i + 1]) > 0
                 and word_components[i + 1][0][0] == i
                 and str(word_components[i + 1][0][1]) == 's.r @ n.r.r'),
            ]
            if all(conditions):
                word_component[0], word_component[1] = (
                    word_component[1], word_component[0]
                )
                word_components[i] = word_component

                word_components_dict = {}
                assert len(word_components_dicts[i]) == 2
                start_ind = min([min(inds)
                                 for inds
                                 in word_components_dicts[i].values()])
                for component in word_components[i]:
                    end_ind = start_ind + len(component[1])
                    word_components_dict[component[0]] = list(range(
                        start_ind, end_ind
                    ))
                    start_ind = end_ind

                word_components_dicts[i] = word_components_dict

    return word_components_dicts, word_components


def tree2diagram(
    root: PregroupTreeNode,
    tokens: list[str],
    no_morphisms: bool = False,
    draw_intermediate: bool = False,
    draw_kwargs: Optional[dict[str, Any]] = None
) -> Diagram:
    """Create a pregroup diagram from a pregroup tree.

    Parameters
    ----------
    root : `pregroup_tree.PregroupTreeNode`
        The root of the tree to be converted.
    tokens : list of str
        The tokens in the sentence.
    no_morphisms : bool, default=False
        Whether to exclude the morphisms from the diagram,
        i.e. only the words. Useful for debugging.
    draw_intermediate : bool, default=False
        Whether to draw intermediate state of the diagram
        as the morphisms are added. Useful for debugging.
    draw_kwargs : dict, optional
        Keyword arguments to be passed to the `Diagram.draw` function
        when `draw_intermediate=True`. Useful for debugging.

    Returns
    -------
    `grammar.Diagram`
        The diagram corresponding to the tree.
    """

    curr_nodes = [root]
    next_nodes = []

    word_components: WordComponentsType = [
        [] for _ in range(len(tokens))
    ]
    word_components[root.ind] = [(-1, root.typ)]
    compound_cups = []

    while curr_nodes:
        for node in curr_nodes:
            # children to the left of node are added in order
            # children to the right of node are added in reverse order
            sorted_node_children = sorted(node.children, key=lambda c: c.ind)
            node_children = []
            for child in sorted_node_children:
                if child.ind < node.ind:
                    node_children.append(child)
                else:
                    break

            for child in sorted_node_children[::-1]:
                if child.ind > node.ind:
                    node_children.append(child)
                else:
                    break

            for child in node_children:
                node_word_comp = word_components[node.ind]

                if child.ind < node.ind:
                    insert_loc = len(node_word_comp)
                    for comp_ind, _ in node_word_comp[::-1]:
                        if comp_ind > node.ind or comp_ind < child.ind:
                            insert_loc -= 1
                        else:
                            break
                    word_components[node.ind] = node_word_comp[:insert_loc] + [
                        (child.ind, child.typ.r)
                    ] + node_word_comp[insert_loc:]
                    compound_cups.append((child.ind, node.ind))
                    word_components[child.ind].append((node.ind, child.typ))
                else:
                    insert_loc = 0
                    for comp_ind, _ in node_word_comp:
                        if comp_ind < node.ind or comp_ind > child.ind:
                            insert_loc += 1
                        else:
                            break

                    word_components[node.ind] = node_word_comp[:insert_loc] + [
                        (child.ind, child.typ.l)
                    ] + node_word_comp[insert_loc:]
                    compound_cups.append((node.ind, child.ind))
                    word_components[child.ind] = ([(node.ind, child.typ)]
                                                  + word_components[child.ind])

                next_nodes.append(child)

        curr_nodes = list(next_nodes)
        next_nodes = []

    # Derive morphisms from `word_components`
    word_components_dicts: WordComponentsDictsType = [
        {} for _ in range(len(tokens))
    ]
    ty_global_index = 0
    free_wires = []
    for i, word_component in enumerate(word_components):
        for subword in word_component:
            other_node_ind, subword_ty = subword
            n_ty = len(subword_ty)
            subword_ty_end = ty_global_index + n_ty
            subword_ty_global_inds = list(range(
                ty_global_index,
                subword_ty_end
            ))
            word_components_dicts[i][other_node_ind] = subword_ty_global_inds
            ty_global_index = subword_ty_end

            if other_node_ind == -1:
                free_wires = list(subword_ty_global_inds)

    # Perform type-related surgery before deriving
    # the final set of morphisms
    word_components_dicts, word_components = _fix_auxiliary_types(
        word_components_dicts,
        word_components,
    )

    morphisms = []
    for (start_word_ind, end_word_ind) in compound_cups:
        start_nodes = word_components_dicts[start_word_ind][end_word_ind]
        start_nodes_rev = start_nodes[::-1]
        end_nodes = word_components_dicts[end_word_ind][start_word_ind]
        assert len(start_nodes_rev) == len(end_nodes)
        for sn, en in zip(start_nodes_rev, end_nodes):
            morphisms.append(('Cup', sn, en))

    morphisms = sorted(morphisms, key=lambda t: abs(t[1] - t[2]) - 1)
    free_wires_morphisms_inds = {}
    for free_wire in free_wires:
        morphisms.append(('Free', free_wire, free_wire))
        free_wires_morphisms_inds[free_wire] = len(morphisms) - 1

    # Build words
    word_boxes = []
    for token, word_component in zip(tokens, word_components):
        word_cod = Ty()
        for _, ty in word_component:
            word_cod @= ty
        word_boxes.append(Word(token, word_cod))

    # Add swaps
    new_morphisms: list[tuple[str, int, int]] = []
    offsets = []
    n_morphisms = len(morphisms)
    for i in range(n_morphisms):
        m_t, m_l, m_r = morphisms[i]
        swaps = []
        for j in range(i + 1, n_morphisms):
            o_t, o_l, o_r = morphisms[j]
            if o_l == o_r and m_l < o_l < m_r:
                # o_l == o_r indicates a free string
                swaps.append(('Swap', m_l, o_l))
                morphisms[j] = (o_t, m_l, m_l)
                m_l = o_l
            if o_l != o_r:
                # Always introduce swaps with the free wires
                # if they are in between wires that will get swapped
                if o_l < m_l < o_r < m_r:
                    for (free_wire,
                         free_wire_ind) in free_wires_morphisms_inds.items():
                        if m_l < free_wire < o_r:
                            swaps.append(('Swap', m_l, free_wire))
                            morphisms[free_wire_ind] = ('Free', m_l, m_l)
                            m_l = free_wire
                    swaps.append(('Swap', m_l, o_r))
                    morphisms[j] = (o_t, o_l, m_l)
                    m_l = o_r
                elif m_l < o_l < m_r < o_r:
                    for (free_wire,
                         free_wire_ind) in free_wires_morphisms_inds.items():
                        if o_l < free_wire < m_r:
                            swaps.append(('Swap', o_l, free_wire))
                            morphisms[free_wire_ind] = ('Free', o_l, o_l)
                            o_l = free_wire
                    swaps.append(('Swap', o_l, m_r))
                    morphisms[j] = (o_t, m_r, o_r)
                    m_r = o_l

        new_morphism = (m_t, m_l, m_r)
        new_morphisms.extend(swaps)
        offsets.extend([swap[1] for swap in swaps])
        new_morphisms.append(new_morphism)
        offsets.append(new_morphism[1])

    morphism_str_to_cls: dict[str, type[Cup | Swap]] = {
        'Cup': Cup, 'Swap': Swap
    }
    new_morphisms_cls: list[tuple[type[Cup | Swap], int, int]] = [
        (morphism_str_to_cls[m_t], m_l, m_r)
        for (m_t, m_l, m_r) in new_morphisms if m_t != 'Free'
    ]

    diagram = Diagram.create_pregroup_diagram(
        words=word_boxes,
        morphisms=[],
    )
    if draw_kwargs is None:
        draw_kwargs = {}
    # Draw just the words
    if draw_intermediate:
        diagram.draw(**draw_kwargs)

    if not no_morphisms:
        n_morphisms = len(new_morphisms_cls)
        for i, ((m_cls, m_l, m_r), offset) in enumerate(zip(new_morphisms_cls,
                                                            offsets)):
            left = diagram.cod[:offset]
            right = diagram.cod[offset + 2:]
            box_l, box_r = diagram.cod[m_l:m_l + 1], diagram.cod[m_r:m_r + 1]
            box = m_cls(box_l, box_r)
            diagram >>= left @ box @ right
            if m_cls == Cup:
                for j in range(i + 1, n_morphisms):
                    other = new_morphisms_cls[j]
                    if other[1] < m_l < m_r < other[2]:
                        new_morphisms_cls[j] = (
                            other[0], other[1], other[2] - 2
                        )
                    elif m_r < other[1] < other[2]:
                        new_morphisms_cls[j] = (
                            other[0], other[1] - 2, other[2] - 2
                        )
                        offsets[j] = offsets[j] - 2
                    elif other[1] < other[2] < m_l:
                        pass

            if draw_intermediate:
                diagram.draw(**draw_kwargs)

    return diagram
