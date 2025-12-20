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
CCGBank parser
==============

The CCGBank is a translation of the Penn Treebank into a corpus of
Combinatory Categorial Grammar derivations, created by Julia Hockenmaier
and Mark Steedman.

This module provides a parser that automatically turns parses from
CCGBank into :py:class:`.CCGTree` s.

"""

from __future__ import annotations

__all__ = ['CCGBankParseError', 'CCGBankParser']

from collections.abc import Iterator
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import SentenceBatchType
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.typing import StrPathT

if TYPE_CHECKING:
    from lambeq.backend.grammar import Diagram


class CCGBankParseError(Exception):
    """Error raised if parsing fails in CCGBank."""

    def __init__(self, sentence: str = '', message: str = '') -> None:
        if message:
            self.sentence = sentence
            self.message = message
        else:
            self.sentence = ''
            self.message = sentence
        super().__init__(sentence, message)

    def __str__(self) -> str:
        if self.sentence:
            return f'Failed to parse {repr(self.sentence)}: {self.message}.'
        return self.message


class CCGBankParser(CCGParser):
    """A parser for CCGBank trees."""

    ccg_type_regex = re.compile(
            r'((?P<bare_cat>N|NP|S|PP)(\[[a-z]+])?|conj|LRB|RRB|[,.:;])')

    id_regex = re.compile(r"""ID=(?P<id>\S+)  # line begins with "ID=<id>"
                              .*              # rest of the line is ignored
                           """, re.DOTALL | re.VERBOSE)

    tree_regex = re.compile(r"""
        \(<                                 # begin with literal "(<"
           ((?P<is_leaf>        L)|T) \s+   # "L" or "T" depending on node type
           (?P<ccg_str>          \S+) \s+   # the CCG category
           (?(is_leaf)                      # if node is a leaf, then the
               (?P<mod_pos>      \S+) \s+   #    following 4 fields are present
               (?P<orig_pos>     \S+) \s+
               (?P<word>         \S+) \s+
               (?P<pred_arg_cat> \S+)
           |                                # otherwise, the following 2 fields
               (?P<head>         0|1) \s+   #    are present
               (?P<children>     \d+)
           )
          >                                 # close with ">"
        (?(is_leaf)\)|) \s*                 # if node is a leaf, then there is
                                            #    a matching ")"
        """, re.VERBOSE)

    escaped_words = {'-LCB-': '{', '-RCB-': '}', '-LRB-': '(', '-RRB-': ')'}

    def __init__(self,
                 root: StrPathT,
                 verbose: str = VerbosityLevel.SUPPRESS.value) -> None:
        """Initialise a CCGBank parser.

        Parameters
        ----------
        root : str or os.PathLike
            Path to the root of the corpus. The sections must be located
            in `<root>/data/AUTO`.
        verbose : str, default: 'suppress',
            See :py:class:`VerbosityLevel` for options.

        """
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'CCGBankParser.')
        self.root = Path(root)
        self.verbose = verbose

    def section2trees(self,
                      section_id: int,
                      suppress_exceptions: bool = False,
                      verbose: str | None = None) -> dict[str, CCGTree | None]:
        """Parse a CCGBank section into trees.

        Parameters
        ----------
        section_id : int
            The section to parse.
        suppress_exceptions : bool, default: False
            Stop exceptions from being raised, instead returning
            :py:obj:`None` for a tree.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Returns
        -------
        trees : dict
            A dictionary of trees labelled by their ID in CCGBank. If a
            tree fails to parse and exceptions are suppressed, that
            entry is :py:obj:`None`.

        Raises
        ------
        CCGBankParseError
            If parsing fails and exceptions are not suppressed.

        """
        return {key: tree for key, tree in self.section2trees_gen(
            section_id,
            suppress_exceptions=suppress_exceptions,
            verbose=verbose)}

    def section2trees_gen(
        self,
        section_id: int,
        suppress_exceptions: bool = False,
        verbose: str | None = None
    ) -> Iterator[tuple[str, CCGTree | None]]:
        """Parse a CCGBank section into trees, given as a generator.

        The generator only reads data when it is accessed, providing the
        user with control over the reading process.

        Parameters
        ----------
        section_id : int
            The section to parse.
        suppress_exceptions : bool, default: False
            Stop exceptions from being raised, instead returning
            :py:obj:`None` for a tree.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Yields
        ------
        ID, tree : tuple of str and CCGTree
            ID in CCGBank and the corresponding tree. If a
            tree fails to parse and exceptions are suppressed, that
            entry is :py:obj:`None`.

        Raises
        ------
        CCGBankParseError
            If parsing fails and exceptions are not suppressed.

        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'CCGBankParser.')
        path = self.root / 'data' / 'AUTO' / f'{section_id:02}'
        for file in sorted(path.iterdir()):
            with open(file) as f:
                if verbose == VerbosityLevel.TEXT.value:
                    print(f'Parsing `{file}`', file=sys.stderr)
                line_no = 0
                for line in tqdm(
                        f,
                        desc=f'Parsing `{file}`',
                        leave=False,
                        disable=verbose != VerbosityLevel.PROGRESS.value):
                    line_no += 1
                    match = self.id_regex.fullmatch(line)
                    if match:
                        line_no += 1
                        tree = None
                        try:
                            tree = self.sentence2tree(next(f).strip())
                        except CCGBankParseError as e:
                            if not suppress_exceptions:
                                raise CCGBankParseError(
                                    f'Failed to parse tree in `{file}` '
                                    f'line {line_no}: {e.message}'
                                ) from e
                        yield match['id'], tree
                    elif not suppress_exceptions:
                        raise CCGBankParseError('Failed to parse ID in '
                                                f'`{file}` line {line_no}')

    def section2diagrams(
        self,
        section_id: int,
        planar: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None
    ) -> dict[str, Diagram | None]:
        """Parse a CCGBank section into diagrams.

        Parameters
        ----------
        section_id : int
            The section to parse.
        planar : bool, default: False
            Force diagrams to be planar when they contain
            crossed composition.
        suppress_exceptions : bool, default: False
            Stop exceptions from being raised, instead returning
            :py:obj:`None` for a diagram.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.
        Returns
        -------
        diagrams : dict
            A dictionary of diagrams labelled by their ID in CCGBank. If
            a diagram fails to draw and exceptions are suppressed, that
            entry is replaced by :py:obj:`None`.

        Raises
        ------
        CCGBankParseError
            If parsing fails and exceptions are not suppressed.

        """
        return {key: diagram for key, diagram in self.section2diagrams_gen(
            section_id,
            planar=planar,
            suppress_exceptions=suppress_exceptions,
            verbose=verbose)}

    def section2diagrams_gen(
        self,
        section_id: int,
        planar: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None
    ) -> Iterator[tuple[str, Diagram | None]]:
        """Parse a CCGBank section into diagrams, given as a generator.

        The generator only reads data when it is accessed, providing the
        user with control over the reading process.

        Parameters
        ----------
        section_id : int
            The section to parse.
        planar : bool, default: False
            Force diagrams to be planar when they contain
            crossed composition.
        suppress_exceptions : bool, default: False
            Stop exceptions from being raised, instead returning
            :py:obj:`None` for a diagram.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Yields
        ------
        ID, diagram : tuple of str and Diagram
            ID in CCGBank and the corresponding diagram. If a
            diagram fails to draw and exceptions are suppressed, that
            entry is replaced by :py:obj:`None`.

        Raises
        ------
        CCGBankParseError
            If parsing fails and exceptions are not suppressed.

        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'CCGBankParser.')
        trees = self.section2trees_gen(section_id,
                                       suppress_exceptions,
                                       verbose=verbose)
        for k, tree in trees:
            if tree is not None:
                try:
                    diagram = tree.to_diagram(planar)
                except Exception as e:
                    if suppress_exceptions:
                        diagram = None
                    else:
                        raise e
            else:
                diagram = None
            yield k, diagram

    def sentences2trees(self,
                        sentences: SentenceBatchType,
                        tokenised: bool = False,
                        suppress_exceptions: bool = False,
                        verbose: str | None = None) -> list[CCGTree | None]:
        """Parse a CCGBank sentence derivation into a CCGTree.

        The sentence must be in the format outlined in the CCGBank
        manual section D.2 and not just a list of words.

        Parameters
        ----------
        sentences : list of str
            List of sentences to parse.
        suppress_exceptions : bool, default: False
            Stop exceptions from being raised, instead returning
            :py:obj:`None` for a tree.
        tokenised : bool, default: False
            Whether the sentence has been passed as a list of tokens.
            For CCGBankParser, it should be kept `False`.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Returns
        -------
        trees : list of CCGTree
            A list of trees. If a tree fails to parse and exceptions are
            suppressed, that entry is :py:obj:`None`.

        Raises
        ------
        CCGBankParseError
            If parsing fails and exceptions are not suppressed.
        ValueError
            If `tokenised` flag is True (not valid for CCGBankParser).

        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'CCGBankParser.')
        trees = []
        for sentence in sentences:
            if tokenised:
                raise ValueError('`tokenised` set to `True`, but this is not '
                                 'a valid value for CCGBankParser.')
            assert isinstance(sentence, str)
            tree = None
            try:
                tree, pos = CCGBankParser._build_ccgtree(sentence, 0)
                if pos < len(sentence):
                    raise CCGBankParseError(f'extra text from index {pos+1} - '
                                            f'{repr(sentence[pos:])}')
            except Exception as e:
                if not suppress_exceptions:
                    raise CCGBankParseError(sentence, str(e)) from e
            trees.append(tree)
        return trees

    @staticmethod
    def _build_ccgtree(sentence: str, start: int) -> tuple[CCGTree, int]:
        tree_match = CCGBankParser.tree_regex.match(sentence, pos=start)
        if not tree_match:
            raise CCGBankParseError(f'malformed tree from index {start+1} - '
                                    f'{repr(sentence[start:])}')

        ccg_str = tree_match['ccg_str']
        if ccg_str == r'((S[b]\NP)/NP)/':  # fix mistake in CCGBank
            ccg_str = r'(S[b]\NP)/NP'
        biclosed_type = CCGType.parse(
            ccg_str,
            map_atomic=CCGBankParser._map_atomic_type
        )
        pos = tree_match.end()
        if tree_match['is_leaf']:
            word = tree_match['word']
            try:
                word = CCGBankParser.escaped_words[word]
            except KeyError:
                pass
            ccg_tree = CCGTree(text=word,
                               biclosed_type=biclosed_type,
                               metadata={'original': tree_match})
        else:
            children = []
            while not sentence[pos] == ')':
                child, pos = CCGBankParser._build_ccgtree(sentence, pos)
                children.append(child)

            rule = CCGRule.infer_rule([child.biclosed_type
                                       for child in children],
                                      biclosed_type)
            ccg_tree = CCGTree(rule=rule,
                               biclosed_type=biclosed_type,
                               children=children,
                               metadata={'original': tree_match})
            pos += 2
        return ccg_tree, pos

    @staticmethod
    def _map_atomic_type(cat: str) -> str:
        match = CCGBankParser.ccg_type_regex.fullmatch(cat)
        if not match:
            raise CCGBankParseError(f'failed to parse atomic type {repr(cat)}')
        cat = match['bare_cat'] or cat
        if cat == 'N':
            return CCGType.NOUN.name
        elif cat == 'NP':
            return CCGType.NOUN_PHRASE.name
        elif cat == 'S':
            return CCGType.SENTENCE.name
        elif cat == 'PP':
            return CCGType.PREPOSITIONAL_PHRASE.name
        elif cat == 'conj':
            return CCGType.CONJUNCTION.name
        return CCGType.PUNCTUATION.name
