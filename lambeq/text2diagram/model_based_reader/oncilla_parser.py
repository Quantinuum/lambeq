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
Oncilla parser
==============
A end-to-end pregroup parser, that directly generates pregroup diagrams
from text bypassing CCG as an intermediate representation.

"""

from __future__ import annotations

__all__ = ['OncillaParser', 'OncillaParseError']

import sys
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from lambeq.backend.grammar import Diagram
from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import (SentenceBatchType,
                               SentenceType)
from lambeq.oncilla import (BertForSentenceToTree,
                            SentenceToTreeBertConfig)
from lambeq.text2diagram.model_based_reader.base import ModelBasedReader
from lambeq.text2diagram.pregroup_tree import PregroupTreeNode
from lambeq.text2diagram.pregroup_tree_converter import (generate_tree,
                                                         remove_cycles)
from lambeq.typing import StrPathT


class OncillaParseError(Exception):
    def __init__(self, sentence: str, reason: str = '') -> None:
        super().__init__(sentence, reason)
        self.sentence = sentence
        self.reason = reason

    def __str__(self) -> str:
        out = f'Oncilla failed to parse {self.sentence!r}'
        if self.reason:
            out += f': {self.reason}'
        out += '.'
        return out


class OncillaParser(ModelBasedReader):
    """Parser using Oncilla as the backend."""

    def __init__(
        self,
        model_name_or_path: str = 'oncilla',
        device: int | str | torch.device = 'cpu',
        cache_dir: StrPathT | None = None,
        force_download: bool = False,
        verbose: str = VerbosityLevel.PROGRESS.value,
    ) -> None:
        """Instantiate an OncillaParser.

        Parameters
        ----------
        model_name_or_path : str, default: 'oncilla'
            Can be either:
                - The path to a directory containing an Oncilla model.
                - The name of a pre-trained model.
                By default, it uses the "bert" model.
                See also: `OncillaParser.available_models()`
        device : int, str, or torch.device, default: 'cpu'
            Specifies the device on which to run the tagger model.
            - For CPU, use `'cpu'`.
            - For CUDA devices, use `'cuda:<device_id>'` or `<device_id>`.
            - For Apple Silicon (MPS), use `'mps'`.
            - You may also pass a :py:class:`torch.device` object.
            - For other devices, refer to the PyTorch documentation.
        cache_dir : str or os.PathLike, optional
            The directory to which a downloaded pre-trained model should
            be cached instead of the standard cache
            (`$XDG_CACHE_HOME` or `~/.cache`).
        force_download : bool, default: False
            Force the model to be downloaded, even if it is already
            available locally.
        verbose : str, default: 'progress',
            See :py:class:`VerbosityLevel` for options.
        """
        super().__init__(model_name_or_path=model_name_or_path,
                         device=device,
                         cache_dir=cache_dir,
                         force_download=force_download,
                         verbose=verbose)

        # Initialise model
        self._initialise_model()

    def _initialise_model(self, **kwargs: Any) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model_config = SentenceToTreeBertConfig.from_pretrained(
            self.model_dir
        )
        self.model = BertForSentenceToTree.from_pretrained(
            self.model_dir, config=self.model_config
        ).eval().to(self.device)

    def _sentences2pregrouptrees(
        self,
        sentences: SentenceBatchType,
        break_cycles: bool = False,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None,
    ) -> list[PregroupTreeNode | None]:
        """Parse multiple sentences into a list of pregroup trees.

        Parameters
        ----------
        sentences : list of str, or list of list of str
            The sentences to be parsed.
        break_cycles : bool, default: False
            Flag that indicates whether cycles will be broken in
            the output pregroup tree. This is done by removing
            duplicate nodes, keeping the copy of the node that is closest
            to its parent in the original sentence.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. Not all parsers
            implement all three levels of progress reporting, see the
            respective documentation for each parser. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Returns
        -------
        list of :py:class:`lambeq.text2diagram.PregroupTreeNode` or None
            The pregroup trees. May contain :py:obj:`None` if
            exceptions are suppressed.

        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value '
                             ' for `OncillaParser`.')

        sentences_valid, empty_indices = self.validate_sentence_batch(
            sentences,
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions
        )

        pregroup_trees: list[PregroupTreeNode | None] = []

        if sentences_valid:
            if verbose == VerbosityLevel.TEXT.value:
                print('Turning sentences to pregroup trees.', file=sys.stderr)
            for sent in tqdm(sentences_valid,
                             desc='Turning sentences to pregroup trees',
                             leave=False,
                             disable=verbose != VerbosityLevel.PROGRESS.value):

                pregroup_tree: PregroupTreeNode | None = None
                try:
                    if sent[-1] == '.':
                        # Remove ending '.' as this was removed from
                        # the training dataset for training.
                        sent = sent[:-1]

                    # Predict types and parents
                    parse_output = self.model._sentence2pred(sent,
                                                             self.tokenizer)

                    # Create tree from type and parent preds
                    root_nodes: list[PregroupTreeNode]
                    root_nodes, _ = generate_tree(parse_output.words,
                                                  parse_output.types,
                                                  parse_output.parents)
                except Exception as e:
                    if not suppress_exceptions:
                        raise OncillaParseError(' '.join(sent)) from e
                else:
                    if len(root_nodes) > 1:
                        if not suppress_exceptions:
                            raise OncillaParseError(
                                ' '.join(sent),
                                reason=f'Got {len(root_nodes)} disjoint trees'
                            )

                    elif not len(root_nodes):
                        if not suppress_exceptions:
                            raise OncillaParseError(
                                ' '.join(sent),
                                reason=f'Got {len(root_nodes)} trees'
                            )
                    else:
                        pregroup_tree = root_nodes[0]

                        if break_cycles:
                            remove_cycles(pregroup_tree)

                pregroup_trees.append(pregroup_tree)

        for i in empty_indices:
            pregroup_trees.insert(i, None)

        return pregroup_trees

    def _sentence2pregrouptree(
        self,
        sentence: SentenceType,
        break_cycles: bool = False,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None,
    ) -> PregroupTreeNode | None:
        """Parse a sentence into a pregroup tree.

        Parameters
        ----------
        sentence : str, list[str]
            The sentence to be parsed, passed either as a string, or as
            a list of tokens.
        break_cycles : bool, default: False
            Flag that indicates whether cycles will be broken in
            the output pregroup tree. This is done by removing
            duplicate nodes, keeping the copy of the node that is closest
            to its parent in the original sentence.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. Not all parsers
            implement all three levels of progress reporting, see the
            respective documentation for each parser. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Returns
        -------
        :py:class:`lambeq.text2diagram.PregroupTreeNode` or None
            The pregroup tree, or :py:obj:`None` on failure.

        """
        return self._sentences2pregrouptrees(
            [sentence],     # type: ignore[arg-type]
            break_cycles=break_cycles,
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions,
            verbose=verbose
        )[0]

    def sentences2diagrams(
        self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None,
    ) -> list[Diagram | None]:
        """Parse multiple sentences into a list of lambeq diagrams.

        Parameters
        ----------
        sentences : list of str, or list of list of str
            The sentences to be parsed.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. Not all parsers
            implement all three levels of progress reporting, see the
            respective documentation for each parser. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Returns
        -------
        list of :py:class:`lambeq.backend.grammar.Diagram` or None
            The parsed diagrams. May contain :py:obj:`None` if
            exceptions are suppressed.

        """
        pregroup_trees = self._sentences2pregrouptrees(
            sentences,
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions,
            verbose=verbose
        )

        diagrams: list[Diagram | None] = []

        if verbose is None:
            verbose = self.verbose
        if verbose is VerbosityLevel.TEXT.value:
            print('Turning pregroup trees to diagrams.', file=sys.stderr)
        for tree in tqdm(
            pregroup_trees,
            desc='Turning pregroup trees to diagrams',
            leave=False,
            total=len(pregroup_trees),
            disable=verbose != VerbosityLevel.PROGRESS.value
        ):
            diagram: Diagram | None = None

            if tree is not None:
                try:
                    tokens = tree.get_words()
                    diagram = tree.to_diagram(tokens=tokens)
                except Exception as e:
                    if not suppress_exceptions:
                        raise OncillaParseError(' '.join(tokens)) from e

            diagrams.append(diagram)

        return diagrams

    def sentence2diagram(
        self,
        sentence: SentenceType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None
    ) -> Diagram | None:
        """Parse a sentence into a lambeq diagram.

        Parameters
        ----------
        sentence : str, or list of str
            The sentence to be parsed.
        tokenised : bool, default: False
            Whether the sentence has been passed as a list of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if
            the sentence fails to parse, instead of raising an
            exception, returns :py:obj:`None`.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. Not all parsers
            implement all three levels of progress reporting, see the
            respective documentation for each parser. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Returns
        -------
        :py:class:`lambeq.backend.grammar.Diagram` or None
            The parsed diagram, or :py:obj:`None` on failure.

        """
        return self.sentences2diagrams(
            [sentence],     # type: ignore[arg-type]
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions,
            verbose=verbose
        )[0]
