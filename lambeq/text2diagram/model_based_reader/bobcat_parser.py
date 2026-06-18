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
Bobcat parser
=============
A chart-based parser based on the C&C parser, with scores predicted by a
transformer.

"""

from __future__ import annotations

__all__ = ['BobcatParser', 'BobcatParseError']

from collections.abc import Iterable
import json
import sys
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from lambeq.bobcat import (BertForChartClassification, Category,
                           ChartParser, Grammar, ParseTree,
                           Sentence, Supertag, Tagger)
from lambeq.bobcat.tagger import TaggerOutputSentence
from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import SentenceBatchType
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.text2diagram.model_based_reader.base import ModelBasedReader
from lambeq.typing import StrPathT


class BobcatParseError(Exception):
    def __init__(self, sentence: str) -> None:
        super().__init__(sentence)
        self.sentence = sentence

    def __str__(self) -> str:
        return f'Bobcat failed to parse {self.sentence!r}.'


class BobcatParser(ModelBasedReader, CCGParser):
    """CCG parser using Bobcat as the backend."""

    def __init__(self,
                 model_name_or_path: str = 'bobcat',
                 root_cats: Iterable[str] | None = None,
                 device: int | str | torch.device = 'cpu',
                 cache_dir: StrPathT | None = None,
                 force_download: bool = False,
                 verbose: str = VerbosityLevel.PROGRESS.value,
                 **kwargs: Any) -> None:
        """Instantiate a BobcatParser.

        Parameters
        ----------
        model_name_or_path : str, default: 'bobcat'
            Can be either:
                - The path to a directory containing a Bobcat model.
                - The name of a pre-trained model.
                  By default, it uses the "bobcat" model.
                  See also: `BobcatParser.available_models()`
        root_cats : iterable of str, optional
            A list of the categories allowed at the root of the parse
            tree.
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
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the underlying
            parsers (see Other Parameters). By default, they are set to
            the values in the `pipeline_config.json` file in the model
            directory.

        Other Parameters
        ----------------
        Tagger parameters:
        batch_size : int, optional
            The number of sentences per batch.
        tag_top_k : int, optional
            The maximum number of tags to keep. If 0, keep all tags.
        tag_prob_threshold : float, optional
            The probability multiplier used for the threshold to keep
            tags.
        tag_prob_threshold_strategy : {'relative', 'absolute'}
            If "relative", the probablity threshold is relative to the
            highest scoring tag. Otherwise, the probability is an
            absolute threshold.
        span_top_k : int, optional
            The maximum number of entries to keep per span. If 0, keep
            all entries.
        span_prob_threshold : float, optional
            The probability multiplier used for the threshold to keep
            entries for a span.
        span_prob_threshold_strategy : {'relative', 'absolute'}
            If "relative", the probablity threshold is relative to the
            highest scoring entry. Otherwise, the probability is an
            absolute threshold.

        Chart parser parameters:
        eisner_normal_form : bool, default: True
            Whether to use eisner normal form.
        max_parse_trees : int, optional
            A safety limit to the number of parse trees that can be
            generated per parse before automatically failing.
        beam_size : int, optional
            The beam size to use in the chart cells.
        input_tag_score_weight : float, optional
            A scaling multiplier to the log-probabilities of the input
            tags. This means that a weight of 0 causes all of the input
            tags to have the same score.
        missing_cat_score : float, optional
            The default score for a category that is generated but not
            part of the grammar.
        missing_span_score : float, optional
            The default score for a category that is part of the grammar
            but has no score, due to being below the threshold kept by
            the tagger.

        """
        super().__init__(model_name_or_path=model_name_or_path,
                         device=device,
                         cache_dir=cache_dir,
                         force_download=force_download,
                         verbose=verbose)
        CCGParser.__init__(self, root_cats=root_cats, verbose=verbose)

        # Initialise model
        self._initialise_model(root_cats=root_cats,
                               **kwargs)

    def _initialise_model(self,
                          root_cats: Iterable[str] | None = None,
                          **kwargs) -> None:
        """Initialise the model and load it into the appropriate device.

        Also handle required miscellaneous initialisation steps here."""

        with open(self.model_dir / 'pipeline_config.json') as f:
            config = json.load(f)
        for subconfig in config.values():
            for key in subconfig:
                try:
                    subconfig[key] = kwargs.pop(key)
                except KeyError:
                    pass

        if kwargs:
            raise TypeError('BobcatParser got unexpected keyword argument(s): '
                            f'{", ".join(map(repr, kwargs))}')

        model = (BertForChartClassification
                 .from_pretrained(self.model_dir)
                 .eval()
                 .to(self.device))
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        self.tagger = Tagger(model, tokenizer, **config['tagger'])

        grammar = Grammar.load(self.model_dir / 'grammar.json')
        self.parser = ChartParser(grammar,
                                  self.tagger.model.config.cats,
                                  root_cats,
                                  **config['parser'])

    @staticmethod
    def _prepare_sentence(sent: TaggerOutputSentence,
                          tags: list[str]) -> Sentence:
        """Turn JSON input into a Sentence for parsing."""
        sent_tags = [[Supertag(tags[id], prob) for id, prob in supertags]
                     for supertags in sent.tags]
        spans = {(start, end): {id: score for id, score in scores}
                 for start, end, scores in sent.spans}
        return Sentence(sent.words, sent_tags, spans)

    def sentences2trees(
        self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None
    ) -> list[CCGTree] | None:
        """Parse multiple sentences into a list of :py:class:`.CCGTree` s.

        Parameters
        ----------
        sentences : list of str, or list of list of str
            The sentences to be parsed, passed either as strings or as
            lists of tokens.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Returns
        -------
        list of CCGTree or None
            The parsed trees. (May contain :py:obj:`None` if exceptions
            are suppressed)

        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for '
                             'BobcatParser.')

        sentences_valid, empty_indices = self.validate_sentence_batch(
            sentences,
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions
        )

        trees: list[CCGTree] = []
        if sentences_valid:
            if verbose == VerbosityLevel.TEXT.value:
                print('Tagging sentences.', file=sys.stderr)
            tag_results = self.tagger(sentences_valid, verbose=verbose)
            tags = tag_results.tags
            if verbose == VerbosityLevel.TEXT.value:
                print('Parsing tagged sentences.', file=sys.stderr)
            for sent in tqdm(
                    tag_results.sentences,
                    desc='Parsing tagged sentences',
                    leave=False,
                    disable=verbose != VerbosityLevel.PROGRESS.value):

                try:
                    sentence_input = self._prepare_sentence(sent, tags)
                    result = self.parser(sentence_input)
                    trees.append(self._build_ccgtree(result[0]))
                except Exception as e:
                    if suppress_exceptions:
                        trees.append(None)
                    else:
                        raise BobcatParseError(' '.join(sent.words)) from e

        for i in empty_indices:
            trees.insert(i, None)

        return trees

    @staticmethod
    def _to_biclosed(cat: Category) -> CCGType:
        """Transform a Bobcat category into a biclosed type."""

        if cat.atomic:
            if cat.atom.is_punct:
                return CCGType.PUNCTUATION
            else:
                atom = str(cat.atom)
                if atom == 'N':
                    return CCGType.NOUN
                elif atom == 'NP':
                    return CCGType.NOUN_PHRASE
                elif atom == 'S':
                    return CCGType.SENTENCE
                elif atom == 'PP':
                    return CCGType.PREPOSITIONAL_PHRASE
                elif atom == 'conj':
                    return CCGType.CONJUNCTION
            raise ValueError(f'Invalid atomic type: {cat.atom!r}')
        else:
            result = BobcatParser._to_biclosed(cat.result)
            argument = BobcatParser._to_biclosed(cat.argument)
            return result.slash(cat.dir, argument)

    @staticmethod
    def _build_ccgtree(tree: ParseTree) -> CCGTree:
        """Transform a Bobcat parse tree into a `CCGTree`."""
        children = [BobcatParser._build_ccgtree(child)
                    for child in filter(None, (tree.left, tree.right))]
        if tree.rule.name == 'ADJ_CONJ':
            rule = CCGRule.FORWARD_APPLICATION
        else:
            rule = CCGRule(tree.rule.name)
        return CCGTree(text=tree.word if tree.is_leaf else None,
                       rule=rule,
                       biclosed_type=BobcatParser._to_biclosed(tree.cat),
                       children=children,
                       metadata={'original': tree})
