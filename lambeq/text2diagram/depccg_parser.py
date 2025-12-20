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
DepCCG parser
=============
Parser that wraps DepCCG.

"""

from __future__ import annotations

__all__ = ['DepCCGParser', 'DepCCGParseError']

from collections.abc import Iterable
import functools
import logging
from typing import Any, TYPE_CHECKING

from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import (
        SentenceBatchType, SentenceType,
        tokenised_batch_type_check, tokenised_sentence_type_check,
        untokenised_batch_type_check)
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_type import CCGType

if TYPE_CHECKING:
    import depccg
    from depccg.annotator import (annotate_XX, english_annotator,
                                  japanese_annotator)
    from depccg.cat import Category

    from lambeq.backend.grammar import Diagram


def _import_depccg() -> None:
    global depccg, Category
    global annotate_XX, english_annotator, japanese_annotator
    import depccg
    import depccg.allennlp.utils
    from depccg.annotator import (annotate_XX, english_annotator,
                                  japanese_annotator)
    from depccg.cat import Category
    import depccg.lang
    import depccg.parsing


# disable irrelevant logging
logging.getLogger('allennlp.common.params').setLevel(logging.ERROR)
logging.getLogger('depccg.chainer.supertagger').setLevel(logging.ERROR)
logging.getLogger('depccg.lang').setLevel(logging.ERROR)


class DepCCGParseError(Exception):
    def __init__(self, sentence: str) -> None:
        super().__init__(sentence)
        self.sentence = sentence

    def __str__(self) -> str:
        return f'depccg failed to parse: "{self.sentence!r}".'


class DepCCGParser(CCGParser):
    """CCG parser using depccg as the backend."""

    _raw_unary_rules = {'N': ['NP'],
                        'NP': [r'(S[X]/(S[X]\NP))',
                               r'((S[X]\NP)\((S[X]\NP)/NP))',
                               r'(((S[X]\NP)/NP)\(((S[X]\NP)/NP)/NP))',
                               r'(((S[X]\NP)/PP)\(((S[X]\NP)/PP)/NP))'],
                        'PP': [r'((S[X]\NP)\((S[X]\NP)/PP))']}
    _unary_rules = None

    def __init__(self,
                 *,
                 lang: str = 'en',
                 model: str | None = None,
                 use_model_unary_rules: bool = False,
                 annotator: str = 'janome',
                 tokenize: bool | None = None,
                 device: int = -1,
                 root_cats: Iterable[str] | None = None,
                 verbose: str = VerbosityLevel.PROGRESS.value,
                 **kwargs: Any) -> None:
        """Instantiate a parser based on `depccg`.

        Parameters
        ----------
        lang : { 'en', 'ja' }
            The language to use: 'en' for English, 'ja' for Japanese.
        model : str, optional
            The name of the model variant to use, if any.
            `depccg` only has English model variants, namely
            'elmo', 'rebank' and 'elmo_rebank'.
        use_model_unary_rules : bool, default: False
            Use the unary rules supplied by the model instead of the
            ones by `lambeq`.
        annotator : str, default: 'janome'
            The annotator to use, if any. `depccg` supports 'candc' and
            'spacy' for English, and 'janome' and 'jigg' for Japanese.
            By default, no annotator is used for English, and 'janome'
            is used for Japanese.
        tokenize : bool, optional
            Whether to tokenise the input when annotating. This option
            should only be specified when using the 'spacy' annotator.
        device : int, optional
            The ID of the GPU to use. By default, uses the CPU.
        root_cats : iterable of str, optional
            A list of categories allowed at the root of the parse. By
            default, the English categories are:
                - S[dcl]
                - S[wq]
                - S[q]
                - S[qem]
                - NP
            and the Japanese categories are:
                - NP[case=nc,mod=nm,fin=f]
                - NP[case=nc,mod=nm,fin=t]
                - S[mod=nm,form=attr,fin=t]
                - S[mod=nm,form=base,fin=f]
                - S[mod=nm,form=base,fin=t]
                - S[mod=nm,form=cont,fin=f]
                - S[mod=nm,form=cont,fin=t]
                - S[mod=nm,form=da,fin=f]
                - S[mod=nm,form=da,fin=t]
                - S[mod=nm,form=hyp,fin=t]
                - S[mod=nm,form=imp,fin=f]
                - S[mod=nm,form=imp,fin=t]
                - S[mod=nm,form=r,fin=t]
                - S[mod=nm,form=s,fin=t]
                - S[mod=nm,form=stem,fin=f]
                - S[mod=nm,form=stem,fin=t]
        verbose : str, default: 'progress',
            Controls the command-line output of the parser. Only
            'progress' option is available for this parser.
        **kwargs : dict, optional
            Optional arguments passed to `depccg`.

        """
        self.verbose = verbose
        if self.verbose != VerbosityLevel.PROGRESS.value:
            raise ValueError('DepCCGParser only supports '
                             '"progress" level of verbosity. '
                             f'`{self.verbose}` was given.')
        _import_depccg()
        if lang.lower() == 'en':
            if root_cats is None:
                root_cats = ['S[dcl]', 'S[wq]', 'S[q]', 'S[qem]', 'NP']
            self.annotator_fun = english_annotator.get(annotator, annotate_XX)
            self.tokenize = tokenize if tokenize is not None else False
        elif lang.lower() == 'ja':
            if root_cats is None:
                root_cats = ['NP[case=nc,mod=nm,fin=f]',
                             'NP[case=nc,mod=nm,fin=t]',
                             'S[mod=nm,form=attr,fin=t]',
                             'S[mod=nm,form=base,fin=f]',
                             'S[mod=nm,form=base,fin=t]',
                             'S[mod=nm,form=cont,fin=f]',
                             'S[mod=nm,form=cont,fin=t]',
                             'S[mod=nm,form=da,fin=f]',
                             'S[mod=nm,form=da,fin=t]',
                             'S[mod=nm,form=hyp,fin=t]',
                             'S[mod=nm,form=imp,fin=f]',
                             'S[mod=nm,form=imp,fin=t]',
                             'S[mod=nm,form=r,fin=t]',
                             'S[mod=nm,form=s,fin=t]',
                             'S[mod=nm,form=stem,fin=f]',
                             'S[mod=nm,form=stem,fin=t]']
            self.annotator_fun = japanese_annotator.get(annotator, annotate_XX)
            self.tokenize = tokenize if tokenize is not None else True
        else:
            raise ValueError('DepCCGParser does not support language: '
                             f'`{lang}`.')

        depccg.lang.set_global_language_to(lang)
        self.supertagger, config = depccg.instance_models.load_model(model,
                                                                     device)
        (self.apply_binary_rules,
         self.apply_unary_rules,
         self.category_dict,
         _) = depccg.allennlp.utils.read_params(config.config)

        if not use_model_unary_rules:
            if self._unary_rules is None:
                DepCCGParser._unary_rules = {
                        Category.parse(key): [*map(Category.parse, values)]
                        for key, values in self._raw_unary_rules.items()}

            self.apply_unary_rules = functools.partial(
                    depccg.instance_models.GRAMMARS[lang].apply_unary_rules,
                    unary_rules=self._unary_rules
            )

        self.root_categories = [*map(Category.parse, root_cats)]
        self.categories: list[Category] | None = None
        self.kwargs = kwargs

        self._last_trees: list[CCGTree | None] = []

    def sentences2trees(self,
                        sentences: SentenceBatchType,
                        tokenised: bool = False,
                        suppress_exceptions: bool = False,
                        verbose: str | None = None) -> list[CCGTree | None]:
        """Parse multiple sentences into a list of :py:class:`.CCGTree` s.

        Parameters
        ----------
        sentences : list of str, or list of list of str
            The sentences to be parsed, passed either as strings or as
            lists of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        verbose : str, optional
            Controls the form of progress tracking. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser. This class only supports 'progress' verbosity level
            - a progress bar.

        Returns
        -------
        list of CCGTree or None
            The parsed trees. May contain :py:obj:`None` if exceptions
            are suppressed.

        Raises
        ------
        ValueError : If `tokenised` does not match with the input type
        or if verbosity is set to an unsupported value

        """
        if verbose is None:
            verbose = self.verbose
        if verbose != VerbosityLevel.PROGRESS.value:
            raise ValueError('DepCCGParser only supports '
                             '"progress" level of verbosity. '
                             f'`{self.verbose}` was given.')
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentences` does not have type '
                                 '`list[list[str]]`.')
            if TYPE_CHECKING:  # temporary fix
                from typing import cast     # noqa: I100
                sentences = cast(list[list[str]], sentences)
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentences` does not have type '
                                 '`list[str]`.')
            sent_list: list[str] = [str(s) for s in sentences]
            sentences = [sentence.split() for sentence in sent_list]
        empty_indices = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                if suppress_exceptions:
                    empty_indices.append(i)
                else:
                    raise ValueError('sentence is empty.')

        for i in reversed(empty_indices):
            del sentences[i]

        trees = self._last_trees = []
        if sentences:
            parses = self._depccg_parse(sentences)
            for (depccg_tree, *_), sentence in zip(parses, sentences):
                if depccg_tree.score > float('-inf'):
                    trees.append(self._build_ccgtree(depccg_tree.tree))
                elif suppress_exceptions:
                    trees.append(None)
                else:
                    raise DepCCGParseError(' '.join(sentence))

        for i in empty_indices:
            trees.insert(i, None)

        return trees

    def sentence2tree(self,
                      sentence: SentenceType,
                      tokenised: bool = False,
                      suppress_exceptions: bool = False) -> CCGTree | None:
        """Parse a sentence into a :py:class:`.CCGTree`.

        Parameters
        ----------
        sentence : str, list[str]
            The sentence to be parsed, passed either as a string, or as
            a list of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if
            the sentence fails to parse, instead of raising an
            exception, returns :py:obj:`None`.
        tokenised : bool, default: False
            Whether the sentence has been passed as a list of tokens.

        Returns
        -------
        CCGTree or None
            The parsed tree, or :py:obj:`None` on failure.

        Raises
        ------
        ValueError : If `tokenised` does not match with the input type.

        """

        if tokenised:
            if not tokenised_sentence_type_check(sentence):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentence` does not have type '
                                 '`list[str]`.')
            sent: list[str] = [str(token) for token in sentence]
            return self.sentences2trees(
                            [sent],
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.PROGRESS.value)[0]
        else:
            if not isinstance(sentence, str):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentence` does not have type `str`.')
            return self.sentences2trees(
                            [sentence],
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.PROGRESS.value)[0]

    def sentence2diagram(self,
                         sentence: SentenceType,
                         tokenised: bool = False,
                         planar: bool = False,
                         collapse_noun_phrases: bool = True,
                         suppress_exceptions: bool = False) -> Diagram | None:
        """Parse a sentence into a lambeq diagram.

        Parameters
        ----------
        sentence : str, list[str]
            The sentence to be parsed, passed either as a string, or as
            a list of tokens.
        tokenised : bool, default: False
            Whether the sentence has been passed as a list of tokens.
        collapse_noun_phrases : bool, default: True
            If set, then before converting each tree to a diagram, all
            noun phrase types in the tree are changed into nouns. This
            includes sub-types, e.g. `S/NP` becomes `S/N`.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if
            the sentence fails to parse, instead of raising an
            exception, returns :py:obj:`None`.

        Returns
        -------
        :py:class:`lambeq.backend.grammar.Diagram` or None
            The parsed diagram, or :py:obj:`None` on failure.

        Raises
        ------
        ValueError : If `tokenised` does not match with the input type.

        """

        if tokenised:
            if not tokenised_sentence_type_check(sentence):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentence` does not have type '
                                 '`list[str]`.')
            sent: list[str] = [str(token) for token in sentence]
            return self.sentences2diagrams(
                            [sent],
                            planar=planar,
                            collapse_noun_phrases=collapse_noun_phrases,
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.PROGRESS.value)[0]
        else:
            if not isinstance(sentence, str):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentence` does not have type `str`.')
            return self.sentences2diagrams(
                            [sentence],
                            planar=planar,
                            collapse_noun_phrases=collapse_noun_phrases,
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.PROGRESS.value)[0]

    def _depccg_parse(
            self,
            sentences: list[list[str]]) -> list[list[depccg.tree.ScoredTree]]:
        doc = self.annotator_fun(sentences, tokenize=self.tokenize)
        score_result, categories = self.supertagger.predict_doc(
                [[token.word for token in sentence] for sentence in doc])

        if self.categories is None:
            self.categories = [*map(Category.parse, categories)]

        doc, score_result = depccg.parsing.apply_category_filters(
                doc, score_result, self.categories, self.category_dict)

        ret: list[list[depccg.tree.ScoredTree]]
        ret = depccg.parsing.run(doc,
                                 score_result,
                                 self.categories,
                                 self.root_categories,
                                 self.apply_binary_rules,
                                 self.apply_unary_rules,
                                 **self.kwargs)
        return ret

    @staticmethod
    def _to_biclosed(cat: Category) -> CCGType:
        """Transform a depccg category into a biclosed type."""

        if not cat.is_functor:
            if cat.base == 'N':
                return CCGType.NOUN
            elif cat.base == 'NP':
                return CCGType.NOUN_PHRASE
            if cat.base == 'S':
                return CCGType.SENTENCE
            if cat.base == 'PP':
                return CCGType.PREPOSITIONAL_PHRASE
            if cat.base == 'conj':
                return CCGType.CONJUNCTION
            if cat.base in ('LRB', 'RRB') or cat.base in ',.:;':
                return CCGType.PUNCTUATION
        else:
            result = DepCCGParser._to_biclosed(cat.left)
            argument = DepCCGParser._to_biclosed(cat.right)
            return result.slash(cat.slash, argument)
        raise Exception(f'Invalid CCG type: {cat.base}')

    @staticmethod
    def _build_ccgtree(tree: depccg.tree.Tree) -> CCGTree:
        """Transform a depccg derivation tree into a `CCGTree`."""
        biclosed_type = DepCCGParser._to_biclosed(tree.cat)
        if tree.is_leaf:
            children = []
            rule = 'L'
        else:
            children = [*map(DepCCGParser._build_ccgtree, tree.children)]
            if tree.op_string == 'tr':
                rule = 'FTR' if biclosed_type.direction == '/' else 'BTR'
            elif tree.op_symbol == '<un>':
                rule = 'U'
            elif tree.op_string in ('gbx', 'gfc'):
                rule = CCGRule.infer_rule(
                    [child.biclosed_type for child in children],
                    biclosed_type
                )
            else:
                rule = tree.op_string.upper()
        return CCGTree(text=tree.word,
                       rule=rule,
                       biclosed_type=biclosed_type,
                       children=children,
                       metadata={'original': tree})
