import pytest
from unittest.mock import MagicMock

from lambeq.text2diagram import BobcatParser, OncillaParser
from lambeq.experimental.discocirc import CoreferenceResolver, DisCoCircReader
from lambeq.backend.grammar import Ty, Box, Id, Frame, Word
from lambeq.text2diagram import PregroupTreeNode


n, s = Ty('n'), Ty('s')


trees = [
    PregroupTreeNode('really', 1, s, children=[
            PregroupTreeNode('Alice', 0, n),
            PregroupTreeNode('likes', 2,
                             n.r @ s, children=[PregroupTreeNode('Bob', 3, n)])
        ]),
    PregroupTreeNode('hates', 1, s, children=[
        PregroupTreeNode('Bob', 0, n),
        PregroupTreeNode('Claire', 2, n)]),
]


frame_diags = [
    (Box('Alice', Ty(), n) @ Box('Bob', Ty(), n)) >> Frame('really', n @ n, n @ n, components=[Box('likes', n @ n, n @ n)]),
    (Box('Bob', Ty(), n) @ Box('Claire', Ty(), n)) >> Box('hates', n @ n, n @ n)
]

sandwich_diags = [
    (Box('Alice', Ty(), n) @ Box('Bob', Ty(), n)) >> Box('really$_{top}$', n @ n, n @ n) >> Box('likes', n @ n, n @ n) >> Box('really$_{bottom}$', n @ n, n @ n),
    (Box('Bob', Ty(), n) @ Box('Claire', Ty(), n)) >> Box('hates', n @ n, n @ n)
]


class MockBobcatParser(BobcatParser):
    def __init__(self):
        pass

    def sentences2trees(self, sentences, **kwargs):
        return None


class MockOncillaParser(OncillaParser):
    def __init__(self):
        pass


class MockCorefResolver(CoreferenceResolver):
    def tokenise_and_coref(self, text):
        return None


@pytest.mark.parametrize(
    'tree, diag_f, diag_s',
    zip(trees, frame_diags, sandwich_diags)
)
def test_discocirc_reader_w_bobcatparser(tree, diag_f, diag_s):
    parser = MockBobcatParser()
    r = DisCoCircReader(parser=parser,
                        coref_resolver=MockCorefResolver())

    d, ns, _ = r._tree2frames_rec(tree, pruned_ids={})
    d = Id().tensor(*ns) >> d
    assert d == diag_f

    d, ns, _, _ = r._tree2sandwiches_rec(tree, pruned_ids={})
    d = Id().tensor(*ns) >> d
    assert d == diag_s


def test_discocirc_reader_w_different_parsers(monkeypatch):
    sentence = 'Alice really likes Bob'.split()
    diag = Word('a', Ty('s'))

    parser = MockOncillaParser()
    monkeypatch.setattr(
        parser, '_sentence2pregrouptree', MagicMock(return_value=trees[0])
    )

    r = DisCoCircReader(parser=parser,
                        coref_resolver=MockCorefResolver())

    r._sentence2tree(sentence, break_cycles=True)
    parser._sentence2pregrouptree.assert_called_once_with(
        sentence, tokenised=True, break_cycles=True
    )

    parser = MockBobcatParser()
    monkeypatch.setattr(
        parser, 'sentence2diagram', MagicMock(return_value=diag)
    )

    r = DisCoCircReader(parser=parser,
                        coref_resolver=MockCorefResolver())

    r._sentence2tree(sentence, break_cycles=True)
    parser.sentence2diagram.assert_called_once_with(
        sentence, tokenised=True,
    )


prune_sentences = [['Alice', 'likes', 'it'],
                   ['it', 'hates', 'it'],
                   ['Alice', 'sees', 'it']]

# `it` in the first two sentences is one entity, the last `it` is another
prune_corefs = [[[0], [],     [0]],
                [[2], [0, 2], []],
                [[],  [],     [2]]]


def _transitive_tree(words):
    return PregroupTreeNode(words[1], 1, s, children=[
        PregroupTreeNode(words[0], 0, n),
        PregroupTreeNode(words[2], 2, n)])


@pytest.mark.parametrize('kwargs, expected', [
    ({'min_noun_freq': 2}, [set(), set(), {2}]),
    ({'pruned_nouns': ['it']}, [{2}, {0, 2}, {2}]),
])
def test_discocirc_reader_pruning(monkeypatch, kwargs, expected):
    parser = MockOncillaParser()
    monkeypatch.setattr(
        parser, '_sentence2pregrouptree',
        MagicMock(side_effect=[_transitive_tree(w) for w in prune_sentences])
    )
    coref_resolver = MockCorefResolver()
    monkeypatch.setattr(
        coref_resolver, 'tokenise_and_coref',
        MagicMock(return_value=(prune_sentences, prune_corefs))
    )

    r = DisCoCircReader(parser=parser, coref_resolver=coref_resolver)
    monkeypatch.setattr(r, '_tree2circuital',
                        MagicMock(wraps=r._tree2circuital))

    r.text2circuit('', **kwargs)
    assert [set(c.args[2]) for c in r._tree2circuital.call_args_list] == expected
