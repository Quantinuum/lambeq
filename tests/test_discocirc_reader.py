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

def _copular_tree():
    return PregroupTreeNode('is', 1, Ty('s'), children=[
        PregroupTreeNode('He', 0, n),
        PregroupTreeNode('', 2, n, children=[
            PregroupTreeNode('very', 3, n, children=[
                PregroupTreeNode('talented', 4, n @ n.l),
                PregroupTreeNode('programmer', 5, n)
            ])
        ])
    ])


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


def test_sandwich_prefers_local_head_noun():
    parser = MockBobcatParser()
    r = DisCoCircReader(parser=parser,
                        coref_resolver=MockCorefResolver())

    tree = _copular_tree()
    _, nouns, nids, _ = r._tree2sandwiches_rec(tree, pruned_ids=set())

    assert [box.name for box in nouns] == ['He', 'programmer']
    assert nids == [0, 5]
