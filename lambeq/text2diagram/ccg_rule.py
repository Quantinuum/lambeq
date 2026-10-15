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

from __future__ import annotations

__all__ = ['CCGRule', 'CCGRuleUseError']

from collections.abc import Sequence
from enum import Enum
from typing import Any

from lambeq.backend.grammar import Diagram, Id
from lambeq.text2diagram.ccg_type import CCGType


class CCGRuleUseError(Exception):
    """Error raised when a :py:class:`CCGRule` is applied incorrectly."""
    def __init__(self, rule: CCGRule, message: str) -> None:
        super().__init__(rule, message)
        self.rule = rule
        self.message = message

    def __str__(self) -> str:
        return f'Illegal use of {self.rule}: {self.message}.'


class CCGRule(str, Enum):
    """An enumeration of the available CCG rules."""

    _symbol: str

    UNKNOWN = 'UNK', ''
    LEXICAL = 'L', ''
    UNARY = 'U', '<U>'
    FORWARD_APPLICATION = 'FA', '>'
    BACKWARD_APPLICATION = 'BA', '<'
    FORWARD_COMPOSITION = 'FC', '>B'
    BACKWARD_COMPOSITION = 'BC', '<B'
    FORWARD_CROSSED_COMPOSITION = 'FX', '>Bx'
    BACKWARD_CROSSED_COMPOSITION = 'BX', '<Bx'
    GENERALIZED_FORWARD_COMPOSITION = 'GFC', '>Bⁿ'
    GENERALIZED_BACKWARD_COMPOSITION = 'GBC', '<Bⁿ'
    GENERALIZED_FORWARD_CROSSED_COMPOSITION = 'GFX', '>Bxⁿ'
    GENERALIZED_BACKWARD_CROSSED_COMPOSITION = 'GBX', '<Bxⁿ'
    REMOVE_PUNCTUATION_LEFT = 'LP', '<p'
    REMOVE_PUNCTUATION_RIGHT = 'RP', '>p'
    FORWARD_TYPE_RAISING = 'FTR', '>T'
    BACKWARD_TYPE_RAISING = 'BTR', '<T'
    CONJUNCTION = 'CONJ', '<&>'

    def __new__(cls, name: str, symbol: str = '') -> CCGRule:
        obj = str.__new__(cls, name)
        obj._value_ = name
        obj._symbol = symbol
        return obj

    @property
    def symbol(self) -> str:
        """The standard CCG symbol for the rule."""
        if self == CCGRule.UNKNOWN:
            raise CCGRuleUseError(self, 'unknown CCG rule')
        else:
            return self._symbol

    @classmethod
    def _missing_(cls, _: Any) -> CCGRule:
        return cls.UNKNOWN

    def check_match(self, /, left: CCGType, right: CCGType) -> None:
        """Raise an exception if the two arguments do not match."""
        if left != right:
            raise CCGRuleUseError(self,
                                  f'mismatched types - {left} != {right}')

    def resolve(self,
                dom: Sequence[CCGType],
                cod: CCGType) -> tuple[CCGType, ...]:
        """Perform type resolution on this rule use.

        This is used to propagate any type changes that has occured in
        the codomain to the domain, such that applying this rule to the
        rewritten domain produces the provided codomain, while remaining
        as compatible as possible with the provided domain.

        Parameters
        ----------
        dom : list of CCGType
            The original domain of this rule use.
        cod : CCGType
            The required codomain of this rule use.

        Returns
        -------
        tuple of CCGType
            The rewritten domain.

        """
        if self == CCGRule.UNKNOWN:
            raise CCGRuleUseError(self, 'unknown CCG rule')
        elif self == CCGRule.LEXICAL:
            assert not dom
            return ()
        elif self == CCGRule.UNARY:
            return cod,
        elif self in (CCGRule.BACKWARD_TYPE_RAISING,
                      CCGRule.FORWARD_TYPE_RAISING):
            return cod.argument.argument,

        left, right = dom
        new_left: CCGType | None
        new_right: CCGType | None
        if self == CCGRule.FORWARD_APPLICATION:
            return cod << right, right
        elif self == CCGRule.BACKWARD_APPLICATION:
            return left, left >> cod
        elif self == CCGRule.FORWARD_COMPOSITION:
            self.check_match(left.right, right.left)
            return cod.result << left.right, right.left << cod.argument
        elif self == CCGRule.BACKWARD_COMPOSITION:
            self.check_match(left.right, right.left)
            return cod.argument >> left.right, right.left >> cod.result
        elif self == CCGRule.FORWARD_CROSSED_COMPOSITION:
            self.check_match(left.right, right.right)
            return cod.right << left.right, cod.left >> right.right
        elif self == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            self.check_match(left.left, right.left)
            return left.left << cod.right, right.left >> cod.left
        elif self == CCGRule.GENERALIZED_FORWARD_COMPOSITION:
            ll, lr = left.left, left.right
            new_right, new_left = cod.replace_result(ll, lr, '/')
            assert new_left is not None
            return new_left << left.right, new_right
        elif self == CCGRule.GENERALIZED_BACKWARD_COMPOSITION:
            rl, rr = right.left, right.right
            new_left, new_right = cod.replace_result(rr, rl, '\\')
            assert new_right is not None
            return new_left, rl >> new_right
        elif self == CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION:
            ll, lr = left.left, left.right
            new_right, new_left = cod.replace_result(ll, lr, r'\|')
            assert new_left is not None
            return new_left << lr, new_right
        elif self == CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION:
            rl, rr = right.left, right.right
            new_left, new_right = cod.replace_result(rr, rl, '/|')
            assert new_right is not None
            return new_left, right.left >> new_right
        elif self == CCGRule.REMOVE_PUNCTUATION_LEFT:
            return left, cod
        elif self == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            return cod, right
        elif self == CCGRule.CONJUNCTION:
            if left.is_conjoinable:
                return cod << right, right
            elif right.is_conjoinable:
                return left, left >> cod
            else:
                raise CCGRuleUseError(self, 'no conjunction found')
        raise AssertionError('unreachable code')

    def __call__(self,
                 dom: Sequence[CCGType],
                 cod: CCGType | None = None) -> Diagram:
        return self.apply(dom, cod)

    def apply(self,
              dom: Sequence[CCGType],
              cod: CCGType | None = None) -> Diagram:
        """Produce a lambeq diagram for this rule.

        This is primarily used by CCG trees that have been resolved.
        This means, for example, that diagrams cannot be produced for
        the conjunction rule, since they are rewritten when resolved.

        Parameters
        ----------
        dom : list of CCGType
            The domain of the diagram.
        cod : CCGType, optional
            The codomain of the diagram. This is only used for
            type-raising rules.

        Returns
        -------
        :py:class:`lambeq.backend.grammar.Diagram`
            The resulting diagram.

        Raises
        ------
        CCGRuleUseError
            If a diagram cannot be produced.

        """
        if self == CCGRule.UNKNOWN:
            raise CCGRuleUseError(self, 'unknown CCG rule')
        elif self == CCGRule.LEXICAL:
            raise CCGRuleUseError(self, 'lexical rules are not applicable')
        elif self == CCGRule.CONJUNCTION:
            raise CCGRuleUseError(
                self, 'conjunctions should be resolved before drawing'
            )

        # unary rules
        elif self in (CCGRule.UNARY,
                      CCGRule.BACKWARD_TYPE_RAISING,
                      CCGRule.FORWARD_TYPE_RAISING):
            if len(dom) != 1:
                raise CCGRuleUseError(
                    self, f'expected a domain of length 1, got {len(dom)}'
                )

            if self == CCGRule.UNARY:
                return Id(dom[0].to_grammar())

            # else type-raising rule
            if cod is None:
                raise CCGRuleUseError(
                    self,
                    'The codomain is required for type-raising rules.'
                )

            result = cod.result.to_grammar()
            if self == CCGRule.BACKWARD_TYPE_RAISING:
                return Id(dom[0].to_grammar()) @ Diagram.caps(result.r, result)
            else:
                return Diagram.caps(result, result.l) @ Id(dom[0].to_grammar())

        # binary rules
        if len(dom) != 2:
            raise CCGRuleUseError(
                self, f'expected a domain of length 2, got {len(dom)}'
            )
        left, right = dom
        if self == CCGRule.FORWARD_APPLICATION:
            # X/Y + Y -> X
            # X @ Y.l + Y -> X
            return Diagram.fa(left.result.to_grammar(), right.to_grammar())
        elif self == CCGRule.BACKWARD_APPLICATION:
            # Y + X\Y -> X
            # Y + Y.r @ X -> X
            return Diagram.ba(left.to_grammar(), right.result.to_grammar())
        elif self == CCGRule.FORWARD_COMPOSITION:
            # X/Y + Y/Z -> X/Z
            # X @ Y.l + Y @ Z.l -> X @ Z.l
            return Diagram.fc(left.left.to_grammar(),
                              left.right.to_grammar(),
                              right.right.to_grammar())
        elif self == CCGRule.BACKWARD_COMPOSITION:
            # Z\Y + X\Y -> X\Z
            # Z.r @ Y + Y.r @ X -> Z.r @ X
            return Diagram.bc(left.left.to_grammar(),
                              left.right.to_grammar(),
                              right.right.to_grammar())
        elif self == CCGRule.FORWARD_CROSSED_COMPOSITION:
            # X/Y + Y\Z -> X\Z
            # X @ Y.l + Z.r @ Y -> Z.r @ X
            return Diagram.fx(left.left.to_grammar(),
                              left.right.to_grammar(),
                              right.left.to_grammar())
        elif self == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            # Y/Z + X\Y -> X/Z
            # Y @ Z.l + Y.r @ X -> X @ Z.l
            return Diagram.bx(left.right.to_grammar(),
                              left.left.to_grammar(),
                              right.right.to_grammar())
        elif self == CCGRule.GENERALIZED_FORWARD_COMPOSITION:
            # X/Y + (Y/Z)/... -> (X/Z)/...
            # X @ Y.l + Y @ Z.l @ ... -> X @ Z.l @ ...
            mid = left.argument.to_grammar()
            return (Id(left.result.to_grammar())
                    @ Diagram.cups(mid.l, mid)
                    @ Id(right.to_grammar()[len(mid):]))
        elif self == CCGRule.GENERALIZED_BACKWARD_COMPOSITION:
            # (Y\Z)\... + X\Y -> (X\Z)\...
            # ... @ Z.r @ Y + Y.r @ X -> ... @ Z.r @ X
            mid = right.argument.to_grammar()
            return (Id(left.to_grammar()[:-len(mid)])
                    @ Diagram.cups(mid, mid.r)
                    @ Id(right.result.to_grammar()))
        elif self == CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION:
            # X/Y + (Y\Z)|... -> (X\Z)|...
            # X @ Y.l + ... @ Z.r @ Y @ ... -> ... @ Z.r @ X @ ...
            mid = left.left.to_grammar()
            l, join, r = right.split(left.right)
            return (
                Diagram.swap(mid << join, l) @ Id(join)
                >> Id(l @ mid) @ Diagram.cups(join.l, join)
            ) @ Id(r)
        elif self == CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION:
            # (Y/Z)|... + X\Y -> (X/Z)|...
            # ... @ Y @ Z.l @ ... + Y.r @ X -> ... @ X @ Z.l @ ...
            mid = right.right.to_grammar()
            l, join, r = left.split(right.left)
            return Id(l) @ (
                Id(join) @ Diagram.swap(r, join >> mid)
                >> Diagram.cups(join, join.r) @ Id(mid @ r)
            )
        elif self == CCGRule.REMOVE_PUNCTUATION_LEFT:
            # punc + X -> X
            return Id(right.to_grammar())
        elif self == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            # X + punc -> X
            return Id(left.to_grammar())
        raise AssertionError('unreachable code')

    @classmethod
    def infer_rule(cls, dom: Sequence[CCGType], cod: CCGType) -> CCGRule:
        """Infer the CCG rule that admits the given domain and codomain.

        Return :py:attr:`CCGRule.UNKNOWN` if no other rule matches.

        Parameters
        ----------
        dom : list of CCGType
            The domain of the rule.
        cod : CCGType
            The codomain of the rule.

        Returns
        -------
        CCGRule
            A CCG rule that admits the required domain and codomain.

        """
        if not dom:
            return CCGRule.LEXICAL
        elif len(dom) == 1:
            if cod.is_complex:
                if cod == cod.result.over(cod.result.under(dom[0])):
                    return CCGRule.FORWARD_TYPE_RAISING
                if cod == cod.result.under(cod.result.over(dom[0])):
                    return CCGRule.BACKWARD_TYPE_RAISING
            return CCGRule.UNARY
        elif len(dom) == 2:
            left, right = dom
            if left == CCGType.PUNCTUATION:
                if cod == right >> right:
                    return CCGRule.CONJUNCTION
                else:
                    return CCGRule.REMOVE_PUNCTUATION_LEFT
            if right == CCGType.PUNCTUATION:
                if cod == left << left:
                    return CCGRule.CONJUNCTION
                else:
                    return CCGRule.REMOVE_PUNCTUATION_RIGHT
            if left == cod << right:
                return CCGRule.FORWARD_APPLICATION
            if right == left >> cod:
                return CCGRule.BACKWARD_APPLICATION
            if CCGType.CONJUNCTION in (left, right):
                return CCGRule.CONJUNCTION

            if cod.is_complex and left.is_complex and right.is_complex:
                ll = left.left
                lr = left.right
                rl = right.left
                rr = right.right

                if lr == rl and (cod.left, cod.right) == (ll, rr):
                    if cod.is_over and left.is_over and right.is_over:
                        return CCGRule.FORWARD_COMPOSITION
                    if cod.is_under and left.is_under and right.is_under:
                        return CCGRule.BACKWARD_COMPOSITION

                if right.is_under:
                    if left.is_over and ll == rl and cod == rr << lr:
                        return CCGRule.BACKWARD_CROSSED_COMPOSITION
                    if left.replace_result(rl, rr, '\\') == (cod, rl):
                        return CCGRule.GENERALIZED_BACKWARD_COMPOSITION
                    if left.replace_result(rl, rr, '/|') == (cod, rl):
                        return CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION
                if left.is_over:
                    if right.is_under and lr == rr and cod == rl >> ll:
                        return CCGRule.FORWARD_CROSSED_COMPOSITION
                    if right.replace_result(lr, ll, '/') == (cod, lr):
                        return CCGRule.GENERALIZED_FORWARD_COMPOSITION
                    if right.replace_result(lr, ll, r'\|') == (cod, lr):
                        return CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION
        return CCGRule.UNKNOWN
