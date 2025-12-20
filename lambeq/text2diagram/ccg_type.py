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

__all__ = ['CCGParseError', 'CCGType']

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from lambeq.backend import grammar


class CCGParseError(Exception):
    """Error when parsing a CCG type string."""

    def __init__(self, cat: str, message: str) -> None:
        super().__init__(cat, message)
        self.cat = cat
        self.message = message

    def __str__(self) -> str:
        return f'Failed to parse {repr(self.cat)}: {self.message}.'


@dataclass
class CCGType:
    r"""A type in the Combinatory Categorical Grammar (CCG).

    Attributes
    ----------
    name : str
        (Atomic types only) The name of an atomic CCG type.
    result : CCGType
        (Complex types only) The result of a complex CCG type.
    direction : '/' or '\'
        (Complex types only) The direction of a complex CCG type.
    argument : CCGType
        (Complex types only) The argument of a complex CCG type.
    is_empty : bool
        Whether the CCG type is the empty type.
    is_atomic : bool
        Whether the CCG type is an atomic type.
    is_complex : bool
        Whether the CCG type is a complex type.
    is_over : bool
        Whether the argument of a complex CCG type appears on the right,
        i.e. X/Y.
    is_under : bool
        Whether the argument of a complex CCG type appears on the left,
        i.e. X\Y.

    """

    _name: str | None
    _result: CCGType | None
    _direction: str | None
    _argument: CCGType | None

    is_empty: bool
    is_atomic: bool
    is_complex: bool
    is_over: bool
    is_under: bool

    CONJ_TAG: ClassVar[str] = '[conj]'
    NOUN: ClassVar[CCGType]
    NOUN_PHRASE: ClassVar[CCGType]
    SENTENCE: ClassVar[CCGType]
    PREPOSITIONAL_PHRASE: ClassVar[CCGType]
    CONJUNCTION: ClassVar[CCGType]
    PUNCTUATION: ClassVar[CCGType]

    def __init__(self,
                 name: str | None = None,
                 result: CCGType | None = None,
                 direction: str | None = None,
                 argument: CCGType | None = None) -> None:
        r"""Initialise a CCG type.

        Parameters
        ----------
        name : str, optional
            (Atomic types only) The name of an atomic CCG type.
        result : CCGType, optional
            (Complex types only) The result of a complex CCG type.
        direction : { '/', '\' }, optional
            (Complex types only) The direction of a complex CCG type.
        argument : CCGType, optional
            (Complex types only) The argument of a complex CCG type.

        """
        self._name = name
        self._result = result
        self._direction = direction
        self._argument = argument

        self.is_under = self._direction == '\\'
        self.is_over = self._direction == '/'
        self.is_complex = self._direction is not None
        self.is_atomic = not self.is_complex and self._name is not None
        self.is_empty = not self.is_complex and self._name is None

        if self.is_complex:
            assert self.is_over or self.is_under
            assert self._result is not None
            assert self._argument is not None
            assert self._name is None
        else:
            assert self._result is None
            assert self._argument is None

    @property
    def name(self) -> str:
        """The name of an atomic CCG type.

        Raises an error if called on a non-atomic CCG type.

        """

        if self._name is not None:
            return self._name
        else:
            raise AttributeError('Non-atomic types do not have a name')

    @property
    def result(self) -> CCGType:
        """The result of a complex CCG type.

        Raises an error if called on a non-complex CCG type.

        """
        if self._result is not None:
            return self._result
        else:
            raise AttributeError('Non-complex types do not have a result')

    @property
    def direction(self) -> str:
        """The direction of a complex CCG type.

        Raises an error if called on a non-complex CCG type.

        """
        if self._direction is not None:
            return self._direction
        else:
            raise AttributeError('Non-complex types do not have a direction')

    @property
    def argument(self) -> CCGType:
        """The argument of a complex CCG type.

        Raises an error if called on a non-complex CCG type.

        """
        if self._argument is not None:
            return self._argument
        else:
            raise AttributeError('Non-complex types do not have a argument')

    @property
    def left(self) -> CCGType:
        """The left-hand side (diagrammatically) of a complex CCG type.

        Raises an error if called on a non-complex CCG type.

        """
        if self.is_over:
            return self.result
        elif self.is_under:
            return self.argument
        else:
            raise AttributeError('Non-complex types do not have a left')

    @property
    def right(self) -> CCGType:
        """The right-hand side (diagrammatically) of a complex CCG type.

        Raises an error if called on a non-complex CCG type.

        """
        if self.is_over:
            return self.argument
        elif self.is_under:
            return self.result
        else:
            raise AttributeError('Non-complex types do not have a right')

    @property
    def is_conjoinable(self) -> bool:
        """Whether the CCG type can be used to conjoin words."""
        return self in (self.CONJUNCTION, self.PUNCTUATION)

    def slash(self, direction: str, argument: CCGType) -> CCGType:
        """Create a complex CCG type."""
        return CCGType(result=self,
                       direction=direction,
                       argument=argument)

    def over(self, argument: CCGType) -> CCGType:
        """Create a complex CCG type with the argument on the right."""
        return self.slash('/', argument)

    def under(self, argument: CCGType) -> CCGType:
        """Create a complex CCG type with the argument on the left."""
        return self.slash('\\', argument)

    def __lshift__(self, rhs: CCGType) -> CCGType:
        return self.over(rhs)

    def __rshift__(self, rhs: CCGType) -> CCGType:
        return rhs.under(self)

    def to_string(self, pretty: bool = False) -> str:
        r"""Convert a CCG type to string.

        Parameters
        ----------
        pretty : bool
            Stringify in a pretty format, using arrows instead of
            slashes. Note that this switches the placement of types in
            an "under" type, i.e. X\Y becomes Y↣X.

        """
        if self.is_empty:
            return '()'
        elif self.is_atomic:
            return self.name
        else:
            result = self.result.to_string(pretty)
            if self.result.is_complex:
                result = f'({result})'

            argument = self.argument.to_string(pretty)
            if self.argument.is_complex:
                argument = f'({argument})'

            if self.is_over:
                template = '{0}↢{1}' if pretty else '{0}/{1}'
            else:
                template = '{1}↣{0}' if pretty else r'{0}\{1}'
            return template.format(result, argument)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f'{class_name}({self.to_string()})'

    @classmethod
    def parse(cls,
              cat: str,
              map_atomic: Callable[[str], str] | None = None) -> CCGType:
        r"""Parse a CCG category string into a CCGType.

        The string should follow the following grammar:

        .. code-block:: text

            atomic_cat  = { <any character except "(", ")", "/", "\"> }
            op          = "/" | "\"
            bracket_cat = atomic_cat
                          | "(" bracket_cat [ op bracket_cat ] ")"
            cat         = bracketed_cat [ op bracket_cat ] [ "[conj]" ]

        Parameters
        ----------
        map_atomic: callable, optional
            If provided, this function is called on the atomic type
            names in the original string, and should return their name
            in the output CCGType. This can be used to fix any
            inconsistencies in capitalisation or unify types, such as
            noun and noun phrase types.

        Returns
        -------
        CCGType
            The parsed category as a CCGType.

        Raises
        ------
        CCGParseError
            If parsing fails.

        Notes
        -----
        Conjunctions follow the CCGBank convention of:

        .. code-block:: text

             x   and  y
             C  conj  C
              \    \ /
               \ C[conj]
                \ /
                 C

        thus ``C[conj]`` is equivalent to ``C\C``.

        """
        is_conj = cat.endswith(cls.CONJ_TAG)
        clean_cat = cat[:-len(cls.CONJ_TAG)] if is_conj else cat

        try:
            ccg_type, end = cls._parse_compound(clean_cat, 0, map_atomic)
        except CCGParseError as e:
            # reraise with original cat string
            raise CCGParseError(cat, e.message) from e

        if is_conj:
            ccg_type = ccg_type >> ccg_type

        extra = clean_cat[end:]
        if extra:
            raise CCGParseError(
                cat,
                f'extra text from index {end} - {repr(extra)}'
            )
        return ccg_type

    @classmethod
    def _parse_compound(
        cls,
        cat: str,
        start: int,
        map_atomic: Callable[[str], str] | None = None
    ) -> tuple[CCGType, int]:
        ccg_type, end = cls._parse_clean(cat, start, map_atomic)
        try:
            op = cat[end]
        except IndexError:
            pass
        else:
            if op in r'\/':
                argument, end = cls._parse_clean(cat, end + 1, map_atomic)
                ccg_type = ccg_type.slash(op, argument)
        return ccg_type, end

    @classmethod
    def _parse_clean(
        cls,
        cat: str,
        start: int,
        map_atomic: Callable[[str], str] | None = None
    ) -> tuple[CCGType, int]:
        if not cat[start:]:
            raise CCGParseError(cat, 'unexpected end of input')
        if cat[start] != '(':
            # base case
            if cat[start] in r'/\)':
                raise CCGParseError(
                    cat,
                    f'unexpected {repr(cat[start])} at index {start}'
                )
            end = start
            while end < len(cat) and cat[end] not in r'/\)':
                if cat[end] == '(':
                    raise CCGParseError(cat, f'unexpected "(" at index {end}')
                end += 1
            atomic_cat = cat[start:end]
            if map_atomic is not None:
                atomic_cat = map_atomic(atomic_cat)
            biclosed_type = cls(atomic_cat)
        else:
            biclosed_type, end = cls._parse_compound(cat, start+1, map_atomic)
            if end >= len(cat):
                raise CCGParseError(
                    cat,
                    f'input ended with unmatched "(" at index {start}'
                )
            assert cat[end] == ')'
            end += 1

        return biclosed_type, end

    def replace(self, original: CCGType, replacement: CCGType) -> CCGType:
        """Replace all occurrences of a sub-type with a different type."""

        if self == original:
            return replacement
        elif self.is_atomic:
            return self
        else:
            new_result = self.result.replace(original, replacement)
            new_argument = self.argument.replace(original, replacement)
            return new_result.slash(self.direction, new_argument)

    def replace_result(self,
                       original: CCGType,
                       replacement: CCGType,
                       direction: str = '|') -> tuple[CCGType, CCGType | None]:
        r"""Replace the innermost category result with a new category.

        This performs a lenient replacement operation. This means that
        it will attempt to replace the specified result category
        `original` with `replacement`, but if `original` cannot be
        found, the innermost result category will be replaced (still by
        `replacement`). This makes it suitable for cases where type
        resolution has occurred, so that type rewrites can propagate.
        This method returns the new category, alongside which category
        has been replaced.
        `direction` can be used to specify a particular structure that
        must be satisfied by the replacement operation. If this is not
        satisfied, then no replacement takes place, and the returned
        replaced result category is `None`.

        Parameters
        ----------
        original : CCGType
            The category that should be replaced.
        replacement : CCGType
            The replacement for the new category.
        direction : str
            Used to check the operations in the category. Consists of
            either 1 or 2 characters, each being one of '/', '\', '|'.
            If 2 characters, the first checks the innermost operation,
            and the second checks the rest. If only 1 character, it is
            used for all checks.

        Returns
        -------
        CCGType
            The new category. If replacement fails, this is set to the
            original category.
        CCGType or None
            The replaced result category. If replacement fails, this is
            set to `None`.

        Notes
        -----
        This function is mainly used for substituting inner types in
        generalised versions of CCG rules. (See :py:meth:`.infer_rule`)

        Examples
        --------
        >>> a, b, c, x, y = map(CCGType, 'abcxy')

        **Example 1**: ``b >> c`` in ``a >> (b >> c)`` is matched and
        replaced with ``x``.

        >>> new, replaced = (a >> (b >> c)).replace_result(b >> c, x)
        >>> print(new, replaced)
        x\a c\b

        **Example 2**: ``x`` cannot be matched, so the innermost
        category ``c`` is replaced instead.

        >>> new, replaced = (a >> (b >> c)).replace_result(x, x << y)
        >>> print(new, replaced)
        ((x/y)\b)\a c

        **Example 3**: if not all operators are ``<<``, then nothing is
        replaced.

        >>> new, replaced = (a >> (c << b)).replace_result(x, y, '/')
        >>> print(new, replaced)
        (c/b)\a None

        **Example 4**: the innermost use of ``<<`` is on ``c`` and
        ``b``, so the target ``c`` is replaced with ``y``.

        >>> new, replaced = (a >> (c << b)).replace_result(x, y, '/|')
        >>> print(new, replaced)
        (y/b)\a c

        **Example 5**: the innermost use of ``>>`` is on ``a`` and
        ``(c << b)``, so its target ``(c << b)`` is replaced by ``y``.

        >>> new, replaced = (a >> (c << b)).replace_result(x, y, r'\|')
        >>> print(new, replaced)
        y\a c/b

        """
        if not (len(direction) in (1, 2) and set(direction).issubset(r'\|/')):
            raise ValueError(f'Invalid direction: `{direction}`')
        if self.is_atomic:
            return self, None

        # `replace` indicates whether `self.result` should be replaced,
        # due to one of the following conditions being true:
        # - `self.result` matches `original`
        # - `self.result` is an atomic type
        # - `self.direction` does not match the required operation
        # - attempting to replace any inner category fails
        replace = self.result == original or self.result.is_atomic
        if not replace:
            if self.direction != direction[-1] != '|':
                replace = True
            else:
                new, old = self.result.replace_result(original,
                                                      replacement,
                                                      direction)
                if old is None:
                    replace = True  # replacing inner category failed

        if replace:
            if self.direction != direction[0] != '|':
                return self, None
            new, old = replacement, self.result

        return new.slash(self.direction, self.argument), old

    def to_grammar(self, Ty: type | None = None) -> grammar.Ty | Any:
        """Turn the CCG type into a lambeq grammar type."""
        if Ty is None:
            from lambeq.backend.grammar import Ty
            assert Ty is not None

        if self.is_over:
            return self.left.to_grammar(Ty) << self.right.to_grammar(Ty)
        elif self.is_under:
            return self.left.to_grammar(Ty) >> self.right.to_grammar(Ty)
        elif self.is_atomic:
            return Ty(self.name)
        else:
            return Ty()

    def split(self, base: CCGType) -> tuple[grammar.Ty,
                                            grammar.Ty,
                                            grammar.Ty]:
        r"""Isolate the inner type of a CCG type, in lambeq.

        For example, if the input is `T = (X\Y)/Z`, the lambeq type
        would be `Y.r @ X @ Z.l` so:

        >>> T = CCGType.parse(r'(X\Y)/Z')
        >>> left, mid, right = T.split(CCGType('X'))
        >>> print(left, mid, right, sep='  +  ')
        Y.r  +  X  +  Z.l

        >>> left, mid, right = T.split(CCGType.parse(r'X\Y'))
        >>> print(left, mid, right, sep='  +  ')
        Ty()  +  Y.r @ X  +  Z.l

        """
        from lambeq.backend.grammar import Ty
        cat = self
        left = right = Ty()
        while cat != base:
            if cat.is_over:
                right = cat.right.to_grammar().l @ right
                cat = cat.left
            else:
                left @= cat.left.to_grammar().r
                cat = cat.right
        return left, cat.to_grammar(), right


CCGType.NOUN = CCGType('n')
CCGType.NOUN_PHRASE = CCGType('np')
CCGType.SENTENCE = CCGType('s')
CCGType.PREPOSITIONAL_PHRASE = CCGType('p')
CCGType.CONJUNCTION = CCGType('conj')
CCGType.PUNCTUATION = CCGType('punc')
