"""Kernel parameter types recovered from Itanium-mangled kernel names.

NVRTC and hiprtc both report a kernel instantiation's lowered (mangled) name,
which encodes the declared type of every parameter. Decoding the subset kernels
use — builtin scalars, pointers, cv-qualifiers, classes, namespaces, templates
and substitutions — lets launches check that each pointer's element type and
each scalar's kind match what the kernel was compiled for.
"""

from __future__ import annotations

from dataclasses import dataclass

_BUILTINS = frozenset("vbcahstijlmxyfdenoz")
_TWO_LETTER_BUILTINS = frozenset({"Dh", "Dn", "Ds", "Di", "Du"})
_STD_ABBREVIATIONS = frozenset("absiod")


@dataclass(frozen=True, slots=True)
class ParameterType:
    """A kernel parameter as a pointer or a by-value type.

    ``builtin`` is the mangled builtin code of the value or of the pointer's
    element (``"f"``, ``"l"``, ...), or ``None`` for class types.
    """

    pointer: bool
    builtin: str | None


class _Unsupported(Exception):
    pass


class _Demangler:
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0
        self.substitutions: list[tuple] = []
        self.template_arguments: list = []

    def peek(self, count: int = 1) -> str:
        return self.text[self.position : self.position + count]

    def take(self, count: int = 1) -> str:
        value = self.peek(count)
        if len(value) != count:
            raise _Unsupported("truncated name")
        self.position += count
        return value

    def number(self) -> int:
        start = self.position
        while self.peek().isdigit():
            self.position += 1
        if start == self.position:
            raise _Unsupported("expected a length")
        return int(self.text[start : self.position])

    def source_name(self) -> str:
        length = self.number()
        return self.take(length)

    def remember(self, node: tuple) -> tuple:
        self.substitutions.append(node)
        return node

    def substitution(self) -> tuple:
        self.take()  # 'S'
        code = self.peek()
        if code in _STD_ABBREVIATIONS:
            self.take()
            return ("class", f"std::{code}")
        index = 0
        if code != "_":
            digits = ""
            while self.peek() not in ("_", ""):
                digits += self.take()
            index = int(digits, 36) + 1
        self.take()  # '_'
        if index >= len(self.substitutions):
            raise _Unsupported("substitution out of range")
        return self.substitutions[index]

    def template_arguments_list(self) -> list:
        self.take()  # 'I'
        arguments = []
        while self.peek() != "E":
            code = self.peek()
            if code == "L":
                end = self.text.index("E", self.position)
                arguments.append(("literal", self.text[self.position + 1 : end]))
                self.position = end + 1
            elif code == "J":
                self.take()
                while self.peek() != "E":
                    arguments.append(self.type())
                self.take()
            elif code == "X" or code == "":
                raise _Unsupported("template expression argument")
            else:
                arguments.append(self.type())
        self.take()  # 'E'
        return arguments

    def name(self, *, is_type: bool) -> tuple[tuple, list | None]:
        """Parse a name; return it and its template arguments, if any."""
        code = self.peek()
        if code == "N":
            return self.nested_name(is_type=is_type)
        if code == "S" and self.peek(2) == "St":
            self.take(2)
            node = ("class", f"std::{self.source_name()}")
        elif code == "S":
            node = self.substitution()
            if self.peek() != "I":
                return node, None
        elif code.isdigit():
            node = ("class", self.source_name())
            if self.peek() != "I":
                return (self.remember(node) if is_type else node), None
            self.remember(node)
        else:
            raise _Unsupported(f"name {code!r}")
        arguments = self.template_arguments_list()
        template = ("class", node[1], tuple(arguments))
        if is_type:
            self.remember(template)
        return template, arguments

    def nested_name(self, *, is_type: bool) -> tuple[tuple, list | None]:
        self.take()  # 'N'
        while self.peek() in ("K", "V", "r", "R", "O"):
            self.take()
        prefix: tuple | None = None
        arguments = None
        while self.peek() != "E":
            code = self.peek()
            if code == "S" and self.peek(2) == "St":
                self.take(2)
                prefix = ("class", "std")
                continue
            if code == "S":
                # A substitution is already a candidate; it is not added again.
                prefix = self.substitution()
                arguments = None
                continue
            if code == "I":
                if prefix is None:
                    raise _Unsupported("template arguments without a name")
                arguments = self.template_arguments_list()
                prefix = ("class", prefix[1], tuple(arguments))
            elif code.isdigit():
                name = self.source_name()
                prefix = ("class", name if prefix is None else f"{prefix[1]}::{name}")
                arguments = None
            else:
                raise _Unsupported(f"nested name component {code!r}")
            if self.peek() != "E" or is_type:
                self.remember(prefix)
        self.take()  # 'E'
        if prefix is None:
            raise _Unsupported("empty nested name")
        return prefix, arguments

    def type(self) -> tuple:
        code = self.peek()
        if code in _BUILTINS:
            self.take()
            return ("builtin", code)
        if self.peek(2) in _TWO_LETTER_BUILTINS:
            return ("builtin", self.take(2))
        if code in ("P", "R", "O"):
            self.take()
            inner = self.type()
            return self.remember(("pointer" if code == "P" else "reference", inner))
        if code in ("K", "V", "r"):
            while self.peek() in ("K", "V", "r"):
                self.take()
            return self.remember(("qualified", self.type()))
        if code == "T":
            self.take()
            index = 0 if self.peek() == "_" else self.number() + 1
            self.take()  # '_'
            node = self.remember(("parameter", index))
            if self.peek() == "I":
                raise _Unsupported("template template parameter")
            return node
        if code == "S" or code == "N" or code.isdigit():
            node, _arguments = self.name(is_type=True)
            return node
        raise _Unsupported(f"type {code!r}")

    def parameters(self) -> list:
        if self.take(2) != "_Z":
            raise _Unsupported("not a mangled name")
        _name, arguments = self.name(is_type=False)
        if arguments is not None:
            self.template_arguments = arguments
            self.type()  # template functions encode their return type
        parameters = []
        while self.position < len(self.text) and self.peek() != ".":
            parameters.append(self.type())
        return [] if parameters == [("builtin", "v")] else parameters


def _resolve(node: tuple, arguments: list, depth: int = 0) -> tuple:
    if depth > 32:
        raise _Unsupported("recursive template parameter")
    if node[0] == "parameter":
        if node[1] >= len(arguments):
            raise _Unsupported("template parameter out of range")
        return _resolve(arguments[node[1]], arguments, depth + 1)
    if node[0] == "qualified":
        return _resolve(node[1], arguments, depth + 1)
    if node[0] in ("pointer", "reference"):
        return (node[0], _resolve(node[1], arguments, depth + 1))
    return node


def parameter_types(lowered: str) -> tuple[ParameterType, ...] | None:
    """Decode the parameter list of a mangled kernel name, or ``None`` if unknown."""

    try:
        demangler = _Demangler(lowered)
        parameters = demangler.parameters()
        types = []
        for parameter in parameters:
            node = _resolve(parameter, demangler.template_arguments)
            if node == ("builtin", "Dn"):
                # decltype(nullptr) passes a null pointer of any type.
                types.append(ParameterType(True, None))
            elif node[0] == "pointer":
                element = node[1]
                types.append(
                    ParameterType(True, element[1] if element[0] == "builtin" else None)
                )
            elif node[0] == "builtin":
                types.append(ParameterType(False, node[1]))
            elif node[0] == "class":
                types.append(ParameterType(False, None))
            else:
                return None
        return tuple(types)
    except (_Unsupported, ValueError, IndexError):
        return None


__all__ = ["ParameterType", "parameter_types"]
