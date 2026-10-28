"""Turn Python type hints into JSON Schema, and check model-supplied input against them.

Supported parameter types: ``str``, ``int``, ``float``, ``bool``, ``pathlib.Path``,
``list[X]`` (and other sequences/sets), ``dict[str, X]``, ``Literal[...]``, ``Enum``
subclasses, ``X | None`` / ``Optional[X]``, unions, ``Any``, and ``Annotated[X, "description"]``.
"""

from __future__ import annotations

import collections.abc
import enum
import inspect
import json
import re
import types
import typing
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Literal, Union, get_args, get_origin

from .errors import ToolDefinitionError, ToolInputError

NoneType = type(None)
EMPTY = inspect.Parameter.empty

_ARRAY_TYPES = (list, tuple, set, frozenset)
_ARRAY_ORIGINS = (
    list,
    tuple,
    set,
    frozenset,
    collections.abc.Sequence,
    collections.abc.MutableSequence,
    collections.abc.Set,
    collections.abc.MutableSet,
    collections.abc.Collection,
    collections.abc.Iterable,
)
_OBJECT_ORIGINS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)


@dataclass(frozen=True)
class Param:
    """One model-facing parameter of a tool function."""

    name: str
    annotation: Any
    default: Any = EMPTY
    description: str | None = None

    @property
    def required(self) -> bool:
        return self.default is EMPTY


# --------------------------------------------------------------------------- schema


def unwrap_annotated(tp: Any) -> tuple[Any, list[str]]:
    """Strip ``Annotated[...]`` wrappers; return the inner type and any string metadata."""
    notes: list[str] = []
    while get_origin(tp) is typing.Annotated:
        inner, *meta = get_args(tp)
        notes.extend(m for m in meta if isinstance(m, str))
        tp = inner
    return tp, notes


def json_schema(tp: Any) -> dict[str, Any]:
    """Return the JSON Schema for a type annotation."""
    tp, notes = unwrap_annotated(tp)
    schema = _schema_for(tp)
    if notes:
        schema = {**schema, "description": " ".join(notes)}
    return schema


def _schema_for(tp: Any) -> dict[str, Any]:
    if tp is Any or tp is EMPTY or tp is object:
        return {}
    if tp is None or tp is NoneType:
        return {"type": "null"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str or _is_path_type(tp):
        return {"type": "string"}
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return _enum_schema([member.value for member in tp])
    origin = get_origin(tp)
    if origin is Literal:
        return _enum_schema(list(get_args(tp)))
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        members = [a for a in args if a is not NoneType]
        if len(members) == 1:
            inner = json_schema(members[0])
        else:
            inner = {"anyOf": [json_schema(m) for m in members]}
        if len(members) < len(args):
            return {"anyOf": [inner, {"type": "null"}]}
        return inner
    if tp in _ARRAY_TYPES or origin in _ARRAY_ORIGINS:
        args = get_args(tp)
        if origin is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):
            raise ToolDefinitionError(
                f"fixed-length tuples aren't supported ({tp!r}); use list[...] instead"
            )
        schema: dict[str, Any] = {"type": "array"}
        if args and args[0] is not Any:
            schema["items"] = json_schema(args[0])
        return schema
    if tp is dict or origin in _OBJECT_ORIGINS:
        args = get_args(tp)
        schema = {"type": "object"}
        if args:
            key, value = args
            if key not in (str, Any):
                raise ToolDefinitionError(f"dict keys must be str, not {key!r}")
            if value is not Any:
                schema["additionalProperties"] = json_schema(value)
        return schema
    raise ToolDefinitionError(
        f"unsupported parameter type {tp!r}; use str, int, float, bool, Path, list, dict, "
        "Literal, an Enum, or a union/Optional of those"
    )


def _enum_schema(values: list[Any]) -> dict[str, Any]:
    kinds = {_json_type(v) for v in values}
    if len(kinds) == 1:
        return {"type": kinds.pop(), "enum": values}
    return {"enum": values}


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_path_type(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, PurePath)


def input_schema(params: list[Param]) -> dict[str, Any]:
    """Build a tool's ``input_schema`` from its parameters."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in params:
        schema = json_schema(param.annotation)
        if param.default is None:
            # `x: int | None = None` reads as "optional integer" to the model; it can just
            # leave the parameter out rather than sending null.
            schema = _drop_null(schema)
        if param.description:
            schema = {**schema, "description": param.description}
        if param.default is not EMPTY and param.default is not None:
            default = _json_default(param.default)
            if default is not None:
                schema = {**schema, "default": default}
        properties[param.name] = schema
        if param.required:
            required.append(param.name)
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


def _drop_null(schema: dict[str, Any]) -> dict[str, Any]:
    any_of = schema.get("anyOf")
    if any_of and len(any_of) == 2 and {"type": "null"} in any_of:
        inner = next(s for s in any_of if s != {"type": "null"})
        rest = {k: v for k, v in schema.items() if k != "anyOf"}
        return {**inner, **rest}
    return schema


def _json_default(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, PurePath):
        value = str(value)
    if isinstance(value, (tuple, set, frozenset)):
        value = list(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return None
    return value


# ------------------------------------------------------------------------ coercion


def coerce(value: Any, tp: Any, where: str, *, lenient: bool = True) -> Any:
    """Check ``value`` against annotation ``tp`` and convert it to the Python type.

    Raises :class:`ToolInputError` with a message the model can act on. With ``lenient``,
    harmless mismatches are repaired (``"42"`` for an int, a JSON-encoded string for a
    list), which happens occasionally when tool input is streamed without server-side
    validation.
    """
    tp, _ = unwrap_annotated(tp)
    if tp is Any or tp is EMPTY or tp is object:
        return value
    if tp is None or tp is NoneType:
        if value is None:
            return None
        raise _mismatch(where, "null", value)
    if tp is bool:
        if isinstance(value, bool):
            return value
        if lenient and isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise _mismatch(where, "boolean", value)
    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if lenient and isinstance(value, str) and re.fullmatch(r"\s*[-+]?\d+\s*", value):
            return int(value)
        raise _mismatch(where, "integer", value)
    if tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if lenient and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise _mismatch(where, "number", value)
    if tp is str:
        if isinstance(value, str):
            return value
        if lenient and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise _mismatch(where, "string", value)
    if _is_path_type(tp):
        if isinstance(value, str):
            return tp(value)
        raise _mismatch(where, "string (a path)", value)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return _coerce_choice(value, [m.value for m in tp], where, lambda v: tp(v), names=tp)
    origin = get_origin(tp)
    if origin is Literal:
        return _coerce_choice(value, list(get_args(tp)), where, lambda v: v)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        if value is None and NoneType in args:
            return None
        members = [a for a in args if a is not NoneType]
        errors: list[ToolInputError] = []
        # Exact matches first, so "5" stays a string in `int | str` ...
        for member in members:
            try:
                return coerce(value, member, where, lenient=False)
            except ToolInputError as exc:
                errors.append(exc)
        # ... then let lenient repairs have a go.
        if lenient:
            for member in members:
                try:
                    return coerce(value, member, where, lenient=True)
                except ToolInputError:
                    pass
        if len(members) == 1:
            raise errors[0]
        raise ToolInputError(f"{where}: {_show(value)} doesn't match any of the allowed types")
    if tp in _ARRAY_TYPES or origin in _ARRAY_ORIGINS:
        if lenient and isinstance(value, str):
            value = _parse_json_container(value)
        if not isinstance(value, (list, tuple)):
            raise _mismatch(where, "array", value)
        args = get_args(tp)
        item_type = args[0] if args else Any
        items = [
            coerce(v, item_type, f"{where}[{i}]", lenient=lenient) for i, v in enumerate(value)
        ]
        container = origin if origin in (tuple, set, frozenset) else tp
        if container in (tuple, set, frozenset):
            return container(items)
        return items
    if tp is dict or origin in _OBJECT_ORIGINS:
        if lenient and isinstance(value, str):
            value = _parse_json_container(value)
        if not isinstance(value, dict):
            raise _mismatch(where, "object", value)
        args = get_args(tp)
        value_type = args[1] if len(args) == 2 else Any
        return {
            str(k): coerce(v, value_type, f"{where}.{k}", lenient=lenient) for k, v in value.items()
        }
    return value


def _coerce_choice(
    value: Any,
    options: list[Any],
    where: str,
    build: typing.Callable[[Any], Any],
    names: type[enum.Enum] | None = None,
) -> Any:
    for option in options:
        if option == value and type(option) is type(value):
            return build(option)
    if names is not None and isinstance(value, str) and value in names.__members__:
        return names[value]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        for option in options:
            if str(option) == str(value):
                return build(option)
    choices = ", ".join(json.dumps(o) for o in options)
    raise ToolInputError(f"{where}: expected one of {choices}, got {_show(value)}")


def _parse_json_container(text: str) -> Any:
    stripped = text.strip()
    if stripped[:1] in ("[", "{"):
        try:
            return json.loads(stripped)
        except ValueError:
            pass
    return text


def _mismatch(where: str, expected: str, value: Any) -> ToolInputError:
    return ToolInputError(f"{where}: expected {expected}, got {_show(value)}")


def _show(value: Any) -> str:
    if value is None:
        return "null"
    kind = _json_type(value)
    if isinstance(value, (list, tuple)):
        return f"array of {len(value)} item(s)"
    if isinstance(value, dict):
        return "object"
    text = json.dumps(value, default=str)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{kind} {text}"


# ---------------------------------------------------------------------- docstrings

_SECTION = re.compile(
    r"^(args|arguments|parameters|params|keyword args|returns?|yields?|raises|"
    r"examples?|notes?|see also|attributes|warnings?|todo)\s*:\s*$",
    re.IGNORECASE,
)
_PARAM_SECTIONS = {"args", "arguments", "parameters", "params", "keyword args"}
_GOOGLE_PARAM = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_REST_PARAM = re.compile(r"^:param\s+(?:[^:]*?\s+)?(\w+)\s*:\s*(.*)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google- or reST-style docstring into (description, {param: description})."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    description: list[str] = []
    params: dict[str, list[str]] = {}
    section: str | None = None
    current: str | None = None
    param_indent: int | None = None

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        heading = _SECTION.match(stripped) if indent == 0 else None
        if heading:
            section = heading.group(1).lower()
            current, param_indent = None, None
            continue
        rest = _REST_PARAM.match(stripped)
        if rest:
            section, current = "rest", rest.group(1)
            params[current] = [rest.group(2)]
            continue
        if stripped.startswith(":") and section in (None, "rest"):
            section, current = "rest-other", None
            continue
        if section is None:
            description.append(line)
        elif section in _PARAM_SECTIONS:
            if not stripped:
                continue
            match = _GOOGLE_PARAM.match(stripped)
            if match and (param_indent is None or indent <= param_indent):
                param_indent = indent
                current = match.group(1).lstrip("*")
                params[current] = [match.group(2)]
            elif current is not None:
                params[current].append(stripped)
        elif section == "rest" and current is not None and stripped and indent > 0:
            params[current].append(stripped)

    text = "\n".join(description).strip()
    return text, {name: " ".join(p for p in parts if p).strip() for name, parts in params.items()}


def resolve_hints(func: typing.Callable[..., Any]) -> dict[str, Any]:
    """Resolve a function's annotations, including ``from __future__ import annotations`` ones."""
    try:
        return typing.get_type_hints(func, include_extras=True)
    except Exception as exc:
        name = getattr(func, "__qualname__", repr(func))
        raise ToolDefinitionError(f"couldn't resolve the type hints of {name}: {exc}") from exc
