from __future__ import annotations

import enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import pytest

from agentforge.errors import ToolDefinitionError, ToolInputError
from agentforge.schema import Param, coerce, input_schema, json_schema, parse_docstring


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (Path, {"type": "string"}),
        (Any, {}),
        (list[int], {"type": "array", "items": {"type": "integer"}}),
        (list, {"type": "array"}),
        (set[str], {"type": "array", "items": {"type": "string"}}),
        (tuple[int, ...], {"type": "array", "items": {"type": "integer"}}),
        (dict[str, int], {"type": "object", "additionalProperties": {"type": "integer"}}),
        (dict, {"type": "object"}),
        (Literal["a", "b"], {"type": "string", "enum": ["a", "b"]}),
        (Literal[1, "x"], {"enum": [1, "x"]}),
        (Color, {"type": "string", "enum": ["red", "green"]}),
        (Optional[int], {"anyOf": [{"type": "integer"}, {"type": "null"}]}),
        (int | str, {"anyOf": [{"type": "integer"}, {"type": "string"}]}),
        (
            Union[int, str, None],
            {"anyOf": [{"anyOf": [{"type": "integer"}, {"type": "string"}]}, {"type": "null"}]},
        ),
        (Annotated[int, "how many"], {"type": "integer", "description": "how many"}),
        (
            list[Literal["x", "y"]],
            {"type": "array", "items": {"type": "string", "enum": ["x", "y"]}},
        ),
    ],
)
def test_json_schema(annotation: Any, expected: dict[str, Any]) -> None:
    assert json_schema(annotation) == expected


@pytest.mark.parametrize("annotation", [tuple[int, str], dict[int, str], complex, object.__class__])
def test_unsupported_types_are_rejected(annotation: Any) -> None:
    with pytest.raises(ToolDefinitionError):
        json_schema(annotation)


def test_input_schema_marks_required_and_defaults() -> None:
    params = [
        Param("path", str, description="Where."),
        Param("limit", int, 10),
        Param("tag", Optional[str], None, "A tag."),
        Param("mode", Literal["fast", "slow"], "fast"),
        Param("color", Color, Color.RED),
    ]
    assert input_schema(params) == {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Where."},
            "limit": {"type": "integer", "default": 10},
            "tag": {"type": "string", "description": "A tag."},
            "mode": {"type": "string", "enum": ["fast", "slow"], "default": "fast"},
            "color": {"type": "string", "enum": ["red", "green"], "default": "red"},
        },
        "required": ["path"],
    }


def test_input_schema_without_required_params_omits_required() -> None:
    assert "required" not in input_schema([Param("x", int, 1)])


@pytest.mark.parametrize(
    ("value", "annotation", "expected"),
    [
        ("hi", str, "hi"),
        (5, str, "5"),
        (5, int, 5),
        (5.0, int, 5),
        (" 42 ", int, 42),
        (3, float, 3.0),
        ("2.5", float, 2.5),
        (True, bool, True),
        ("false", bool, False),
        ("a.txt", Path, Path("a.txt")),
        ([1, "2"], list[int], [1, 2]),
        ("[1, 2]", list[int], [1, 2]),
        ([1, 2], tuple[int, ...], (1, 2)),
        (["a", "a"], set[str], {"a"}),
        ({"a": "1"}, dict[str, int], {"a": 1}),
        ('{"a": 1}', dict[str, int], {"a": 1}),
        ("red", Color, Color.RED),
        ("GREEN", Color, Color.GREEN),
        ("b", Literal["a", "b"], "b"),
        ("1", Literal[1, 2], 1),
        (None, Optional[int], None),
        ("5", int | str, "5"),  # exact match wins over a lenient conversion
        ("5", int | None, 5),
        ({"x": [1]}, Any, {"x": [1]}),
    ],
)
def test_coerce_accepts_and_converts(value: Any, annotation: Any, expected: Any) -> None:
    assert coerce(value, annotation, "p") == expected


@pytest.mark.parametrize(
    ("value", "annotation", "message"),
    [
        ("abc", int, "p: expected integer, got string"),
        (True, int, "p: expected integer, got boolean"),
        (1.5, int, "expected integer"),
        ("maybe", bool, "expected boolean"),
        ([], str, "expected string, got array"),
        ("c", Literal["a", "b"], 'expected one of "a", "b"'),
        ("blue", Color, "expected one of"),
        ([1, "x"], list[int], "p[1]: expected integer"),
        ({"k": "v"}, dict[str, int], "p.k: expected integer"),
        ("nope", list[int], "expected array"),
        (None, int, "expected integer, got null"),
        ([1], int | str, "doesn't match any of the allowed types"),
    ],
)
def test_coerce_rejects_with_actionable_messages(value: Any, annotation: Any, message: str) -> None:
    with pytest.raises(ToolInputError, match=message.replace("[", r"\[").replace("]", r"\]")):
        coerce(value, annotation, "p")


def test_strict_coercion_does_not_repair() -> None:
    with pytest.raises(ToolInputError):
        coerce("42", int, "p", lenient=False)


def test_parse_google_docstring() -> None:
    def f() -> None:
        """Do the thing.

        Longer explanation
        over two lines.

        Args:
            path: The file path.
            limit (int): Maximum rows. Continues
                on the next line.
            *rest: Ignored.

        Returns:
            Something.
        """

    description, params = parse_docstring(f.__doc__)
    assert description == "Do the thing.\n\nLonger explanation\nover two lines."
    assert params == {
        "path": "The file path.",
        "limit": "Maximum rows. Continues on the next line.",
        "rest": "Ignored.",
    }


def test_parse_rest_docstring() -> None:
    doc = """Fetch a thing.

    :param name: The name.
    :param int retries: How many
        times to retry.
    :returns: The thing.
    """
    description, params = parse_docstring(doc)
    assert description == "Fetch a thing."
    assert params == {"name": "The name.", "retries": "How many times to retry."}


def test_parse_docstring_handles_none() -> None:
    assert parse_docstring(None) == ("", {})
