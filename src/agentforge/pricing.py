"""Rough cost estimates for the usage footer, from list prices (your account may differ)."""

from __future__ import annotations

from .model import Usage

# USD per million tokens: input, output, cache write (5-minute TTL), cache read.
_PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
    "claude-fable-5-1": (10.00, 50.00, 12.50, 0.25),
}


def estimate_cost(model: str, usage: Usage) -> float | None:
    """Estimated USD cost of ``usage`` on ``model``, or None for models without a known price."""
    prices = _PRICES.get(model.removeprefix("anthropic."))
    if prices is None:
        return None
    base_in, base_out, cache_write, cache_read = prices
    total = (
        usage.input_tokens * base_in
        + usage.output_tokens * base_out
        + usage.cache_creation_input_tokens * cache_write
        + usage.cache_read_input_tokens * cache_read
    )
    return total / 1_000_000
