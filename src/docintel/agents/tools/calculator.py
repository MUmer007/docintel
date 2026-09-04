"""
Safe arithmetic evaluation tool for the agent.

Why numexpr instead of Python's eval(): eval() on arbitrary LLM-generated
strings is a real code-execution risk (an LLM could be prompted/manipulated
into generating `__import__('os').system(...)`-style payloads). numexpr only
understands a restricted numeric expression grammar -- no function calls,
no attribute access, no imports -- so it cannot execute arbitrary code even
if the input is adversarial.
"""

from __future__ import annotations

import numexpr
from pydantic import BaseModel


class CalculatorResult(BaseModel):
    expression: str
    result: float | None
    error: str | None = None


def calculate(expression: str) -> CalculatorResult:
    """Safely evaluate a numeric expression, e.g. '(391035 - 383285) / 383285 * 100'."""
    try:
        result = numexpr.evaluate(expression)
        return CalculatorResult(expression=expression, result=float(result))
    except Exception as e:
        return CalculatorResult(expression=expression, result=None, error=str(e))


# Tool schema for Groq function calling (Phase 5.4 will wire this into the agent)
CALCULATOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a numeric arithmetic expression (percentages, growth, sums).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A numeric expression, e.g. '(391035 - 383285) / 383285 * 100'",
                }
            },
            "required": ["expression"],
        },
    },
}