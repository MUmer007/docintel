"""
Agent orchestration: gives the LLM access to search/SQL/calculator tools via
Groq function calling and lets it decide which to use, in what order, for a
given question.

Design choices:
- MAX_STEPS guardrail: prevents runaway tool-calling loops (a model stuck
  calling the same tool repeatedly, or bouncing between tools without
  converging). This is a real production concern -- unbounded agent loops
  are a common source of runaway API cost and latency.
- Tool execution errors are caught and fed back to the model as a tool
  result (not raised), so the agent can recover (e.g. retry with a
  corrected SQL query) rather than the whole request crashing.
- Tool results are truncated (see MAX_TOOL_RESULT_CHARS) to stay within
  Groq's free-tier TPM (tokens-per-minute) rate limit, which large tool
  outputs (e.g. list_available_tables across 178 tables) can exceed within
  1-2 agent steps.

KNOWN LIMITATION -- table discovery: list_available_tables returns only
table_name/ticker/filing_date, not table CONTENTS. Table names are
auto-generated from source position (e.g. "AAPL_2025_10_31_table_326") and
carry no semantic meaning, so the agent has no signal about which of ~60
tables per filing might contain a given line item -- it must guess and
query blindly, often exhausting MAX_STEPS without finding the right table
(observed empirically: a query for "total operating expenses" tried 4 wrong
tables before running out of steps, despite the correct table existing in
the store). A column-header or first-row preview in list_available_tables
would likely help, but this is left unimplemented pending Phase 6 eval
data -- measuring the actual SQL-tool success rate first avoids guessing at
a fix without evidence it helps, which is the same discipline applied to
the retrieval design elsewhere in this project.
"""

from __future__ import annotations

import json

from groq.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from docintel.agents.tools.calculator import CALCULATOR_TOOL_SCHEMA, calculate
from docintel.agents.tools.retrieval_tool import RETRIEVAL_TOOL_SCHEMA, search_filings
from docintel.agents.tools.sql_store import (
    LIST_TABLES_TOOL_SCHEMA,
    SQL_TOOL_SCHEMA,
    list_available_tables,
    run_sql_query,
)
from docintel.core.config import get_settings
from docintel.generation.llm_client import get_groq_client

MAX_STEPS = 5

SYSTEM_PROMPT = """Financial research assistant for SEC 10-K filings.

For a question asking for a SPECIFIC NUMBER (expenses, revenue, a line item): \
call list_available_tables FIRST, then run_sql_query. Do NOT use search_filings \
for precise figures -- it returns prose, not exact numbers.

Use search_filings only for qualitative questions (risks, strategy, descriptions).
Use calculate for any math on numbers you've retrieved.
Cite chunk_id/table names used. If tools don't have the answer, say so."""

TOOL_SCHEMAS = [
    RETRIEVAL_TOOL_SCHEMA,
    LIST_TABLES_TOOL_SCHEMA,
    SQL_TOOL_SCHEMA,
    CALCULATOR_TOOL_SCHEMA,
]


class AgentStep(BaseModel):
    tool_name: str
    tool_args: dict[str, str]
    tool_result_summary: str


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    hit_max_steps: bool = False


# Hard cap on tool-result size fed back to the model. Groq's free-tier TPM limit
# (8000 tokens/min at time of writing) means large tool outputs -- especially
# list_available_tables returning 178 rows, or run_sql_query returning many
# rows -- can blow the budget within just 1-2 agent steps. Truncating here is
# a real production pattern: an agent should summarize/paginate large tool
# outputs rather than dumping everything into context.
MAX_TOOL_RESULT_CHARS = 1500


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CHARS] + f"... [truncated, {len(text)} chars total]"


def _execute_tool(name: str, args: dict[str, str]) -> str:
    """Dispatch a tool call by name, returning a JSON-serializable string result."""
    try:
        if name == "search_filings":
            result = search_filings(args["query"])
            return _truncate(json.dumps(result))
        elif name == "list_available_tables":
            df = list_available_tables()
            # Only surface table_name + ticker + filing_date -- the agent doesn't
            # need "position" to decide what to query, and dropping it shrinks output.
            summary = df[["table_name", "ticker", "filing_date"]]
            return _truncate(summary.to_json(orient="records"))
        elif name == "run_sql_query":
            df = run_sql_query(args["query"])
            return _truncate(df.to_json(orient="records"))
        elif name == "calculate":
            calc_result = calculate(args["expression"])
            return calc_result.model_dump_json()
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        # Feed the error back to the model as a tool result so it can adapt
        # (e.g. fix a malformed SQL query) instead of crashing the whole request.
        return json.dumps({"error": str(e)})


def run_agent(question: str) -> AgentResponse:
    """Run the agentic tool-use loop until the model produces a final answer or hits MAX_STEPS."""
    settings = get_settings()
    client = get_groq_client()

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[AgentStep] = []

    for _ in range(MAX_STEPS):
        # Groq SDK's overload resolution is stricter than necessary here: TOOL_SCHEMAS is a
        # plain list[dict] rather than the SDK's precise ChatCompletionToolParam TypedDict
        # shape, so mypy can't match an overload even though this is correct and works at
        # runtime (verified against the live API in manual testing).
        response = client.chat.completions.create(  # type: ignore[call-overload]
            model=settings.llm.groq_model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1024,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return AgentResponse(answer=message.content or "", steps=steps, hit_max_steps=False)

        # The SDK's response message type isn't directly assignable to the request
        # message param type despite being structurally compatible at runtime.
        messages.append(message)

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = _execute_tool(tool_call.function.name, args)

            steps.append(
                AgentStep(
                    tool_name=tool_call.function.name,
                    tool_args={k: str(v) for k, v in args.items()},
                    tool_result_summary=result[:200],
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return AgentResponse(
        answer="I wasn't able to fully answer this within the allowed number of tool-use steps.",
        steps=steps,
        hit_max_steps=True,
    )
