# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hands-on learning project that builds up **ShopAssist AI**, a fictional e-commerce customer-support agent, using the Anthropic Claude API. The material is a sequence of numbered Jupyter notebooks (`01_*` → `13_*`), each introducing one capability and layering it onto the same ShopAssist domain. It is a teaching codebase, not a deployable application — there is no package manifest, test suite, or entrypoint beyond the notebooks and one standalone MCP server.

## Setup & running

- Python 3.12 with a local virtualenv at `.venv/`. Run notebooks/scripts with `.venv/bin/python` and `.venv/bin/jupyter`.
- `.env` holds `ANTHROPIC_API_KEY`. Every notebook calls `load_dotenv()` then `Anthropic()` (the client reads the key from the environment) — so any code you add must keep the key in `.env`, never inline.
- Key installed packages: `anthropic` (0.111), `mcp` (1.28, the FastMCP server SDK), `python-dotenv`, `jupyter`.
- Run the MCP server: `.venv/bin/python shopassist_mcp_server.py` (starts a FastMCP stdio server; it has no CLI output on its own — it's meant to be driven by an MCP client).
- Inspect a notebook as a script without launching Jupyter:
  `.venv/bin/python -c "import json; nb=json.load(open('13_shopassist_agents.ipynb')); print('\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'))"`

## Notebook progression (the "curriculum")

The numbering is the intended reading/build order; later notebooks assume the concepts of earlier ones:

- `01`–`05`: API basics — first request, multi-turn `messages` list, `temperature`, evaluation, prompt engineering. Establishes the `add_user_message` / `add_assistant_message` / `chat(messages)` helper pattern and the ShopAssist `system_prompt`.
- `06`–`08`: tool use and structured output — tool definitions, forcing JSON, and tool-based extraction.
- `09_agentic_loop`: the core **agentic loop** — `while True` calling `client.messages.create(..., tools=...)`, appending `response.content` as the assistant turn, dispatching `tool_use` blocks through a `tool_functions` dict, and feeding `tool_result` blocks back as a user turn until `stop_reason == "end_turn"`. Every later agent notebook is a variation on this loop.
- `10_tools_errors`: structured tool-error convention (see below).
- `11_shopassist_tools`: the canonical ShopAssist tool set, defined as MCP `@mcp.tool()` functions and mirrored in `shopassist_mcp_server.py`.
- `12_agents_hub`: multi-agent design — coordinator + specialized subagents (billing, order, policy) expressed as agent-config dicts with `allowedTools` and per-agent `system` prompts.
- `13_shopassist_agents`: the most complete example — a single agentic loop that maintains an accumulating `case_facts` dict, updated via `update_case_facts()` after each tool result, driving a damaged-item + duplicate-charge refund case end to end.

## Domain conventions to preserve

These patterns are consistent across notebooks and `shopassist_mcp_server.py`; match them when adding tools or examples.

- **Tool result shape**: tools return dicts, never raise. Success and failure are both structured. The error convention is:
  ```python
  {
      "isError": True,
      "errorCategory": "validation" | "permission" | "business",
      "isRetryable": False,
      "customerMessage": "...",   # safe to show the end user
      "developerMessage": "...",  # internal detail / log
  }
  ```
  A "not found" is `isError: False` with a `null` payload + `message`, distinct from a hard error.
- **Verification gating**: order/refund tools require a *verified* `customer_id`, not an email. `get_customer_by_email` resolves email → `customer_id`; `lookup_order_by_id` rejects mismatched ownership with `errorCategory: "permission"`. Tool docstrings explicitly state these preconditions ("Use this only when...", "Do not use this tool to...") — this is intentional prompt surface, keep it.
- **Refund policy is enforced by tools, not the model**: `check_refund_eligibility` (30-day window, must be `delivered`) is policy-only and does not process anything; `process_refund` performs the action; an automatic-refund limit blocks large refunds and routes to `create_human_escalation`. The agent prompts explicitly tell the model *not* to decide eligibility itself.
- **Fixture data**: `CUSTOMERS` / `ORDERS` are in-memory dicts (canonical customer `alex@example.com` → `CUS-1001`; orders `ORD-12345678`, `ORD-87654321`). Reuse these IDs in new examples rather than inventing new ones.
- **Model**: notebooks pin `model = "claude-sonnet-4-6"`. Keep model IDs in a single top-of-notebook variable as the existing code does.

## Other files

- `shopassist_mcp_server.py`: standalone FastMCP server exposing the five ShopAssist tools (`get_customer_by_email`, `lookup_order_by_id`, `check_refund_eligibility`, `process_refund`, `create_human_escalation`). It is the deployable mirror of the tools prototyped in `11_shopassist_tools.ipynb` — keep the two in sync.
- `claude-temperature-slide.html`: a self-contained teaching slide for notebook `03`.
