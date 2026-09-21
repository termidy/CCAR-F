# ShopAssist AI

A hands-on course that builds **ShopAssist AI** - a fictional e-commerce customer-support
agent - on the Anthropic Claude API. Each numbered notebook introduces one capability and
layers it onto the same domain, so the project grows as you work through it.

## Yes, this is a Python course

The repository is mostly `.ipynb` files, but `.ipynb` is only a container: every code cell
inside is ordinary Python calling the `anthropic` SDK. Nothing here depends on
notebook-specific magic, so you can copy any cell into a `.py` file and run it with
`python myfile.py` - the behaviour is identical.

Notebooks are used for the lessons because API work means constantly tweaking a prompt and
re-reading the response. A notebook lets you re-run a single cell and see the reply
immediately, instead of re-running a whole script each time.

## Setup

### 1. Python and a virtual environment

Recorded on Python 3.12 (3.12.7). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install the packages

```bash
pip install "anthropic<1" python-dotenv jupyter mcp
```

Versions used in the recordings: `anthropic` 0.111.0, `python-dotenv` 1.2.2, `mcp` 1.28.1.
The `anthropic<1` pin is deliberate - see [Known issue: `temperature`](#known-issue-temperature) below.

Only `anthropic` and `python-dotenv` are needed for notebooks 01–10 and 12–13. `mcp` is
needed by notebook 11 and by `shopassist_mcp_server.py`, which import
`mcp.server.fastmcp.FastMCP`. Everything else the notebooks import (`json`, `pprint`,
`statistics`) is in the Python standard library.

### 3. Your API key

Create a file named `.env` in the repository root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at https://console.anthropic.com. Every notebook starts with `load_dotenv()` and
then `Anthropic()`; the client reads the key from the environment, so the key never appears
in the code. `.env` is already listed in `.gitignore` - keep it that way and never commit it.

## Running the notebooks

```bash
jupyter lab          # then open any notebook in the browser
```

Or open the folder in VS Code with the Python and Jupyter extensions and select `.venv` as
the kernel.

Work through the notebooks in numerical order and run the cells top to bottom - later cells
depend on variables defined by earlier ones.

**One gotcha:** the `%pip install "anthropic<1" python-dotenv` line lives only in the first cell
of `01_first_request.ipynb`, because the later notebooks assume the packages are already
installed. If you start from a later notebook in a fresh environment and hit
`ModuleNotFoundError: No module named 'anthropic'`, run that install line once and carry on.

## The notebooks

| Notebook | What it adds |
| --- | --- |
| `01_first_request.ipynb` | A single `client.messages.create(...)` call - your first response from Claude |
| `02_multi_turn_chat.ipynb` | The `messages` list as conversation state, plus the `add_user_message` / `add_assistant_message` / `chat()` helpers and the ShopAssist `system` prompt |
| `03_temperature.ipynb` | `temperature`, `stop_sequences`, and reading `stop_reason` |
| `04_test_evaluation.ipynb` | Evaluating the agent's answers instead of eyeballing them |
| `05_prompt_eng.ipynb` | Prompt engineering on the ShopAssist system prompt |
| `06_tool_use.ipynb` | Tool definitions and the `tools=` parameter |
| `07_json_output.ipynb` | Forcing reliable structured JSON out of the model |
| `08_tools_extract.ipynb` | Using a tool purely to extract structured data |
| `09_agentic_loop.ipynb` | **The core agentic loop** - dispatch `tool_use` blocks, feed `tool_result` blocks back, repeat until `stop_reason == "end_turn"` |
| `10_tools_errors.ipynb` | Returning structured tool errors instead of raising |
| `11_shopassist_tools.ipynb` | The five canonical ShopAssist tools as `@mcp.tool()` functions |
| `12_agents_hub.ipynb` | Multi-agent design - a coordinator plus billing / order / policy subagents with `allowedTools` |
| `13_shopassist_agents.ipynb` | The full case: one loop with an accumulating `case_facts` dict, driving a damaged-item and duplicate-charge refund end to end |

## The MCP server

`shopassist_mcp_server.py` is the one part of the course that is a plain script rather than a
notebook. It is a FastMCP stdio server exposing the five ShopAssist tools
(`get_customer_by_email`, `lookup_order_by_id`, `check_refund_eligibility`, `process_refund`,
`create_human_escalation`) - the deployable mirror of what notebook 11 prototypes.

```bash
python shopassist_mcp_server.py
```

It prints nothing on its own: it speaks the MCP protocol over stdin/stdout and is meant to be
driven by an MCP client, not read in a terminal.

## Known issue: `temperature`

Students started reporting in late August 2026 that `03_temperature.ipynb` fails with

```
TypeError: Messages.create() got an unexpected keyword argument 'temperature'
```

**Cause.** `anthropic` 1.0.0 (released 20 August 2026) removed `temperature`, `top_p` and
`top_k` from `client.messages.create()`. The error is raised by the SDK before any request is
sent. Independently, Claude Sonnet 5, Opus 4.7 and newer reject the parameter at the API level
(HTTP 400), so the setting is being retired on both sides.

**What changed in this repository.**

- Notebook 01 and the install command above pin the SDK: `pip install "anthropic<1"`. With that
  pin and `model = "claude-sonnet-4-6"` every notebook runs as recorded.
- `temperature=0` was removed from notebooks 04, 05 and 07, where it was only a habit and
  taught nothing. Those notebooks now run on any SDK version.
- Notebook 03 keeps `temperature=0`, because that lesson is about the parameter, and opens with
  a note explaining the situation. On the 1.x SDK the only way to still send it is
  `extra_body={"temperature": 0}`.

**If you already installed the latest SDK**, either re-run the pinned install line and restart
the kernel, or simply delete the `temperature=0` line. The videos show the parameter in a few
places; the affected lectures carry an on-screen note at those moments.

## Model

Every notebook pins the model in a single variable near the top of the file:

```python
model = "claude-sonnet-4-6"
```

To try a different model, change that one variable rather than editing each call.

## A note on cost

These notebooks make real API calls, so they consume credits on your Anthropic account. The requests are small, but re-running cells in a loop adds up - keep an eye on usage in the console while you experiment.
