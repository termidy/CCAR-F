# Labs

Hands-on labs for the course, one per lecture. They are the same labs that run inside
Udemy, published here so anyone can practise them without a Udemy lab plan.

**You do not need an API key.** Each lab ships with `shopassist_lab.py`, a small
standard-library-only simulator of the `anthropic` SDK. The code you write is exactly the
code you would write against the real API; the replies are canned fixtures chosen by rules,
not a real model. Set `ANTHROPIC_API_KEY` in your environment (or in a `.env` file) and the
very same notebook calls Claude for real.

## How to run a lab

```bash
cd labs/lab_first_request
jupyter lab            # or open notebook.ipynb in VS Code
```

Open **`notebook.ipynb`** and work through it top to bottom. Every cell already holds the
code, with blanks marked `...` and a comment saying what goes in each one. Each task ends
with a `check(...)` call that prints PASS with an explanation or FAIL with a specific hint.
`START_HERE.md` in the lab folder has the details; `solution.ipynb` is the completed
notebook if you get stuck.

The only requirement is Python 3.10+ and Jupyter (`pip install jupyter`). The `anthropic`
and `python-dotenv` packages are optional: if they are missing, the simulator stands in for
them; if they are installed, they are used as-is.

## The labs

| Lab | Practises |
| --- | --- |
| `lab_first_request` | Your first `client.messages.create(...)`, `max_tokens`, `stop_reason` |
| `lab_multi_turn` | The `messages` list as conversation state |
| `lab_helpers` | `add_user_message` / `add_assistant_message` / `chat()` helpers |
| `lab_system_prompt` | System prompts and behaviour the application enforces |
| `lab_temperature` | `temperature`, `stop_sequences` and reading `stop_reason` |
| `lab_evaluation` | A prompt evaluation workflow with test cases |
| `lab_grading` | Grading Claude outputs with code, a second model call and a human |
| `lab_structured_output` | Structured JSON via tool use and `tool_choice` |
| `lab_validation` | Validating extracted data before trusting it |
| `lab_extract_returns` | BUILD: ShopAssist extracts structured return requests |

## A note on `temperature`

The `anthropic` package 1.0+ removed `temperature` from `messages.create()`, and Claude
Sonnet 5 / Opus 4.7+ reject it at the API level. `lab_temperature` still teaches the
concept; inside the simulator the parameter works, and with a real key `shopassist_lab.py`
forwards it to the SDK for you. See the note at the top of that lab and the
[Known issue](../README.md#known-issue-temperature) section in the main README.
