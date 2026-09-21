# Lab - Temperature and Stop Sequences

Open **`notebook.ipynb`** and work through it top to bottom.

## What you need to know

This lab runs **offline**. There is no API key and no internet in this workspace, so
`shopassist_lab.py` provides a Claude *simulator* instead. Its replies are canned
fixtures, not a real model.

What is real is everything you type. `max_tokens`, `temperature`, `stop_sequences` and
your system prompt all genuinely change what comes back, so the lessons hold. And the
code is exactly the code you would write against the real API - take this notebook
home, set `ANTHROPIC_API_KEY`, and the same cells call Claude for real.

## Heads-up: `temperature` is being retired

Since August 2026 the `anthropic` Python package (1.0 and later) no longer accepts
`temperature` in `client.messages.create()` and fails with
`TypeError: ... unexpected keyword argument 'temperature'`. Claude Sonnet 5, Opus 4.7 and
newer also reject the parameter at the API level.

In this workspace nothing changes - the simulator accepts it. If you take the notebook home
with a real key, `shopassist_lab.py` forwards the value to the SDK for you and prints a
one-line note. The lesson itself still holds; on newer models you steer consistency with
your prompt instead of this knob.

## How to check your work

Each task ends with a `check(...)` call. Run the cell and you get either a PASS with an
explanation, or a FAIL with a specific hint about what to change.

```
[PASS] first_request - Make your first Claude request
```

## Useful commands

| Call | What it does |
| --- | --- |
| `lab_info()` | What is simulated and what is not |
| `explain(message)` | Why the simulator returned that particular reply |
| `check_all(...)` | Run every check you have artifacts for |

## If something goes wrong

- **"You still have ... in this request"** - a blank is still empty. Every `...`
  has to be replaced with the value named in the comment beside it.
- **`NameError`** - you skipped a cell. Run the cells in order, from the top.
- **A check fails** - read the `Hint:` line; it names the specific thing to change.
- **Nothing prints** - run the setup cell at the top first, then re-run from there.
