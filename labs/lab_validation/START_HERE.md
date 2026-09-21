# Lab - Validating Extracted Data

Open **`notebook.ipynb`** and work through it top to bottom.

## What you need to know

This lab runs **offline**. There is no API key and no internet in this workspace, so
`shopassist_lab.py` provides a Claude *simulator* instead. Its replies are canned
fixtures, not a real model.

What is real is everything you type. `max_tokens`, `temperature`, `stop_sequences` and
your system prompt all genuinely change what comes back, so the lessons hold. And the
code is exactly the code you would write against the real API - take this notebook
home, set `ANTHROPIC_API_KEY`, and the same cells call Claude for real.

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
