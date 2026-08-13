# Contributing to PromptLens

Thanks for your interest in contributing. PromptLens is a lightweight evaluation harness for prompts, LLMs, and agent workflows, and contributions of all sizes are welcome: bug reports, docs fixes, new providers, judges, and exporters.

## Development setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/brainsparker/PromptLens.git
cd PromptLens
pip install -e .
pip install pytest pytest-asyncio pytest-mock pytest-cov
```

## Running tests

```bash
pytest tests/ -q
```

Tests run without API keys. Live evaluation runs require provider keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY), but please do not add tests that require live credentials.

## Code style

The project uses black and ruff, both configured in pyproject.toml (line length 100).

```bash
black promptlens/
ruff check promptlens/
```

## Adding a provider, judge, or exporter

PromptLens has a plugin architecture. The existing implementations are the best reference:

- Providers live in `promptlens/providers/`. Subclass the base provider, implement the query method, and register the provider name.
- Judges and scoring live in `promptlens/judges/`.
- Exporters live in `promptlens/exporters/` (HTML, JSON, CSV, Markdown).

Keep new dependencies to a minimum. If a provider can be reached with `requests` or `aiohttp`, prefer that over adding an SDK.

## Pull requests

1. Fork and create a feature branch.
2. Keep the change focused and reviewable in one sitting.
3. Add or update tests for behavior changes.
4. Make sure `pytest`, `black --check`, and `ruff check` pass.
5. Describe what changed and why in the PR body, with a usage example if you added a feature.

## Reporting bugs

Open an issue with your Python version, the command or config that failed, and the full error output. A minimal config YAML that reproduces the problem is the fastest path to a fix.
