# PromptLens

**A lightweight, open-source evaluation harness for prompts, LLMs, and agent workflows.**

PromptLens runs golden test sets against multiple models, scores outputs using LLM-as-judge, tracks cost and latency, and generates beautiful visual reports—all locally, with no cloud dependencies.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

---

## Features

- **Multi-Provider Support** - Test Anthropic (Claude), OpenAI (GPT), Google (Gemini), You.com, and local models (Ollama, LM Studio)
- **Tool/Function Calling Evaluation** - Test tool usage with automatic + LLM judge scoring across 5 criteria
- **LLM-as-Judge Scoring** - Automated evaluation using another LLM with configurable criteria
- **Cost & Latency Tracking** - Monitor per-query costs and response times across models
- **Beautiful Reports** - Interactive HTML reports with charts, comparisons, and detailed results
- **Multiple Export Formats** - HTML, JSON, CSV, and Markdown outputs
- **Parallel Execution** - Async execution with configurable concurrency and retry logic
- **Strict Config Validation** - Early validation for model temperatures, token limits, retries, and output formats
- **Portable & Local** - No cloud backend, all data stays on your machine
- **Easy to Extend** - Plugin architecture for custom providers, judges, and exporters

---

## Installation

### Using pip (PyPI - Coming Soon)

```bash
pip install promptlens
```

### From Source

```bash
git clone https://github.com/sparker/promptlens.git
cd promptlens
pip install -e .

# Or with Poetry
poetry install
```

### Requirements

- Python 3.9+
- API keys for the providers you want to use (Anthropic, OpenAI, Google, You.com)

---

## Quick Start

### 1. Set up API keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
export YOU_API_KEY=...
```

### 2. Run an example evaluation

```bash
# Run the basic customer support evaluation
promptlens run examples/configs/basic_config.yaml
```

### 3. View the results

```bash
# Open the HTML report (path shown in CLI output)
open promptlens_results/latest/report.html
```

That's it! You've just evaluated an LLM against a golden test set.

**→ [Read the Complete Getting Started Guide](GETTING_STARTED.md)** for detailed workflows and use cases.

---

## Usage

### Creating a Golden Set

Golden sets are test cases in YAML or JSON format:

```yaml
name: "My Test Set"
description: "Testing customer support responses"
version: "1.0"

test_cases:
  - id: "test-001"
    query: "How do I reset my password?"
    expected_behavior: "Provide clear step-by-step instructions"
    category: "account_management"
    tags: ["password", "account"]

  - id: "test-002"
    query: "What's your refund policy?"
    expected_behavior: "Explain the 30-day refund policy clearly"
    category: "policy"
    tags: ["refund", "billing"]
```

Save as `my_tests.yaml`.

### Creating a Configuration File

```yaml
golden_set: ./my_tests.yaml

models:
  - name: "Claude 3.5 Sonnet"
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    temperature: 0.7
    max_tokens: 1024

  - name: "GPT-4 Turbo"
    provider: openai
    model: gpt-4-turbo-preview
    temperature: 0.7
    max_tokens: 1024

judge:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.3

execution:
  parallel_requests: 3
  retry_attempts: 3

output:
  directory: ./promptlens_results
  formats: [html, json, csv, md]
  run_name: "My Evaluation"
```

Save as `my_config.yaml`.

### Running the Evaluation

```bash
promptlens run my_config.yaml
```

### CLI Commands

```bash
# Run evaluation
promptlens run <config.yaml>

# Validate a golden set
promptlens validate <golden_set.yaml>

# List past runs
promptlens list-runs

# Export a run to different format
promptlens export <run_id> --format html

# Get help
promptlens --help
```

---

## Supported Providers

### Anthropic (Claude)

```yaml
models:
  - name: "Claude 3.5 Sonnet"
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    temperature: 0.7
    max_tokens: 1024
```

**Supported Models:**
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-haiku-20240307`

### OpenAI (GPT)

```yaml
models:
  - name: "GPT-4 Turbo"
    provider: openai
    model: gpt-4-turbo-preview
    temperature: 0.7
    max_tokens: 1024
```

**Supported Models:**
- `gpt-4-turbo-preview`, `gpt-4`, `gpt-3.5-turbo`
- `o1-preview`, `o1-mini`

### Google (Gemini)

```yaml
models:
  - name: "Gemini Pro"
    provider: google
    model: gemini-1.5-pro
    temperature: 0.7
    max_tokens: 1024
```

**Supported Models:**
- `gemini-1.5-pro`, `gemini-1.5-flash`
- `gemini-pro`

### You.com AI

```yaml
models:
  - name: "You.com GPT-4"
    provider: you
    model: gpt-4
    temperature: 0.7
    max_tokens: 1024
```

**Supported Models:**
- `gpt-4`, `claude-3-5-sonnet`, `llama-3-70b`
- Any model available through You.com's unified API

**Setup:**
1. Get API key from: https://api.you.com/
2. Set `YOU_API_KEY` environment variable
3. Use model names as specified in You.com docs

### Local Models (Ollama, LM Studio)

```yaml
models:
  - name: "Local Llama"
    provider: http
    model: llama3.1:8b
    temperature: 0.7
    max_tokens: 1024
    additional_params:
      endpoint: "http://localhost:11434/api/generate"
```

**Setup:**
1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama3.1:8b`
3. Use the `http` provider with the endpoint URL

---

## Configuration Reference

### Models

```yaml
models:
  - name: "Display Name"           # Human-readable name
    provider: anthropic             # anthropic, openai, google, http
    model: model-identifier         # Model ID
    temperature: 0.7                # 0.0-1.0
    max_tokens: 1024                # Maximum output tokens
    additional_params:              # Provider-specific params
      endpoint: "http://..."        # For HTTP provider
```

### Judge

```yaml
judge:
  provider: anthropic               # Provider for judge model
  model: claude-3-5-sonnet-20241022 # Judge model (typically Claude or GPT-4)
  temperature: 0.3                  # Lower for consistent scoring
  criteria:                         # Evaluation criteria
    - accuracy
    - helpfulness
    - safety
```

### Execution

```yaml
execution:
  parallel_requests: 3              # Concurrent API calls
  retry_attempts: 3                 # Retries for failed requests
  timeout_seconds: 60               # Request timeout
```

### Output

```yaml
output:
  directory: ./promptlens_results   # Output directory
  formats:                          # Export formats
    - html                          # Interactive report
    - json                          # Raw JSON data
    - csv                           # Flattened spreadsheet
    - md                            # Markdown summary
  run_name: "My Evaluation"         # Display name
```

---

## Examples

### Basic Single Model Evaluation

```bash
promptlens run examples/configs/basic_config.yaml
```

### Multi-Model Comparison

```bash
promptlens run examples/configs/multi_model.yaml
```

### Local vs Cloud Models

```bash
promptlens run examples/configs/local_model.yaml
```

See [examples/README.md](examples/README.md) for more details.

---

## Report Features

The HTML report includes:

- **Summary Dashboard** - Total cost, time, test cases, and models
- **Model Comparison Cards** - Side-by-side metrics for each model
- **Score Distribution Charts** - Visual breakdown of scores (1-5)
- **Detailed Test Results** - Expandable cards for each test case with:
  - Original query and expected behavior
  - Model responses
  - Judge scores and explanations
  - Cost and latency per response
- **Dark Theme** - Easy on the eyes with accent colors for data
- **Responsive Design** - Works on desktop and mobile

---

## Use Cases

### Prompt Iteration

Test different prompt versions to find the best performer:

1. Create test cases for your use case
2. Update your prompt
3. Run evaluation
4. Compare scores with previous run
5. Iterate

### Model Selection

Compare models before committing to one:

1. Add multiple models to config
2. Run the same test set against all models
3. Compare costs, latency, and quality scores
4. Make data-driven decision

### Regression Testing

Ensure prompt changes don't break existing behavior:

1. Maintain a golden set of important test cases
2. Run before and after making changes
3. Compare results to catch regressions
4. Integrate into CI/CD pipeline

### Agent Workflow Testing

Evaluate multi-step agent workflows:

1. Create test cases for agent tasks
2. Implement agent logic
3. Evaluate with PromptLens
4. Iterate on tools and prompting

### Tool/Function Calling Evaluation

Test how well models use tools and functions:

1. Define tools with JSON schema
2. Specify expected tool calls
3. Evaluate parameter correctness, tool selection, and efficiency
4. Get multi-criteria scores with detailed feedback

**Example test case:**
```yaml
- id: "tool-001"
  query: "What's the weather in San Francisco?"
  expected_behavior: "Call get_weather with location='San Francisco'"
  evaluation_mode: "tool_and_answer"

  tools:
    - name: "get_weather"
      description: "Get current weather"
      parameters:
        location:
          type: "string"
          required: true

  expected_tool_calls:
    - name: "get_weather"
      arguments:
        location: "San Francisco"
```

**Evaluation includes:**
- Automatic comparison (expected vs actual tool calls)
- Parameter correctness scoring (1-5)
- Tool selection accuracy (1-5)
- Tool usage efficiency (1-5)
- Final answer quality (1-5)

**Supported providers:** Anthropic Claude, OpenAI GPT (other providers will warn gracefully)

**Try it:**
```bash
promptlens run examples/configs/tool_evaluation.yaml
```

See [examples/golden_sets/tool_calling.yaml](examples/golden_sets/tool_calling.yaml) for complete examples.

---

## Advanced Usage

### Custom Judge Prompts

```yaml
judge:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  custom_prompt: |
    You are evaluating a coding assistant's response.

    Query: {query}
    Expected: {expected_behavior}
    Response: {response}

    Rate 1-5 based on code correctness, efficiency, and style.

    SCORE: [1-5]
    EXPLANATION: [Your reasoning]
```

### Provider-Specific Parameters

```yaml
models:
  - name: "GPT-4 with JSON mode"
    provider: openai
    model: gpt-4-turbo-preview
    additional_params:
      response_format: {"type": "json_object"}
```

### Parallel Execution Tuning

```yaml
execution:
  parallel_requests: 10   # Higher for faster execution
  retry_attempts: 5       # More retries for flaky APIs
  timeout_seconds: 120    # Longer timeout for slow models
```

---

## Architecture

```
promptlens/
├── models/          # Pydantic data models
├── providers/       # LLM provider implementations
├── loaders/         # Golden set loaders (JSON/YAML)
├── runners/         # Orchestration and execution
├── judges/          # LLM-as-judge scoring
├── exporters/       # Report generators
├── utils/           # Utilities (cost, retry, timing)
└── templates/       # HTML report templates
```

**Key Design Principles:**
- **Plugin Architecture** - Easy to add new providers, judges, exporters
- **Async-First** - Parallel execution for speed
- **Type-Safe** - Pydantic models throughout
- **Modular** - Each component is independent and testable

---

## Extending PromptLens

### Custom Provider

```python
from promptlens.providers.base import BaseProvider
from promptlens.models.result import ModelResponse

class MyProvider(BaseProvider):
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        # Your implementation
        pass

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0

    @property
    def provider_name(self) -> str:
        return "my_provider"

# Register it
from promptlens.providers.factory import register_provider
register_provider("my_provider", MyProvider)
```

### Custom Judge

```python
from promptlens.judges.base import BaseJudge
from promptlens.models.result import JudgeScore

class RuleBasedJudge(BaseJudge):
    async def evaluate(self, test_case, model_response) -> JudgeScore:
        # Your scoring logic
        score = self.calculate_score(model_response.content)
        return JudgeScore(
            score=score,
            explanation="Rule-based evaluation",
            judge_model="rule-based",
            judge_provider="custom"
        )
```

---

## Roadmap

- [x] Multi-provider support (Anthropic, OpenAI, Google, HTTP)
- [x] LLM-as-judge scoring
- [x] HTML reports with charts
- [x] JSON/CSV/Markdown export
- [x] Parallel execution with retry logic
- [ ] Multi-judge consensus scoring
- [ ] Synthetic test case generation
- [ ] Cross-run comparison and tracking
- [ ] GitHub Action for CI/CD
- [ ] Web UI for report exploration
- [ ] Embedding-based similarity scoring
- [ ] Custom plugin marketplace

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Inspired by the need for simple, local LLM evaluation tools
- Built with [Anthropic](https://anthropic.com), [OpenAI](https://openai.com), and [Google AI](https://ai.google.dev) APIs
- Uses [Rich](https://rich.readthedocs.io/) for beautiful CLI output
- Charts powered by [Chart.js](https://www.chartjs.org/)

---

## Support

- **Issues**: https://github.com/sparker/promptlens/issues
- **Discussions**: https://github.com/sparker/promptlens/discussions
- **Email**: sparker@example.com

---

**Made with ❤️ for the LLM developer community**
