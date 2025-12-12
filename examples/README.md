# PromptLens Examples

This directory contains example golden sets and configuration files to help you get started with PromptLens.

## Golden Sets

Golden sets are collections of test cases used to evaluate LLM performance. Each test case includes:
- A query/prompt
- Expected behavior description
- Optional metadata (category, tags, reference answers)

### Available Golden Sets

1. **`customer_support.yaml`** - Customer service chatbot evaluation
   - 5 test cases covering common support scenarios
   - Topics: password reset, refunds, shipping, subscriptions, login issues

2. **`code_generation.yaml`** - Code generation capabilities
   - 5 test cases for different programming languages and tasks
   - Topics: algorithms, React components, SQL queries, decorators

3. **`summarization.yaml`** - Text summarization and explanation
   - 5 test cases for different summarization styles
   - Topics: bullet points, one-sentence summaries, comparisons, explanations

## Configuration Files

Configuration files define how evaluations are run, including which models to test, judge settings, and output options.

### Available Configurations

1. **`basic_config.yaml`** - Single model evaluation
   - Tests Claude 3.5 Sonnet on customer support cases
   - Good for quick testing and development
   - Exports: HTML + JSON

2. **`multi_model.yaml`** - Multi-model comparison
   - Compares Claude, GPT-4, and Gemini on code generation
   - Comprehensive output formats (HTML, JSON, CSV, Markdown)
   - Higher parallelism for faster execution

3. **`local_model.yaml`** - Cloud vs local model comparison
   - Compares Claude with local Ollama models
   - Useful for cost-effective testing and privacy-sensitive use cases
   - Requires Ollama running locally

## Quick Start

### 1. Set up API keys

```bash
# Copy the example environment file
cp ../.env.example .env

# Edit .env and add your API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

### 2. Run a basic evaluation

```bash
# From the project root
promptlens run examples/configs/basic_config.yaml
```

### 3. View the results

```bash
# Open the HTML report
open promptlens_results/latest/report.html
```

## Customizing Golden Sets

Create your own golden set by following this YAML format:

```yaml
name: "My Custom Test Set"
description: "Description of what this tests"
version: "1.0"

test_cases:
  - id: "test-001"
    query: "Your test query or prompt"
    expected_behavior: "What you expect the model to do"
    category: "optional-category"
    tags: ["tag1", "tag2"]
```

Save as `my_tests.yaml` and reference it in your config:

```yaml
golden_set: ./my_tests.yaml
```

## Customizing Configurations

Key configuration sections:

### Models
```yaml
models:
  - name: "Display Name"
    provider: anthropic  # or openai, google, http
    model: model-identifier
    temperature: 0.7
    max_tokens: 1024
```

### Judge
```yaml
judge:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.3  # Lower temperature for more consistent scoring
```

### Execution
```yaml
execution:
  parallel_requests: 3  # Number of concurrent API calls
  retry_attempts: 3
  timeout_seconds: 60
```

### Output
```yaml
output:
  directory: ./promptlens_results
  formats: [html, json, csv, md]
  run_name: "My Evaluation Run"
```

## Tips

- **Start small**: Begin with 3-5 test cases to iterate quickly
- **Use descriptive expected behaviors**: The judge uses this to evaluate responses
- **Adjust temperature**: Lower (0.3) for consistent outputs, higher (0.9) for creative tasks
- **Monitor costs**: Check the HTML report for per-model cost breakdowns
- **Iterate on prompts**: Use results to improve your prompt engineering

## Local Models (Ollama)

To use local models:

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.1:8b`
3. Use the `http` provider in your config with `endpoint: "http://localhost:11434/api/generate"`

## Need Help?

- Check the main README: [../README.md](../README.md)
- Read the docs: [../docs/](../docs/)
- Report issues: https://github.com/sparker/promptlens/issues
