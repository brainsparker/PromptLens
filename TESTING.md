# Testing PromptLens

## Quick Start Test

### 1. Set up your API key

Create a `.env` file in the project root:

```bash
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

Or export it directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Get your API key from: https://console.anthropic.com/

### 2. Run a quick test

```bash
python3 -m promptlens run examples/configs/quicktest.yaml
```

This will:
- Test Claude 3.5 Sonnet on 5 customer support queries
- Use LLM-as-judge to score responses (1-5)
- Generate an HTML report with charts
- Take ~30-60 seconds and cost ~$0.01-0.02

### 3. View the results

```bash
# The path will be shown in the CLI output
open promptlens_results/latest/report.html
```

## What You'll See

The HTML report includes:
- Summary stats (cost, time, scores)
- Model performance card
- Score distribution chart
- Detailed results for each test case
  - Original query
  - Model response
  - Judge score & explanation

## Testing Different Scenarios

### Test Multiple Models

```bash
python3 -m promptlens run examples/configs/multi_model.yaml
```

Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY

### Test You.com Models

```bash
python3 -m promptlens run examples/configs/you_config.yaml
```

Requires: YOU_API_KEY (get from https://api.you.com/)

### Test Local Model (Ollama)

First install Ollama and pull a model:
```bash
# Install from https://ollama.ai
ollama pull llama3.1:8b

# Then run
python3 -m promptlens run examples/configs/local_model.yaml
```

### Test Code Generation

```bash
python3 -m promptlens run examples/configs/basic_config.yaml
# Edit the config to use code_generation.yaml as golden_set
```

## Troubleshooting

**"API key not found"**
- Make sure ANTHROPIC_API_KEY is set in .env or environment
- Run: `echo $ANTHROPIC_API_KEY` to verify

**"Command not found: promptlens"**
- Use: `python3 -m promptlens` instead
- Or add `~/Library/Python/3.9/bin` to PATH

**Import errors**
- Reinstall: `python3 -m pip install -e .`
- Check: `python3 -m pip list | grep promptlens`

**Slow execution**
- Increase `parallel_requests` in config
- Use smaller test sets for quick iterations

## Validating Files

Check if your golden set is valid:

```bash
python3 -m promptlens validate your_tests.yaml
```

List all past runs:

```bash
python3 -m promptlens list-runs
```

Export to different format:

```bash
python3 -m promptlens export <run_id> --format csv
```

## Next Steps

1. Create your own golden set (see examples/)
2. Customize the config for your use case
3. Compare multiple models
4. Integrate into CI/CD pipeline
