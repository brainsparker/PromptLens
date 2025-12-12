# Getting Started with PromptLens

## What is PromptLens For?

PromptLens helps you **objectively compare and improve LLM outputs** by:
- Testing multiple prompts to find which works best
- Comparing different models (Claude vs GPT vs Gemini) on the same tasks
- Catching regressions when you modify prompts
- Tracking cost and performance across models

**Instead of manually copy-pasting queries into ChatGPT/Claude, you:**
1. Define test cases once
2. Run them automatically across models
3. Get scored results with detailed reports

---

## Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/sparker/promptlens.git
cd promptlens

# Install
python3 -m pip install -e .

# Verify installation
python3 -m promptlens --version
```

---

## Your First Evaluation (5 minutes)

### Step 1: Get an API Key

Pick one provider to start:

**Option A: Anthropic Claude** (Recommended)
```bash
# Sign up: https://console.anthropic.com/
# Get your API key
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Option B: OpenAI**
```bash
# Sign up: https://platform.openai.com/
export OPENAI_API_KEY=sk-your-key-here
```

### Step 2: Run the Quick Test

```bash
python3 -m promptlens run examples/configs/quicktest.yaml
```

**What this does:**
- Tests Claude 3.5 Sonnet on 5 customer support questions
- Scores each response using LLM-as-judge (1-5 scale)
- Generates a beautiful HTML report
- Takes ~30 seconds, costs ~$0.01

### Step 3: View Your Results

```bash
# Open the report (path shown in terminal output)
open promptlens_results/latest/report.html
```

**You'll see:**
- 📊 Summary stats (average score, cost, time)
- 📈 Score distribution chart
- 📝 Each test case with:
  - The query you asked
  - Model's full response
  - Judge's score + explanation
  - Cost and latency

---

## Understanding the Results

### The HTML Report Explained

**Summary Cards** (top)
- **Total Cost**: How much you spent (in USD)
- **Total Time**: How long it took (milliseconds)
- **Test Cases**: Number of queries tested
- **Models Tested**: How many models you compared

**Model Performance** (middle)
- Average score for each model (out of 5)
- Cost breakdown per model
- Response time metrics

**Score Distribution Chart**
- How many responses got scores 1, 2, 3, 4, or 5
- Helps you see consistency

**Detailed Results** (bottom)
- Click to expand each test case
- See exactly what the model said
- Read judge's reasoning for the score

### What the Scores Mean

- **5/5**: Excellent - exceeded expectations
- **4/5**: Good - met expectations well
- **3/5**: Okay - met basic expectations
- **2/5**: Poor - significant issues
- **1/5**: Failed - completely missed the mark

---

## Common Use Cases

### Use Case 1: Find the Best Prompt

**Scenario**: You have a task, and you want to find which prompt works best.

**Steps:**

1. Create test cases for your task:

```yaml
# my_tests.yaml
name: "Email Response Test"
version: "1.0"

test_cases:
  - id: "test-001"
    query: "Write a professional email declining a meeting"
    expected_behavior: "Polite, professional, offers alternative, brief"

  - id: "test-002"
    query: "Write a professional email following up on a proposal"
    expected_behavior: "Friendly but professional, includes clear ask, concise"
```

2. Test different prompt versions:

```yaml
# config_v1.yaml - Simple prompt
models:
  - name: "Claude - Simple Prompt"
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    # Your prompt is in the query field of test cases
```

3. Modify your prompts in the test cases (add context, examples, constraints)

4. Run multiple times and compare scores:

```bash
python3 -m promptlens run config_v1.yaml
# Modify prompts in my_tests.yaml
python3 -m promptlens run config_v1.yaml
# Compare the two runs
```

5. Pick the prompt version with higher scores!

---

### Use Case 2: Choose the Right Model

**Scenario**: You want to know which model is best for your use case.

**Steps:**

1. Create a config testing multiple models:

```yaml
# compare_models.yaml
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

  - name: "Gemini 1.5 Pro"
    provider: google
    model: gemini-1.5-pro
    temperature: 0.7
    max_tokens: 1024

judge:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.3

output:
  formats: [html, json, csv]
```

2. Run the comparison:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...

python3 -m promptlens run compare_models.yaml
```

3. Look at the report to see:
   - Which model scored highest
   - Which was fastest
   - Which was cheapest
   - Quality/cost trade-offs

4. Make your decision based on data!

---

### Use Case 3: Regression Testing

**Scenario**: You're updating your prompts and want to make sure you don't break existing functionality.

**Steps:**

1. Create a comprehensive golden set covering all important cases:

```yaml
# regression_tests.yaml
name: "Product Chatbot - Regression Suite"
version: "1.0"

test_cases:
  - id: "product-001"
    query: "What are the features of the Pro plan?"
    expected_behavior: "Lists all Pro features accurately"

  - id: "product-002"
    query: "How do I cancel my subscription?"
    expected_behavior: "Clear cancellation steps, mentions refund policy"

  # ... 20-50 test cases covering all features
```

2. Run baseline before changes:

```bash
python3 -m promptlens run regression_config.yaml
# Note the run ID and average scores
```

3. Make your prompt changes

4. Run again:

```bash
python3 -m promptlens run regression_config.yaml
```

5. Compare results:
   - Did average score go down? (regression!)
   - Which specific test cases got worse?
   - Is the trade-off worth it?

6. Fix any regressions before deploying

---

### Use Case 4: Cost Optimization

**Scenario**: Your current model is expensive. Can you use a cheaper one without losing quality?

**Steps:**

1. Test current model vs alternatives:

```yaml
models:
  - name: "Current: GPT-4"
    provider: openai
    model: gpt-4

  - name: "Alternative: GPT-3.5 Turbo"
    provider: openai
    model: gpt-3.5-turbo

  - name: "Alternative: Claude Haiku"
    provider: anthropic
    model: claude-3-haiku-20240307
```

2. Run evaluation and check report:
   - Compare average scores
   - Look at cost column
   - Check latency differences

3. Example findings might be:
   - GPT-4: Avg score 4.5, $0.10 per run
   - GPT-3.5: Avg score 4.2, $0.01 per run (10x cheaper!)
   - Claude Haiku: Avg score 4.3, $0.005 per run (20x cheaper!)

4. Decide: Is a 0.3 point drop worth 10x cost savings?

---

## Creating Good Test Cases

### What Makes a Good Test Case?

**Good:**
```yaml
- id: "support-001"
  query: "I haven't received my order confirmation email"
  expected_behavior: "Ask for order number, check spam folder, offer to resend, provide support email"
  category: "order_issues"
```

**Why it's good:**
- Specific, realistic scenario
- Clear expectations (4 things to check)
- Represents real user queries

**Bad:**
```yaml
- id: "test-001"
  query: "Help with order"
  expected_behavior: "Be helpful"
```

**Why it's bad:**
- Too vague
- Unclear expectations
- Judge can't evaluate objectively

### How Many Test Cases?

- **Start small**: 3-5 test cases
- **Basic coverage**: 10-20 test cases
- **Comprehensive**: 30-50 test cases
- **Regression suite**: 50-100+ test cases

Start with 5 critical cases, then expand!

### Categories to Cover

For a well-rounded test set, include:

1. **Happy path** - Normal, expected queries
2. **Edge cases** - Unusual but valid scenarios
3. **Error handling** - Invalid input, missing information
4. **Ambiguity** - Questions that could be interpreted multiple ways
5. **Complexity** - Multi-part questions requiring nuanced answers

---

## Best Practices

### 1. Start Simple

Don't try to test everything at once:
```bash
# First run: Just 3 test cases, 1 model
# Takes 20 seconds, costs pennies
# Learn the workflow
```

### 2. Use Descriptive Names

```yaml
# Good
run_name: "Customer Support Bot - Pre-Launch Testing"

# Bad
run_name: "test"
```

### 3. Keep Test Cases Realistic

Use actual user queries from:
- Support tickets
- User research
- Analytics
- Your own product usage

### 4. Version Your Test Sets

```yaml
name: "Email Generator Tests"
version: "2.0"  # Track changes over time
```

### 5. Review Judge Scores

The judge isn't perfect! If you see unexpected scores:
- Read the explanation
- Adjust expectations if they were unclear
- Try a different judge model if needed

### 6. Track Runs Over Time

```bash
# List all past runs
python3 -m promptlens list-runs

# Keep a log of what changed between runs
# "Run abc123: Simplified system prompt"
# "Run def456: Added examples to prompt"
```

### 7. Export for Sharing

```bash
# Share CSV with team for analysis
python3 -m promptlens export <run_id> --format csv

# Share Markdown in GitHub PRs
python3 -m promptlens export <run_id> --format md
```

---

## Workflow Tips

### Daily Development Flow

```bash
# 1. Morning: Run regression suite
python3 -m promptlens run regression.yaml

# 2. Make changes to prompts
# (edit your app code or test cases)

# 3. Quick test on small set
python3 -m promptlens run quicktest.yaml

# 4. If good: Full regression again
python3 -m promptlens run regression.yaml

# 5. Review report, iterate
open promptlens_results/latest/report.html
```

### Before Deployment Flow

```bash
# 1. Full test suite
python3 -m promptlens run full_suite.yaml

# 2. Check all scores are acceptable
# Minimum threshold: 4.0/5.0 average

# 3. Export results for documentation
python3 -m promptlens export <run_id> --format md > evaluation_report.md

# 4. Commit report with code
git add evaluation_report.md
git commit -m "Evaluation: Avg score 4.2/5.0"
```

### Continuous Integration Flow

```yaml
# .github/workflows/prompt-evaluation.yml
name: Evaluate Prompts
on: [pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install PromptLens
        run: pip install -e .
      - name: Run Evaluation
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: python3 -m promptlens run tests/prompts.yaml
      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-report
          path: promptlens_results/latest/report.html
```

---

## Troubleshooting

### "API key not found"

```bash
# Check it's set
echo $ANTHROPIC_API_KEY

# Or add to .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### "Command not found: promptlens"

```bash
# Use Python module syntax instead
python3 -m promptlens run config.yaml
```

### Scores seem wrong

1. Check your `expected_behavior` - is it clear?
2. Try a different judge model
3. Read judge explanations to understand reasoning
4. Consider using a custom judge prompt

### It's slow

```yaml
# Increase parallel requests in config
execution:
  parallel_requests: 10  # Instead of 3
```

### It's expensive

1. Start with fewer test cases
2. Use cheaper models for testing (Haiku, GPT-3.5)
3. Use local models (Ollama) during development

---

## Next Steps

### You've Run Your First Evaluation - Now What?

1. **Create your own test cases**
   - Copy `examples/golden_sets/customer_support.yaml`
   - Replace with your actual use case
   - Start with 5 cases

2. **Compare models**
   - Add multiple models to config
   - See which works best for YOUR tasks

3. **Iterate on prompts**
   - Change prompts in test cases
   - Run multiple times
   - Track score improvements

4. **Build a regression suite**
   - Collect 20-30 important cases
   - Run before any changes
   - Protect quality over time

5. **Share results**
   - Export to Markdown for PRs
   - Share HTML reports with team
   - Track metrics over time

---

## Quick Reference

```bash
# Install
pip install -e .

# Validate test file
python3 -m promptlens validate my_tests.yaml

# Run evaluation
python3 -m promptlens run config.yaml

# List past runs
python3 -m promptlens list-runs

# Export to different format
python3 -m promptlens export <run_id> --format csv

# View help
python3 -m promptlens --help
```

---

## Real-World Example

Here's a complete example from start to finish:

### Problem
You're building a customer support chatbot and want to pick between Claude and GPT-4.

### Solution

**1. Create test cases** (10 minutes)
```yaml
# support_tests.yaml
name: "Support Chatbot Evaluation"
version: "1.0"

test_cases:
  - id: "refund-001"
    query: "I want a refund for my order #12345"
    expected_behavior: "Acknowledge request, ask for reason, explain 30-day policy, provide next steps"

  - id: "shipping-001"
    query: "Where is my package?"
    expected_behavior: "Ask for order number, explain tracking process, provide tracking link format"

  # ... 8 more cases covering common support scenarios
```

**2. Create config** (2 minutes)
```yaml
# support_config.yaml
golden_set: ./support_tests.yaml

models:
  - name: "Claude 3.5 Sonnet"
    provider: anthropic
    model: claude-3-5-sonnet-20241022

  - name: "GPT-4 Turbo"
    provider: openai
    model: gpt-4-turbo-preview

judge:
  provider: anthropic
  model: claude-3-5-sonnet-20241022

output:
  formats: [html, csv]
```

**3. Run evaluation** (1 minute)
```bash
python3 -m promptlens run support_config.yaml
```

**4. Review results** (5 minutes)
```
Results:
- Claude: Avg 4.3/5, Cost: $0.08, Time: 2100ms
- GPT-4: Avg 4.1/5, Cost: $0.15, Time: 3200ms

Decision: Use Claude (slightly better quality, cheaper, faster)
```

**5. Total time**: 18 minutes
**Cost**: $0.23
**Value**: Confident model choice backed by data

---

## FAQ

**Q: Can I test local models?**
A: Yes! Use Ollama with the `http` provider (see examples/configs/local_model.yaml)

**Q: How much does it cost?**
A: Depends on model and test count. Typical costs:
- Quick test (5 cases, 1 model): $0.01-0.02
- Full comparison (20 cases, 3 models): $0.50-1.00
- Comprehensive suite (50 cases, 3 models): $2-5

**Q: Can I use my own judge criteria?**
A: Yes! Use `custom_prompt` in judge config (see README)

**Q: Does it work offline?**
A: For local models (Ollama) yes. For API providers, you need internet.

**Q: Can I integrate with CI/CD?**
A: Yes! See the CI/CD workflow example above.

---

## Get Help

- **Issues**: https://github.com/sparker/promptlens/issues
- **Discussions**: https://github.com/sparker/promptlens/discussions
- **Examples**: See `examples/` directory in the repo

---

**Ready to start?** Run your first evaluation now:

```bash
export ANTHROPIC_API_KEY=your-key-here
python3 -m promptlens run examples/configs/quicktest.yaml
open promptlens_results/latest/report.html
```
