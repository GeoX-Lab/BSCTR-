# GeoPlan Benchmark Example Code

This directory contains example code for using GeoPlan Benchmark.

## Example List

### 1. example_generate_tasks.py

Task generation example demonstrating how to:
- Generate a single task
- Generate tasks in batch
- View task statistics

Run with:
```bash
python examples/example_generate_tasks.py
```

### 2. example_filter_tasks.py

Task filtering example demonstrating how to:
- Load raw tasks
- Execute filtering pipeline
- View filtering statistics
- Save filtered tasks
- Analyze tool importance

Run with:
```bash
python examples/example_filter_tasks.py
```

### 3. example_evaluate.py

Task evaluation example demonstrating how to:
- Load tasks
- Initialize evaluator
- Evaluate a single task
- View evaluation results

Run with:
```bash
python examples/example_evaluate.py
```

### 4. example_custom_agent.py

Custom agent example demonstrating how to:
- Create a custom agent
- Implement BaseAgent interface
- Add tools
- Integrate into evaluation pipeline

Run with:
```bash
python examples/example_custom_agent.py
```

### 5. example_custom_metric.py

Custom evaluation metric example demonstrating how to:
- Create a custom evaluation metric
- Compute evaluation scores
- Integrate into evaluation pipeline

Run with:
```bash
python examples/example_custom_metric.py
```

## Usage Instructions

1. Create and activate a virtual environment (venv or conda):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Optional, for command-line tools
```

3. Configure environment variables (create `.env` file):
```
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://api.openai.com/v1
GEMINI_API_KEY=your_key
```


## Notes

- Running task generation and evaluation examples requires LLM API calls and incurs costs
- Paths in example code may need adjustment based on actual situation
- It's recommended to run simple examples first to confirm environment configuration before running full pipeline

## Extension Examples

Based on these examples, you can:
- Create your own agent implementations
- Add new evaluation metrics
- Customize task generation pipeline
- Integrate into your own projects

For more details, please refer to:
- [Usage Guide](../docs/usage.md)
- [Architecture Documentation](../docs/architecture.md)
