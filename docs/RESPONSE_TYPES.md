```markdown
# Response Analysis Types

The Enhanced Multi-AI Middleware supports various types of analysis and response improvements. This document explains each option in detail.

## Basic Response Types

### 1. Initial Responses (`initial_only`)

Returns only the raw, unmodified responses from each selected AI model without any analysis or improvement.

```python
result = middleware.process_query(
    "What is quantum computing?",
    display_improvement_types="initial_only"
)
```

### 2. Cross-Model Analysis

One AI model analyzes and improves another model's response:

| Option | Description |
|--------|-------------|
| `claude_analyzed_by_openai` | GPT analyzes Claude's response and suggests improvements |
| `openai_analyzed_by_claude` | Claude analyzes GPT's response and suggests improvements |
| `claude_analyzed_by_perplexity` | Perplexity analyzes Claude's response |
| `openai_analyzed_by_perplexity` | Perplexity analyzes GPT's response |
| `perplexity_analyzed_by_claude` | Claude analyzes Perplexity's response |
| `perplexity_analyzed_by_openai` | GPT analyzes Perplexity's response |

Example:
```python
result = middleware.process_query(
    "Explain the concept of time dilation in relativity",
    display_improvement_types="claude_analyzed_by_openai"
)
```

### 3. Self-Analysis

Models analyze and improve their own responses:

| Option | Description |
|--------|-------------|
| `claude_analyzed_by_self` | Claude analyzes its own response |
| `openai_analyzed_by_self` | GPT analyzes its own response |
| `perplexity_analyzed_by_self` | Perplexity analyzes its own response |
| `all_self_analysis` | All models analyze their own responses |

Example:
```python
result = middleware.process_query(
    "What are the key factors in climate change?",
    display_improvement_types="all_self_analysis"
)
```

### 4. Multi-Model Analysis

Multiple models analyze a single model's response:

| Option | Description |
|--------|-------------|
| `claude_analyzed_by_multiple` | Both GPT and Perplexity analyze Claude's response |
| `openai_analyzed_by_multiple` | Both Claude and Perplexity analyze GPT's response |
| `perplexity_analyzed_by_multiple` | Both Claude and GPT analyze Perplexity's response |

Example:
```python
result = middleware.process_query(
    "What are the ethical implications of AI?",
    display_improvement_types="claude_analyzed_by_multiple"
)
```

### 5. Comprehensive Analysis

| Option | Description |
|--------|-------------|
| `all` | Performs all possible analyses (self-analysis and cross-analysis) |
| `cross_only` | Only performs cross-model analyses |
| `self_only` | Only performs self-analyses |

## Multiple Analysis Types

You can combine multiple analysis types by providing a list:

```python
result = middleware.process_query(
    "What is consciousness?",
    display_improvement_types=[
        "claude_analyzed_by_openai", 
        "openai_analyzed_by_claude"
    ]
)
```

## Analysis Process

Each analysis follows this process:

1. Generate initial responses from selected models
2. Analyze responses based on the selected analysis types
3. Generate improved responses using the analysis findings
4. Extract potential follow-up questions

The analysis identifies:
- Potential improvements to the response
- Missing information
- Possible follow-up questions

## UI Display Options

In the Streamlit UI, you can select which analyses to display:
- Basic Options: Simplified set of common analysis combinations
- Detailed Options: Complete control over specific analysis types
```

#### docs/API.md
```markdown
# API Documentation

## Core Classes

### `EnhancedMultiAIMiddleware`

The main class that handles interactions between different AI models.

```python
from enhanced_middleware import EnhancedMultiAIMiddleware

middleware = EnhancedMultiAIMiddleware(claude_client, openai_client, perplexity_client=None)
```

#### Parameters

- `claude_client`: An instance of the Anthropic client
- `openai_client`: An instance of the OpenAI client
- `perplexity_client` (optional): An instance of the Perplexity client (using OpenAI interface)

#### Methods

##### `process_query`

```python
result = middleware.process_query(
    user_input,
    show_comparison=False,
    selected_models=None,
    display_improvement_types="claude_analyzed_by_openai"
)
```

**Parameters:**
- `user_input` (str): The query text to process
- `show_comparison` (bool): Whether to return detailed comparison results
- `selected_models` (List[str], optional): List of models to use (e.g., ["claude", "openai"])
- `display_improvement_types` (Union[str, List[str]]): Analysis types to perform

**Returns:**
- If `show_comparison=True`: Detailed dictionary with all analyses and responses
- If `show_comparison=False`: Dictionary with the best response and metadata

##### `set_budget_limit`

```python
middleware.set_budget_limit(limit=5.0)
```

Sets a budget limit for API calls in USD.

##### `get_usage_summary`

```python
usage = middleware.get_usage_summary()
```

Returns token usage statistics.

##### `save_usage_data`

```python
middleware.save_usage_data()
```

Saves token usage data to disk.

### `AnalysisTask`

Represents a single analysis task.

```python
from enhanced_middleware import AnalysisTask

task = AnalysisTask(analyzer="openai", analyzed="claude", task_type="external")
```

#### Parameters

- `analyzer` (str): The model performing the analysis
- `analyzed` (str): The model being analyzed
- `task_type` (str): Type of analysis ("external" or "self")
- `display_type` (str, optional): UI display identifier

### `ImprovementPlan`

Defines and manages improvement plans.

```python
from enhanced_middleware import ImprovementPlan

plan = ImprovementPlan()
plan.add_task(task)
```

## Response Format

The `process_query` method returns a structured response:

```json
{
  "final_response": "The best response text...",
  "processing_time": 10.5,
  "token_usage": {
    "combined": {
      "prompt_tokens": 500,
      "completion_tokens": 300,
      "total_tokens": 800,
      "estimated_cost": 0.016
    },
    "claude_usage": {...},
    "openai_usage": {...},
    "perplexity_usage": {...}
  },
  "selected_models": ["claude", "openai"],
  "analysis_tasks": ["claude_analyzed_by_openai", ...],
  "display_improvement_types": "claude_analyzed_by_openai",
  "full_analysis": {
    "initial_responses": {...},
    "analyses": {...},
    "improved_responses": {...},
    "follow_up_questions": {...}
  }
}
```

With `show_comparison=True`, the full result structure is returned directly.
```
