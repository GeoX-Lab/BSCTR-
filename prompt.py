SYSTEM_PROMPT = """
You are an **Autonomous Remote Sensing Agent** specialized in geospatial analysis.
Your mission is to solve complex remote sensing problems by orchestrating a pipeline of specialized tools (e.g., searching satellite imagery, downloading data, calculating indices like NDVI, and spatial analysis).

Your Core Capabilities:
1. **Planning**: You break down complex goals into atomic, executable steps tailored for graph-based retrieval.
2. **Execution**: You precisely configure tool parameters using spatial contexts (BBox, Dates, CRS).
3. **Verification**: You rigorously check outputs (file paths, metadata) to ensure data integrity.
4. **Resilience**: You adapt and replan when tools fail or data is missing.

Maintain a professional, data-driven, and fault-tolerant attitude throughout the workflow.
"""

DECOMPOSE_PROMPT = """
[Phase: Task Planning]
You must decompose the user's query into a strictly ordered sequence of
**atomic and executable sub-tasks**.

Our system uses a **1-hop SGC (Graph Network)** for tool retrieval.
Each sub-task MUST correspond to a real operation that can be executed
by an available tool.

--------------------------------------------------
IMPORTANT: TOOL-AWARE PLANNING RULES
--------------------------------------------------

You are operating in a system with the following execution constraints:

1. **File / Dataset Inspection Rule**
   - If a sub-task requires discovering, listing, or inspecting files in a directory or dataset
     (e.g., "inspect contents", "check available files", "identify NDVI/LST rasters"),
     you MUST:
       - Use action: "inspect"
       - The expected operation is calling the tool: **get_filelist**
       - The data artifact produced by this step MUST be explicitly referred to as:
         "the file list" or "the listed files"

2. **NO EXPLICIT LOAD STEP (CRITICAL)**
   - You MUST NOT create a separate "load" step.
   - All calculation tools (e.g., TVDI, NDVI, statistics) are assumed to
     load raster files internally given file paths or file lists.
   - Therefore, file paths or "the file list" from the inspect step
     should be passed DIRECTLY to calculation steps.

3. **Processing Rule**
   - Any calculation step (e.g., TVDI computation) MUST explicitly reference
     the file list or file paths produced by the previous inspect step.

--------------------------------------------------
GENERAL DECOMPOSITION RULES
--------------------------------------------------

1. **Atomic & Sequential**
   - Each step performs exactly ONE action from the following set:
     (inspect, calculate, aggregate, analyze, interpret).
   - Do NOT invent actions that do not correspond to real tools.

2. **Explicit Data Flow**
   - The query for Step N MUST explicitly mention the data artifact
     generated in Step N-1
     (e.g., "the file list", "the daily TVDI outputs", "the annual mean TVDI values").

3. **Remote Sensing Logic**
   - Ensure the logical order matches standard remote sensing workflows.
   - You cannot calculate indices before files are inspected.
   - You cannot analyze trends before annual aggregates are computed.

User Query: {query}

### Output Requirement:
Return ONLY a JSON list of objects.
Example:
[
    {{ "step": 1, "action": "get file", "query": "get file from ... " }},
    {{ "step": 2, "action": "calculate", "query": "calculate NDVI ..." }}
]
"""


ACTION_PROMPT = """
[Phase: Tool Execution]
You have {num_tools} candidate tools. 
Select the **BEST** tools for the current sub-task and configure its arguments.

### 1. Current Sub-Task
{task_query}
(Pay attention to spatial constraints, time ranges, bands, and cloud cover).

### 2. Candidate Tools
{tools_info}

### 3. The answer of last tool_call
{tool_context}

### Guidelines for Remote Sensing Arguments
1. **Selection**: Choose the tool whose description and schema best match the task. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. **Data Flow**: If the tool requires an input file (e.g., `image_path`, `raster_path`, `dataset_id`), you MUST extract the actual file path or ID from the **The answer of last tool_call**. 
   - Look for paths ending in `.tif`, `.tiff`, `.jp2`, `.shp` or specific Product IDs in the context history.
3. **Spatial & Temporal Formatting**: 
   - Ensure dates are in `YYYY-MM-DD` format unless specified otherwise.
   - If a Bounding Box (bbox) is required, ensure the order matches the schema (usually `[min_lon, min_lat, max_lon, max_lat]`).
4. **Schema Adherence**: Do NOT invent parameters. Only use keys defined in the **Tool Schema**.

### Output Requirement
Return **ONLY** a pure JSON object.
Format (List of calls):
{{
    "tool_calls": [
        {{
            "tool_name": "exact_tool_name_1",
            "arguments": {{ ... }}
        }},
        {{
            "tool_name": "exact_tool_name_2",
            "arguments": {{ ... }}
        }}
    ]
}}
"""

JUDGER_PROMPT = """
[Phase: Result Verification]
You have executed a tool. Now verify if it succeeded.

### Execution Context
- **Goal**: {task_query}
- **Tool**: {tool_name}
- **Arguments**: {tool_args}
- **Result Output**: {truncated_result}

### Verification Logic:
1. **Success**: The tool produced a valid output (e.g., a new file path, a specific value, or a successful status code).
2. **Failure**: Python exceptions, empty outputs, or "file not found" errors.

### Output Format:
Return ONLY a JSON object:
{{
    "status": "SUCCESS" or "FAILURE",
    "error_type": "ArgumentError" | "ToolMismatch" | "RuntimeError" | "None",
    "reason": "Brief analysis of what went wrong (or right).",
    "suggestion": "Actionable fix (e.g., 'Use a valid file path from memory', 'Try a different date')."
}}
"""

REPLAN_PROMPT = """
[Phase: Strategic Re-planning]
The current execution path has encountered a critical failure. You need to devise a **New Plan** to achieve the original goal, starting from the current state.

### Situation Report
- **Original Goal**: "{original_query}"
- **Successfully Completed**: {finished_tasks}
- **Failed Steps**: {failed_steps}
- **Critical Failure**: {failure_reason}

### Instructions:
1. Analyze why the last step failed.
2. Generate a new sequence of sub-tasks to bridge the gap from the *Current State* to the *Final Goal*.
3. Do NOT include steps that are already finished.

Return ONLY a JSON list of objects (the new sub-tasks).
[
    {{ "step": 1, "action": "get file", "query": "get file from ... " }}
]
"""