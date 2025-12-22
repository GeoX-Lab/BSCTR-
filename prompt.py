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
You must decompose the user's query into a **strictly ordered sequence of atomic, executable sub-tasks**.
Our system uses a **1-hop SGC (Graph Network)** for tool retrieval. Each sub-task MUST correspond to a **real operation** that can be executed by an available tool.

--------------------------------------------------
IMPORTANT: TOOL-AWARE PLANNING RULES
--------------------------------------------------

You are operating in a system with the following execution constraints:

1. **File / Dataset Inspection Rule**
   - If a sub-task requires discovering, listing, or inspecting files in a directory or dataset
     (e.g., "inspect contents", "check available files", "identify NDVI/LST rasters"),
     you MUST:
       - Use "inspect the dataset directory to discover available data files" or "list available raster files in the dataset directory" or "identify what data files exist in the dataset directory" to search get_filelist tool.
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
    
You are aware that the system provides specialized tools for
numeric aggregation and statistical analysis, including:

- computing mean values from raster datasets
- averaging batches of images
- aggregating multiple GeoTIFF files into a single raster
- computing pixel ratios based on thresholds
- calculating numeric differences between results

You should actively decompose tasks to leverage these tools instead of performing implicit or vague analysis steps.

--------------------------------------------------
GENERAL DECOMPOSITION RULES
--------------------------------------------------

1. **Atomic & Sequential**
   - Each step performs exactly ONE action 
   - Do NOT invent actions that do not correspond to real tools.

2. **Explicit Data Flow**
   - The query for Step N MUST explicitly mention the data artifact
     generated in Step N-1
     (e.g., "the file list", "the daily TVDI outputs", "the annual mean TVDI values").

3. **Remote Sensing Logic**
   - Ensure the logical order matches standard remote sensing workflows.
   - You cannot calculate indices before files are inspected.
   - You cannot analyze trends before annual aggregates are computed.

4. **Raster-to-Scalar Rule**
   - If an analysis requires numeric values (e.g., annual means),
     there MUST be an explicit aggregation step that converts raster outputs
     into numeric values using a valid tool.
     
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
1. **Selection**: Choose the tool whose description and schema best match the task. If a tool produces a result like "Result saved at /path/to/file", use the full path "/path/to/file" in subsequent calls.
2. **Data Flow**: If the tool requires an input file (e.g., `image_path`, `raster_path`), you MUST extract the actual file path from **The answer of last tool_call**.
   - Look for file paths ending in `.tif`, `.tiff`, `.jp2`, `.shp`, or Product IDs.
3. **Spatial & Temporal Formatting**: Ensure that all dates follow `YYYY-MM-DD` format and that Bounding Boxes are correctly formatted (e.g., `[min_lon, min_lat, max_lon, max_lat]`).
4. **Schema Adherence**: Never invent parameters. Only use keys defined in the **Tool Schema**.

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
You are performing **Strategic Re-planning** after a critical failure.
This phase follows the SAME planning principles, constraints, and rules
as the original **Task Planning (DECOMPOSE)** phase.

You must generate a **new, strictly ordered sequence of atomic and executable sub-tasks**
that continue from the CURRENT EXECUTION STATE, NOT from the beginning.

Our system uses a **1-hop SGC (Graph Network)** for tool retrieval.
Each sub-task MUST correspond to a **real operation** that can be executed
by an available tool.

--------------------------------------------------
CURRENT EXECUTION STATE (FROM WORKING MEMORY)
--------------------------------------------------

### Original User Goal
"{original_query}"

### Successfully Completed Tasks (FINAL & REUSABLE)
{finished_tasks}

- These steps are VALID and COMPLETE.
- Their outputs already exist in the system.
- You MUST assume these results can be directly reused.

### Available Data Artifacts (IMPORTANT)
The following intermediate outputs are already stored in Working Memory
and MUST be reused instead of recomputed:
{working_memory_outputs}

### Failed Steps
{failed_steps}

### Failure Reason
{failure_reason}

--------------------------------------------------
IMPORTANT: TOOL-AWARE PLANNING RULES (STILL APPLY)
--------------------------------------------------

You are operating under the SAME execution constraints as in Task Planning:

1. **File / Dataset Inspection Rule**
   - Only introduce an "inspect" step (calling **get_filelist**) IF and ONLY IF:
     - the required files are NOT already available in Working Memory.
   - You are STRICTLY FORBIDDEN from re-inspecting datasets
     whose results already exist.

2. **NO EXPLICIT LOAD STEP (CRITICAL)**
   - You MUST NOT introduce any "load" step.
   - Existing file paths or file lists from Working Memory
     MUST be passed DIRECTLY into calculation steps.

3. **Processing Rule**
   - Any calculation step MUST explicitly reference
     existing file paths, file lists, or outputs
     produced by completed tasks.

--------------------------------------------------
GENERAL RE-PLANNING RULES (ALIGNED WITH DECOMPOSE)
--------------------------------------------------

1. **Atomic & Sequential**
   - Each step performs exactly ONE action.
   - Do NOT invent actions that do not correspond to real tools.

2. **Explicit Data Flow**
   - Each new step MUST explicitly reference
     the data artifact produced by:
       - either a previously completed task, or
       - the immediately preceding re-planned step.
   - Examples:
     - "use the existing NDVI rasters"
     - "based on the annual mean NDVI values"

3. **Remote Sensing Logic**
   - The workflow order MUST still follow standard remote sensing logic.
   - You cannot recompute indices that already exist.
   - You cannot analyze trends before required aggregates exist.

4. **Raster-to-Scalar Rule**
   - If the remaining analysis requires numeric values:
     - You MUST include an explicit aggregation step
       using a valid tool.
     - Do NOT assume raster values are already numeric.

--------------------------------------------------
RE-PLANNING OBJECTIVE
--------------------------------------------------

Your task is to:
- Continue from the CURRENT STATE
- Repair the failed part of the workflow
- Reach the FINAL GOAL with the MINIMUM necessary new steps

You are STRICTLY FORBIDDEN from:
- Restarting the workflow
- Repeating completed steps
- Ignoring existing data artifacts

--------------------------------------------------
OUTPUT REQUIREMENT
--------------------------------------------------

Return ONLY a JSON list of objects (the new sub-tasks).
[
    {{ "step": 1, "action": "get file", "query": "get file from ... " }}
]
"""