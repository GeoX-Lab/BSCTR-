tool_agent_prompt = """
    你是一个工具选择智能体。下面是可用工具列表：
    {tool_list_str}
    任务描述：{task}
    请从工具中选择最合适的一个执行，并输出 JSON 格式：
    {{"tool_name": "XXX", "arguments": {{...}}}}
"""
graph_build_prompt = """
"""
decompose_prompt = """
    You are an advanced Graph-Aware Task Planner. Your goal is to decompose the User Query into a strictly ordered sequence of **fine-grained, atomic sub-tasks**.
        
        The system uses a 1-hop Graph Neural Network (SGC) for tool retrieval. This means a tool's vector representation is influenced by the tool used immediately before it.
        
        ### Critical Granularity Rules:
        1. **One Step, One Action**: Never combine retrieval and processing in one step. 
           - BAD: "Search for Apple's stock and calculate the PE ratio." (This skips the graph edge).
           - GOOD: Step 1 "Search for stock...", Step 2 "Calculate PE ratio using search results...".
        
        2. **Explicit Data Flow (Input-Action Pattern)**: 
           Since the "Child" tool aggregates features from the "Parent" tool, your query for the Child tool MUST explicitly mention the **output of the previous step**.
           - Format: "[Action verb] the [specific data] retrieved/generated from [previous step context]."
           
        3. **Bridge the Gap**: The query for Step N should sound like a bridge connecting Step N-1 to Step N.
           - If Step 1 is "Search", Step 2 shouldn't just be "Summarize". It must be "Summarize the **search results**".
           - This ensures the vector search matches the SGC aggregated features.

        User Query: {query}

        ### Output Format:
        Return ONLY a JSON list of objects.
        
        Example:
        [
            {{
                "step": 1, 
                "action": "search", 
                "query": "search for the Q3 financial report PDF of Tesla"
            }},
            {{
                "step": 2, 
                "action": "extract", 
                "query": "extract the revenue text from the **financial report PDF** found in step 1"
            }},
            {{
                "step": 3, 
                "action": "analyze", 
                "query": "analyze the sentiment of the **extracted revenue text**"
            }}
        ]
"""

tool_call_prompt = """
     You are an expert **Remote Sensing Geospatial Assistant**. 
        Your job is to extract precise arguments to execute the tool '{tool_name}' for a specific sub-task.

        ### Input Information
        1. **Current Sub-Task**: "{task_query}"
           (Pay attention to spatial constraints, time ranges, bands, and cloud cover).
        2. **Tool Description**: {tool_desc} 
           (Use this to understand what the tool expects).
        3. **Tool Schema** (Strict constraints):
           {schema}
        4. **Pipeline Context** (Previous results):
           {context}

        ### Guidelines for Remote Sensing Arguments
        1. **Data Flow**: If the tool requires an input file (e.g., `image_path`, `raster_path`, `dataset_id`), you MUST extract the actual file path or ID from the **Pipeline Context**. 
           - Look for paths ending in `.tif`, `.tiff`, `.jp2`, `.shp` or specific Product IDs in the context history.
        2. **Spatial & Temporal Formatting**: 
           - Ensure dates are in `YYYY-MM-DD` format unless specified otherwise.
           - If a Bounding Box (bbox) is required, ensure the order matches the schema (usually `[min_lon, min_lat, max_lon, max_lat]`).
        3. **Schema Adherence**: Do NOT invent parameters. Only use keys defined in the **Tool Schema**.
        
        ### Output Requirement
        Return **ONLY** a pure JSON object representing the arguments. No Markdown. No explanations.
        
        Example: {{"input_raster": "./downloads/sentinel2_ndvi.tif", "threshold": 0.4}}
"""