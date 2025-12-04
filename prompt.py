tool_agent_prompt = """
    你是一个工具选择智能体。下面是可用工具列表：
    {tool_list_str}
    任务描述：{task}
    请从工具中选择最合适的一个执行，并输出 JSON 格式：
    {{"tool_name": "XXX", "arguments": {{...}}}}
"""
graph_build_prompt = """
    You are an expert in remote-sensing workflow construction.
    
    Your task:
    Given the tool definitions and candidate downstream tools,
    infer correct workflow edges.
    
    Rules for edges (very important):
    - Only create edges when the OUTPUT of one tool is REQUIRED as INPUT of another
    - The connection must reflect a real remote-sensing processing pipeline
    - Do NOT create edges only because tools are similar
    - Avoid connecting tools that compute similar indices or statistics
    - Prefer the semantic order:
        data_loading → preprocessing → index/segmentation/time_series → analysis → reporting
    
    Examples of VALID edges:
    - RemoteSAM → ComputeArea
      Because RemoteSAM outputs a segmentation mask, which is required as input for area calculation.
    
    - ComputeNDVI → TrendAnalysis
      Because the NDVI time series is required as input for trend analysis.
    
    Examples of INVALID edges:
    - RemoteSAM → SAM2 (both are segmentation models)
    - NDVI → EVI (two indices of similar type)
    - MannKendallTest → SensSlope (two statistical trend estimation methods)
    
    Output format (strict JSON):
    {
      "edges": [
        {
          "start_tool": "ToolA",
          "end_tool": "ToolB",
          "messages": "why this connection exists"
        }
      ]
    }
    
    If NO valid edges exist for the current tool, output:
    {"edges": []}
"""