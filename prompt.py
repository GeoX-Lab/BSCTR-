tool_agent_prompt = """
    你是一个工具选择智能体。下面是可用工具列表：
    {tool_list_str}
    任务描述：{task}
    请从工具中选择最合适的一个执行，并输出 JSON 格式：
    {{"tool_name": "XXX", "arguments": {{...}}}}
"""

tool_graph_build = """
You are a remote-sensing tool graph builder. Given remote-sensing tool information (e.g., name, description, inputs, outputs), you need to generate the connections (edges) between tools. These relationships can be of the type “output -> input,” meaning the output of one tool serves as the input to another tool.
Based on each tool’s inputs and outputs, infer the logical connections among tools and produce the connection information for each tool.
Below is the tool information. Please construct the tool-graph edges for each tool accordingly.

## Output Format
{
    "edges": [
        {
            "start_tool": "NDVI Calculator",
            "end_tool": "Land Surface Temperature (LST) Calculator",
            "messages": "The output of the NDVI calculator (NDVI image) can be used as an input to the LST calculator."
        },
        {
            "start_tool": "NDVI Calculator",
            "end_tool": "Fire Detector",
            "messages": "The NDVI image can be used as an input to the Fire Detector to help identify wildfire areas."
        }
    ]
}
"""