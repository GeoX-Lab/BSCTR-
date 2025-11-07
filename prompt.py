tool_agent_prompt = """
    你是一个工具选择智能体。下面是可用工具列表：
    {tool_list_str}
    任务描述：{task}
    请从工具中选择最合适的一个执行，并输出 JSON 格式：
    {{"tool_name": "XXX", "arguments": {{...}}}}
"""