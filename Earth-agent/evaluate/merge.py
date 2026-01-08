import json
from collections import OrderedDict

def merge_adjacent_same_tools(input_file_path, output_file_path):
    """
    Merge consecutive tool calls with the same name into a batch call.
    Adapted for structure: {'tool_name': '...', 'arguments': {...}}
    """
    # 读取数据
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    merged_data = []

    for question_data in data:
        # 保持其他字段不变 (如 question_index)
        new_entry = question_data.copy()
        
        tool_calls = question_data.get('tool_calls', [])
        
        # 执行合并逻辑
        if tool_calls:
            new_entry['tool_calls'] = merge_consecutive_same_tools(tool_calls)
        else:
            new_entry['tool_calls'] = []

        merged_data.append(new_entry)

    # 保存结果
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Merge completed! Processed {len(merged_data)} entries.")
    print(f"Saved to: {output_file_path}")

def merge_consecutive_same_tools(tool_calls):
    """
    具体的合并算法逻辑
    """
    if not tool_calls:
        return []

    merged = []
    i = 0
    n = len(tool_calls)

    while i < n:
        current_tool = tool_calls[i]
        current_name = current_tool.get('tool_name')

        # 向后查找连续的同名工具
        j = i + 1
        while j < n and tool_calls[j].get('tool_name') == current_name:
            j += 1

        # 判断是一次调用还是多次连续调用
        if j - i == 1:
            # 只有一个，直接添加，保持原样
            merged.append(current_tool)
        else:
            # 是一组连续调用，进行合并
            group = tool_calls[i:j]
            merged_tool = merge_tool_group(group)
            merged.append(merged_tool)

        # 更新指针
        i = j

    return merged

def merge_tool_group(group):
    """
    将一组同名工具调用的参数合并为列表
    """
    if not group:
        return None

    first_tool = group[0]
    
    # 初始化合并后的结构
    merged_tool = {
        'tool_name': first_tool.get('tool_name'),
        'arguments': OrderedDict()
    }

    # --- 处理 arguments 合并 ---
    # 获取第一个工具调用的所有参数名
    original_args = first_tool.get('arguments', {})
    if isinstance(original_args, dict):
        param_keys = list(original_args.keys())

        # 遍历每个参数名，将组内所有工具的对应值收集到列表中
        for key in param_keys:
            merged_tool['arguments'][key] = []
            for tool in group:
                args = tool.get('arguments', {})
                # 使用 .get 以防某些后续调用缺少该参数（虽然理论上不应该发生）
                merged_tool['arguments'][key].append(args.get(key))

    # --- 处理 output 合并 (如果存在) ---
    # 检查第一个元素是否有 output 字段，如果有，则合并 output
    if 'output' in first_tool:
        merged_tool['output'] = [t.get('output') for t in group]

    return merged_tool

# ================= 使用示例 =================
if __name__ == "__main__":
    # 请替换为你的实际文件路径
    input_json_path = "/media/csudxy0218/ZL/AgentToolmem/Earth-agent/evaluate/Qwen3-32B/extracted_tool_calls.json" 
    output_json_path = "/media/csudxy0218/ZL/AgentToolmem/Earth-agent/evaluate/Qwen3-32B/merged_tool_calls.json"
    try:
        merge_adjacent_same_tools(input_json_path, output_json_path)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {input_json_path}，请确保路径正确。")