import json
import ast
import os


def safe_parse_content(content):
    """
    尝试将字符串内容解析为 Python 字典或 JSON 对象。
    如果失败，返回原始内容。
    """
    if not isinstance(content, str):
        return content

    content = content.strip()

    # 尝试 JSON
    try:
        return json.loads(content)
    except:
        pass

    # 尝试 Python 字面量 (处理单引号的情况)
    try:
        parsed = ast.literal_eval(content)
        return parsed
    except:
        pass

    return content


def process_assistant_message(content):
    """
    处理 Assistant 的消息，尝试分离 thought 和 tool_calls
    """
    result = {
        "role": "assistant",
        "thought": None,
        "tool_calls": None,
        "content": None  # 用于兜底
    }

    # 常见模式 1: [Tool Selection] 后面跟着字典
    # 或者内容中包含 'tool_calls' 字符串
    if "[Tool Selection]" in content or "tool_calls" in content:
        # 尝试分离标签和字典
        lines = content.split('\n', 1)
        header = lines[0].strip()
        result['thought'] = header

        body_str = lines[1] if len(lines) > 1 else content
        parsed_body = safe_parse_content(body_str)

        if isinstance(parsed_body, dict):
            # === 修复点在这里 ===
            # 先获取 raw_tool_calls，并确保它不是 None 且是列表
            raw_tool_calls = parsed_body.get('tool_calls')

            if raw_tool_calls and isinstance(raw_tool_calls, list):
                result['tool_calls'] = []
                for tc in raw_tool_calls:
                    # 再次防御，防止 list 里混入 None
                    if not tc: continue

                    result['tool_calls'].append({
                        "type": "function",
                        "function": {
                            "name": tc.get('tool_name'),
                            "arguments": tc.get('arguments')
                        }
                    })
            # 如果 tool_calls 是空的或者 None，或者解析出来的不是列表
            else:
                # 这种情况下，可能只是普通的思考，或者格式不标准的输出
                # 我们保留 thought，不强行塞 tool_calls
                pass

    # 常见模式 2: [Task Decompose] 或普通思考
    elif content.startswith("["):
        result['thought'] = content
    else:
        # 普通回复
        result['content'] = content
        del result['thought']
        del result['tool_calls']

    # 清理 None 字段
    return {k: v for k, v in result.items() if v is not None}


def convert_jsonl_to_dataset(input_path, output_path):
    final_dataset = {}

    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return

    print(f"正在读取: {input_path} ...")

    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                original_entry = json.loads(line)

                # 初始化新的条目结构
                new_entry = {
                    "tools": None,
                    "files": None,
                    "dialogs": []
                }

                # 处理 history 转换为 dialogs
                history = original_entry.get('history', [])

                for step in history:
                    role = step.get('role')
                    raw_content = step.get('content', '')

                    if role == 'user':
                        new_entry['dialogs'].append({
                            "role": "user",
                            "content": raw_content
                        })

                    elif role == 'assistant':
                        # 智能处理 Assistant 的输出
                        processed_msg = process_assistant_message(raw_content)
                        new_entry['dialogs'].append(processed_msg)

                    elif role == 'tool':
                        # 处理 Tool 输出
                        tool_name = step.get('name', 'unknown_tool')

                        # 解析工具返回的内容
                        parsed_content = safe_parse_content(raw_content)

                        # 尝试模仿 structure (type + content)
                        final_tool_content = parsed_content
                        if isinstance(parsed_content, list):
                            # 如果全都是字符串，标记为 list(str)
                            is_str_list = all(isinstance(x, str) for x in parsed_content)
                            final_tool_content = {
                                "type": "list(str)" if is_str_list else "list",
                                "content": parsed_content
                            }

                        new_entry['dialogs'].append({
                            "role": "tool",
                            "name": tool_name,
                            "content": final_tool_content
                        })

                # 使用字符串索引 "1", "2"...
                final_dataset[str(idx + 1)] = new_entry

            except json.JSONDecodeError:
                print(f"解析第 {idx + 1} 行失败")
            except Exception as e:
                print(f"处理第 {idx + 1} 行时发生未知错误: {e}")

    # 写入最终的 JSON 文件
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在写入: {output_path} ...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(final_dataset, f_out, indent=4, ensure_ascii=False)

    print("转换完成！")
if __name__ == "__main__":
    # === 配置输入输出路径 ===
    input_file = "/media/csudxy0218/ZL/AgentToolmem/Earth-agent/evaluate/qwen3-max/qwen3-max_outputs.jsonl"  # 你的原始 jsonl 文件路径
    output_file = ("./qwen3-max/qwen3-max.json")  # 你想保存的 json 文件路径
    # ========================

    convert_jsonl_to_dataset(input_file, output_file)