import json
import ast
import os
import re  # 新增正则模块，用于提取 <think>

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

    # 尝试 Python 字面量
    try:
        parsed = ast.literal_eval(content)
        return parsed
    except:
        pass

    return content

def process_assistant_message(content):
    """
    处理 Assistant 的消息，适配多种格式：
    1. <think>...</think> Final Answer
    2. [Tool Selection] {json}
    3. [Subtask_verify] {json}
    4. [RePlan] [list]
    5. [Task Decompose] [list]
    """
    result = {
        "role": "assistant",
        "thought": None,
        "tool_calls": None,
        "content": None
    }
    
    if not content:
        return result

    # === 场景 1: 包含 <think> 标签 (通常是最终回复) ===
    if "<think>" in content:
        # 使用正则提取 <think> 内部的内容和标签后的内容
        # re.DOTALL 让 . 可以匹配换行符
        match = re.search(r"<think>(.*?)</think>\s*(.*)", content, re.DOTALL)
        if match:
            result['thought'] = match.group(1).strip()
            final_answer = match.group(2).strip()
            if final_answer:
                result['content'] = final_answer
            return {k: v for k, v in result.items() if v is not None}

    # === 场景 2: 工具选择 [Tool Selection] ===
    if "[Tool Selection]" in content or "tool_calls" in content:
        # 将整个原始文本存为 thought，以便追溯
        result['thought'] = content.strip()
        
        # 尝试提取 JSON 部分
        try:
            # 找到第一个 { 或 [ 的位置
            json_start = -1
            for i, char in enumerate(content):
                if char == '{':
                    json_start = i
                    break
            
            if json_start != -1:
                json_str = content[json_start:]
                parsed_body = safe_parse_content(json_str)

                if isinstance(parsed_body, dict):
                    raw_tool_calls = parsed_body.get('tool_calls')
                    if raw_tool_calls and isinstance(raw_tool_calls, list):
                        result['tool_calls'] = []
                        for tc in raw_tool_calls:
                            if not tc: continue
                            result['tool_calls'].append({
                                "type": "function",
                                "function": {
                                    "name": tc.get('tool_name'),
                                    "arguments": tc.get('arguments')
                                }
                            })
        except Exception:
            pass # 解析失败则只保留 thought
        
        return {k: v for k, v in result.items() if v is not None}

    # === 场景 3: 结构化思维/日志 ([Subtask_verify], [RePlan], [Task Decompose]) ===
    # 这些通常没有 tool_calls，也没有 content (对用户的回复)，纯粹是 Agent 的思考过程
    if content.strip().startswith("["):
        result['thought'] = content
        return {k: v for k, v in result.items() if v is not None}

    # === 场景 4: 普通对话 ===
    result['content'] = content
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
                        # === 使用新的处理逻辑 ===
                        processed_msg = process_assistant_message(raw_content)
                        new_entry['dialogs'].append(processed_msg)

                    elif role == 'tool':
                        tool_name = step.get('name', 'unknown_tool')
                        # Tool 的内容可能是 JSON 字符串，也可能是普通报错文本 "Failed to call..."
                        parsed_content = safe_parse_content(raw_content)

                        final_tool_content = parsed_content
                        
                        # 简单的格式化，如果是 list 则包装一下
                        if isinstance(parsed_content, list):
                            is_str_list = all(isinstance(x, str) for x in parsed_content)
                            final_tool_content = {
                                "type": "list(str)" if is_str_list else "list",
                                "content": parsed_content
                            }
                        
                        # 如果解析出来还是字符串（例如报错信息），直接放进去
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
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在写入: {output_path} ...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(final_dataset, f_out, indent=4, ensure_ascii=False)

    print("转换完成！")


if __name__ == "__main__":
    # === 配置输入输出路径 ===
    # input_file = "test.jsonl" 
    input_file = "/media/csudxy0218/ZL/AgentToolmem/Earth-agent/evaluate/Qwen3-32B/Qwen3-32B.jsonl"
    output_file = "./Qwen3-32B/Qwen3-32B.json"
    # ========================

    convert_jsonl_to_dataset(input_file, output_file)