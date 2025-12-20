import argparse
import json


def process_benchmark_data(input_file, eval_output_file, tool_output_file):
    # 读取原始 JSON 文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 {input_file} 不是有效的 JSON 格式")
        return

    eval_data = {}
    tool_chain_data = {}

    # 遍历每一个数据条目 (例如 "1", "2"...)
    for key, item in data.items():
        # --- 任务 1: 提取 Evaluation 和 Choices ---
        eval_data[key] = {
            "choices": item.get("choices"),
            "evaluation": item.get("evaluation")
        }

        # --- 任务 2: 提取工具链 ---
        tool_names = []
        dialogs = item.get("dialogs", [])

        if dialogs:
            for message in dialogs:
                # 检查该条消息是否有 tool_calls
                if "tool_calls" in message and message["tool_calls"]:
                    for tool in message["tool_calls"]:
                        # 提取函数名称
                        func_name = tool["function"]["name"]
                        tool_names.append(func_name)

        tool_chain_data[key] = tool_names

    # 保存 Evaluation 和 Choices 数据
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False)
    print(f"成功保存 Evaluation 数据到: {eval_output_file}")

    # 保存工具链数据
    with open(tool_output_file, 'w', encoding='utf-8') as f:
        json.dump(tool_chain_data, f, indent=4, ensure_ascii=False)
    print(f"成功保存工具链数据到: {tool_output_file}")
if __name__ == '__main__':
    process_benchmark_data("./question.json", "./evaluation.json", "./tool_chain.json")