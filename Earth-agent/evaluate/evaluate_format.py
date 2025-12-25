import json
import re
import ast
import os


def extract_question_id(text):
    """从文本中提取 question ID，例如从 'benchmark/data/question124' 提取 '124'"""
    match = re.search(r'question(\d+)', text)
    if match:
        return match.group(1)
    return None


def parse_tool_calls_from_history(history):
    """从对话历史中提取所有的工具调用"""
    all_tool_calls = []

    for turn in history:
        if turn.get('role') == 'assistant':
            content = turn.get('content', '')

            # 检查是否包含 Tool Selection 块
            if '[Tool Selection]' in content:
                try:
                    # 提取字典部分的字符串
                    # 假设格式是 "[Tool Selection]\n{'tasks': ...}"
                    dict_str = content.split('[Tool Selection]', 1)[1].strip()

                    # 使用 ast.literal_eval 安全地解析 Python 风格的字典字符串
                    # 注意：你的数据中使用的是单引号和 None，这是 Python 语法而不是标准 JSON
                    tool_data = ast.literal_eval(dict_str)

                    if 'tool_calls' in tool_data and isinstance(tool_data['tool_calls'], list):
                        all_tool_calls.extend(tool_data['tool_calls'])

                except Exception as e:
                    print(f"解析工具调用失败: {e}")
                    continue

    return all_tool_calls


def convert_jsonl_to_eval_format(input_file, output_dir):
    """主转换函数"""

    # 准备两个列表存放转换后的数据
    results_summary = []  # 对应 results_summary_polished.json
    extracted_tools = []  # 对应 extracted_tool_calls.json

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在处理文件: {input_file} ...")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                entry = json.loads(line)

                # 1. 获取 Question ID
                query = entry.get('query', '')
                q_id = extract_question_id(query)

                if not q_id:
                    print(f"警告: 第 {line_num} 行无法提取 Question ID，跳过。")
                    continue

                # 构造 question_index (例如 question124)
                q_index = f"question{q_id}"

                # 2. 处理预测结果 (Results Summary)
                final_res = entry.get('final_result', '')
                # 确保答案格式化，虽然评估脚本支持裸字符，但加上标签更稳健
                # 如果 final_result 已经是 "C"，可以包装成 "<Answer>C</Answer>"
                formatted_answer = f"<Answer>{final_res}</Answer>" if final_res else "FAIL"

                results_summary.append({
                    "question_id": q_id,  # 评估脚本会将数字转为 questionX
                    "final_answer": formatted_answer,
                    "polished_answer": formatted_answer,  # 备用字段
                    "status": entry.get('status', 'unknown')
                })

                # 3. 处理工具调用 (Tool Calls)
                history = entry.get('history', [])
                tool_calls = parse_tool_calls_from_history(history)

                extracted_tools.append({
                    "question_index": q_index,  # 必须匹配 GT 文件的格式
                    "tool_calls": tool_calls
                })

            except json.JSONDecodeError:
                print(f"错误: 第 {line_num} 行 JSON 格式不正确。")
                continue

    # 保存结果文件
    summary_path = os.path.join(output_dir, "results_summary_polished.json")
    tools_path = os.path.join(output_dir, "extracted_tool_calls.json")

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    with open(tools_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_tools, f, ensure_ascii=False, indent=2)

    print("-" * 40)
    print("转换完成！")
    print(f"生成文件 1: {summary_path} (包含 {len(results_summary)} 条数据)")
    print(f"生成文件 2: {tools_path} (包含 {len(extracted_tools)} 条数据)")


# ================= 使用示例 =================

# 假设你的原始文件名为 model_output.jsonl
# 这里的 input_content 只是模拟你的输入文件，实际使用时请替换为真实文件路径
if __name__ == "__main__":
    # 配置你的文件路径
    INPUT_FILE = "deepseek/deepseek_outputs.jsonl"  # 你的原始 jsonl 文件路径
    OUTPUT_DIR = "deepseek"  # 你希望输出的目录
    # 运行转换
    convert_jsonl_to_eval_format(INPUT_FILE, OUTPUT_DIR)