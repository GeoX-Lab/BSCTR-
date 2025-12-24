#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import os
from typing import Dict, List

# ================= 核心路径配置 (自动获取当前脚本所在路径) =================
# 获取当前脚本文件的绝对路径目录 (即 evaluate 文件夹)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 标准答案就在当前目录下
GROUND_TRUTH_FILE = os.path.join(CURRENT_DIR, "extracted_tool_calls_GT.json")

# 搜索目录设为当前目录 (脚本会扫描同级文件夹，如 qwen3-max)
SEARCH_DIR = CURRENT_DIR


# ====================================================================

def load_json_data(file_path: str) -> List[Dict]:
    """加载 JSON 文件"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_answer_from_text(text: str) -> str:
    """提取答案逻辑"""
    if not text: return "FAIL"
    if "FAIL" in text: return "FAIL"

    match = re.search(r'<Answer>([A-F])</Answer>', text, re.IGNORECASE)
    if match: return match.group(1).upper()

    match = re.search(r'<Answer>([A-F])<Answer>', text, re.IGNORECASE)
    if match: return match.group(1).upper()

    matches = re.findall(r'\b([A-F])\b', text)
    if matches: return matches[-1].upper()

    return "UNKNOWN"


def count_tool_calls(data: Dict) -> int:
    """统计工具调用次数"""
    tool_calls = data.get("tool_calls", [])
    return len(tool_calls)


def calculate_accuracy(ground_truth_data: List[Dict], predicted_data: List[Dict]) -> Dict:
    """计算准确率"""
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    pred_dict = {}

    for item in predicted_data:
        key = str(item["question_id"])
        if key.isdigit(): key = f"question{key}"
        pred_dict[key] = item

    results = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "correct_answers": 0,
        "accuracy": 0.0,
        "detailed_results": []
    }

    for question_index, gt_item in gt_dict.items():
        # ✅ FIX: Handle the case where final_answer can be null/None
        gt_answer = (gt_item.get("final_answer") or "").strip()  # <--- 修复点 (FIXED HERE)

        if question_index not in pred_dict:
            continue

        results["evaluated_questions"] += 1
        pred_item = pred_dict[question_index]

        pred_answer_text = pred_item.get("final_answer") or pred_item.get("polished_answer", "")
        pred_answer = extract_answer_from_text(pred_answer_text)

        is_correct = (pred_answer == gt_answer)
        if is_correct:
            results["correct_answers"] += 1

        results["detailed_results"].append({
            "question_index": question_index,
            "ground_truth": gt_answer,
            "predicted": pred_answer,
            "correct": is_correct
        })

    if results["evaluated_questions"] > 0:
        results["accuracy"] = results["correct_answers"] / results["evaluated_questions"]

    return results


def load_model_tool_calls(extracted_tool_calls_path: str) -> Dict:
    """加载模型工具调用文件"""
    data = load_json_data(extracted_tool_calls_path)
    tool_calls_dict = {}
    for item in data:
        key = item["question_index"]
        if key.isdigit(): key = f"question{key}"
        tool_calls_dict[key] = item
    return tool_calls_dict


def calculate_efficiency_with_tool_calls(ground_truth_data: List[Dict], model_tool_calls_data: Dict) -> Dict:
    """计算效率"""
    gt_dict = {item["question_index"]: item for item in ground_truth_data}

    results = {
        "efficiency_scores": [],
        "average_efficiency": 0.0
    }

    for question_index, gt_item in gt_dict.items():
        if question_index not in model_tool_calls_data:
            continue

        gt_tool_count = count_tool_calls(gt_item)
        model_item = model_tool_calls_data[question_index]
        model_tool_count = count_tool_calls(model_item)

        if gt_tool_count == 0:
            efficiency = 1.0 if model_tool_count == 0 else 999.0
        else:
            efficiency = model_tool_count / gt_tool_count

        results["efficiency_scores"].append(efficiency)

    if results["efficiency_scores"]:
        results["average_efficiency"] = sum(results["efficiency_scores"]) / len(results["efficiency_scores"])

    return results


def run_evaluation(ground_truth_file: str, predicted_file: str, tool_file: str) -> Dict:
    gt_data = load_json_data(ground_truth_file)
    pred_data = load_json_data(predicted_file)

    if not gt_data:
        print("错误: GT 文件为空或读取失败")
        return {}

    acc_results = calculate_accuracy(gt_data, pred_data)

    eff_results = {}
    if tool_file and os.path.exists(tool_file):
        tool_data = load_model_tool_calls(tool_file)
        eff_results = calculate_efficiency_with_tool_calls(gt_data, tool_data)

    return {
        "accuracy": acc_results,
        "efficiency": eff_results
    }


def find_model_directories(root_dir: str) -> List[str]:
    """扫描当前目录下的所有子文件夹，寻找包含结果文件的模型目录"""
    model_dirs = []

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path) and not item.startswith(".") and item != "__pycache__":
            results_file = os.path.join(item_path, "results_summary_polished.json")
            if os.path.exists(results_file):
                model_dirs.append(item_path)

    return sorted(model_dirs)


def main():
    print(f"当前工作目录: {CURRENT_DIR}")
    print(f"标准答案文件: {GROUND_TRUTH_FILE}")

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"致命错误: 在当前目录下找不到 {os.path.basename(GROUND_TRUTH_FILE)}")
        print("请确保 JSON 文件和脚本在同一个文件夹内。")
        return

    model_dirs = find_model_directories(SEARCH_DIR)

    if not model_dirs:
        print("❌ 未找到包含 'results_summary_polished.json' 的子文件夹。")
        return

    print(f"发现 {len(model_dirs)} 个模型目录待评估...")

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print("\n" + "=" * 50)
        print(f"正在评估模型: {model_name}")
        print("=" * 50)

        pred_file = os.path.join(model_dir, "results_summary_polished.json")
        tool_file = os.path.join(model_dir, "extracted_tool_calls.json")

        results = run_evaluation(GROUND_TRUTH_FILE, pred_file, tool_file)

        if not results:
            continue

        acc = results.get("accuracy", {})
        eff = results.get("efficiency", {})

        print(f"正确数: {acc.get('correct_answers', 0)} / {acc.get('evaluated_questions', 0)}")
        print(f"准确率: {acc.get('accuracy', 0):.2%}")

        if eff:
            print(f"平均效率: {eff.get('average_efficiency', 0):.4f} (越低越好)")
        else:
            print("未找到工具调用文件，跳过效率评估")

        output_file = os.path.join(model_dir, "eval_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    main()