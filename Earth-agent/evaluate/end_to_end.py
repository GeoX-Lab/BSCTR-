#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import os
from typing import Dict, List

# ================= 配置区域 =================
# 1. 设置标准答案路径
GROUND_TRUTH_FILE = "./extracted_tool_calls_GT.json"
# 2. 设置你刚刚转换生成的模型输出目录
MODEL_OUTPUT_DIR = "deepseek"


# ===========================================

def load_json_data(file_path: str) -> List[Dict]:
    """加载 JSON 文件"""
    if not os.path.exists(file_path):
        print(f"找不到文件 {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_answer_from_text(text: str) -> str:
    """提取答案逻辑"""
    if not text: return "FAIL"
    if "FAIL" in text: return "FAIL"

    # 尝试提取 <Answer>X</Answer>
    match = re.search(r'<Answer>([A-F])</Answer>', text, re.IGNORECASE)
    if match: return match.group(1).upper()

    match = re.search(r'<Answer>([A-F])<Answer>', text, re.IGNORECASE)
    if match: return match.group(1).upper()

    # 兜底：找最后一个单独的字母
    matches = re.findall(r'\b([A-F])\b', text)
    if matches: return matches[-1].upper()

    return "UNKNOWN"


def count_tool_calls(data: Dict) -> int:
    return len(data.get("tool_calls", []))


def calculate_accuracy(ground_truth_data: List[Dict], predicted_data: List[Dict]) -> Dict:
    """计算准确率 (已移除 [188:] 切片限制，以便测试任意数据)"""
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    pred_dict = {}

    for item in predicted_data:
        key = str(item["question_id"])
        if key.isdigit():
            key = f"question{key}"
        pred_dict[key] = item

    results = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "correct_answers": 0,
        "fail_answers": 0,
        "unknown_answers": 0,
        "missing_predictions": [],
        "accuracy": 0.0,
        "detailed_results": []
    }

    # 注意：这里去掉了原代码中的 [188:] 切片，改为遍历所有数据
    for question_index, gt_item in gt_dict.items():
        # 如果预测数据里没有这个题，跳过（或者记录缺失）
        if question_index not in pred_dict:
            # 只有当你想严格评估所有GT时才取消注释下面两行，
            # 为了测试方便，我们只评估预测文件中存在的题目
            # results["missing_predictions"].append(question_index)
            continue

        gt_answer = gt_item.get("final_answer", "").strip()

        results["evaluated_questions"] += 1
        pred_item = pred_dict[question_index]
        pred_answer_text = pred_item.get("final_answer") or pred_item.get("polished_answer", "")
        pred_answer = extract_answer_from_text(pred_answer_text)

        is_correct = False
        status = "incorrect"

        if pred_answer == "FAIL":
            results["fail_answers"] += 1
            status = "fail"
        elif pred_answer == "UNKNOWN":
            results["unknown_answers"] += 1
            status = "unknown"
        elif pred_answer == gt_answer:
            results["correct_answers"] += 1
            is_correct = True
            status = "correct"

        results["detailed_results"].append({
            "question_index": question_index,
            "ground_truth": gt_answer,
            "predicted_raw": pred_answer_text,
            "predicted_extracted": pred_answer,
            "correct": is_correct,
            "status": status
        })

    if results["evaluated_questions"] > 0:
        results["accuracy"] = results["correct_answers"] / results["evaluated_questions"]

    return results


def calculate_efficiency(ground_truth_data: List[Dict], model_tool_calls_data: List[Dict]) -> Dict:
    """计算效率 (已移除 [188:] 切片限制)"""
    gt_dict = {item["question_index"]: item for item in ground_truth_data}

    # 转换模型工具调用数据为字典
    model_tool_dict = {}
    for item in model_tool_calls_data:
        key = item["question_index"]
        if key.isdigit(): key = f"question{key}"
        model_tool_dict[key] = item

    results = {
        "evaluated_questions": 0,
        "efficiency_scores": [],
        "average_efficiency": 0.0,
        "detailed_results": []
    }

    for question_index, gt_item in gt_dict.items():
        if question_index not in model_tool_dict:
            continue

        gt_tool_count = count_tool_calls(gt_item)
        model_item = model_tool_dict[question_index]
        model_tool_count = count_tool_calls(model_item)

        results["evaluated_questions"] += 1

        if gt_tool_count == 0:
            efficiency = 1.0 if model_tool_count == 0 else 999.0  # 避免除以零
        else:
            efficiency = model_tool_count / gt_tool_count

        results["efficiency_scores"].append(efficiency)
        results["detailed_results"].append({
            "question_index": question_index,
            "gt_count": gt_tool_count,
            "model_count": model_tool_count,
            "efficiency": efficiency
        })

    if results["efficiency_scores"]:
        results["average_efficiency"] = sum(results["efficiency_scores"]) / len(results["efficiency_scores"])

    return results


def main():
    print("=" * 50)
    print("开始测试评估逻辑")
    print("=" * 50)

    # 1. 构造文件路径
    pred_file = os.path.join(MODEL_OUTPUT_DIR, "results_summary_polished.json")
    tool_file = os.path.join(MODEL_OUTPUT_DIR, "extracted_tool_calls.json")

    # 2. 检查文件是否存在
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Ground Truth 文件不存在: {GROUND_TRUTH_FILE}")
        return
    if not os.path.exists(pred_file):
        print(f"预测文件不存在: {pred_file}")
        return
    if not os.path.exists(tool_file):
        print(f"工具调用文件不存在: {tool_file}")
        return

    # 3. 加载数据
    print(f"正在加载数据...")
    gt_data = load_json_data(GROUND_TRUTH_FILE)
    pred_data = load_json_data(pred_file)
    tool_data = load_json_data(tool_file)

    print(f"Ground Truth 数据量: {len(gt_data)}")
    print(f"模型预测 数据量: {len(pred_data)}")

    # 4. 运行评估
    print("\n---------- 正在评估准确率 ----------")
    acc_results = calculate_accuracy(gt_data, pred_data)

    print(f"匹配到的问题数: {acc_results['evaluated_questions']}")
    print(f"正确数量: {acc_results['correct_answers']}")
    print(f"准确率 (Accuracy): {acc_results['accuracy']:.2%}")

    if acc_results['evaluated_questions'] > 0:
        print("\n详细结果示例 (前3条):")
        for res in acc_results['detailed_results'][:3]:
            print(
                f"  {res['question_index']}: GT={res['ground_truth']}, Pred={res['predicted_extracted']} -> {res['status']}")

    print("\n---------- 正在评估效率 ----------")
    eff_results = calculate_efficiency(gt_data, tool_data)

    print(f"匹配到的问题数: {eff_results['evaluated_questions']}")
    print(f"平均效率 (Efficiency): {eff_results['average_efficiency']:.4f}")
    print("(注: 效率值 < 1 表示模型用的工具更少，> 1 表示用的更多)")

    # 5. 保存结果
    output_result_file = os.path.join(MODEL_OUTPUT_DIR, "final_test_result.json")
    final_output = {
        "accuracy_metrics": acc_results,
        "efficiency_metrics": eff_results
    }
    with open(output_result_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n测试完成！详细结果已保存至: {output_result_file}")


if __name__ == "__main__":
    main()