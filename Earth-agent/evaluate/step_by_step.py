#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import os
from typing import Dict, List, Any, Tuple

# ================= 配置路径 =================
ROOT_DIR = "./qwen3-max"
GROUND_TRUTH_FILE = "./extracted_tool_calls_GT.json"
# ===========================================

def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON file data"""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_answer_from_text(text: str) -> str:
    """Extract answer from text"""
    if not text: return "FAIL"
    if "FAIL" in text: return "FAIL"

    # Extract <Answer>X</Answer>
    match = re.search(r'<Answer>([A-F])</Answer>', text, re.IGNORECASE)
    if match: return match.group(1).upper()

    # Extract <Answer>X<Answer>
    match = re.search(r'<Answer>([A-F])<Answer>', text, re.IGNORECASE)
    if match: return match.group(1).upper()

    # Find single A-F
    matches = re.findall(r'\b([A-F])\b', text)
    if matches: return matches[-1].upper()

    return "UNKNOWN"


def count_tool_calls(data: Dict) -> int:
    """Count tool calls"""
    tool_calls = data.get("tool_calls", [])
    return len(tool_calls)


def calculate_accuracy(ground_truth_data: List[Dict], predicted_data: List[Dict]) -> Dict:
    """Calculate accuracy (Fixed: Removed [188:] slicing)"""
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    pred_dict = {}

    # Normalize prediction keys
    for item in predicted_data:
        key = str(item["question_id"])
        if key.isdigit(): key = f"question{key}"
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

    # Iterate through ALL ground truth items
    for question_index, gt_item in gt_dict.items():
        gt_answer = gt_item.get("final_answer", "")
        if gt_answer is None: gt_answer = ""
        gt_answer = gt_answer.strip()

        # Check if prediction exists
        if question_index not in pred_dict:
            # Uncomment below if you want to count missing files as errors
            # results["missing_predictions"].append(question_index)
            continue

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
            "predicted": pred_answer,
            "correct": is_correct,
            "status": status
        })

    if results["evaluated_questions"] > 0:
        results["accuracy"] = results["correct_answers"] / results["evaluated_questions"]

    return results


def load_model_tool_calls(extracted_tool_calls_path: str) -> Dict:
    """Load model tool calls data"""
    data = load_json_data(extracted_tool_calls_path)
    tool_calls_dict = {}
    for item in data:
        key = item["question_index"]
        if key.isdigit(): key = f"question{key}"
        tool_calls_dict[key] = item
    return tool_calls_dict


def calculate_efficiency_with_tool_calls(ground_truth_data: List[Dict],
                                         model_tool_calls_data: Dict) -> Dict:
    """Calculate efficiency (Fixed: Removed [188:] slicing)"""
    gt_dict = {item["question_index"]: item for item in ground_truth_data}

    results = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "efficiency_scores": [],
        "average_efficiency": 0.0,
        "detailed_results": []
    }

    for question_index, gt_item in gt_dict.items():
        gt_tool_count = count_tool_calls(gt_item)

        if question_index not in model_tool_calls_data:
            continue

        results["evaluated_questions"] += 1
        model_item = model_tool_calls_data[question_index]
        model_tool_count = count_tool_calls(model_item)

        if gt_tool_count == 0:
            efficiency = 1.0 if model_tool_count == 0 else 999.0  # Use 999 to indicate infinite/bad efficiency
        else:
            efficiency = model_tool_count / gt_tool_count

        results["efficiency_scores"].append(efficiency)
        results["detailed_results"].append({
            "question_index": question_index,
            "gt_tool_count": gt_tool_count,
            "model_tool_count": model_tool_count,
            "efficiency": efficiency
        })

    if results["efficiency_scores"]:
        results["average_efficiency"] = sum(results["efficiency_scores"]) / len(results["efficiency_scores"])

    return results


def run_end_to_end_evaluation(ground_truth_file: str,
                              predicted_answers_file: str,
                              model_tool_calls_file: str = None) -> Dict:
    print(f"Loading Ground Truth: {ground_truth_file}")
    ground_truth_data = load_json_data(ground_truth_file)
    print(f"Loading Predictions: {predicted_answers_file}")
    predicted_data = load_json_data(predicted_answers_file)

    if not ground_truth_data or not predicted_data:
        print("Error: Empty data files.")
        return {"summary": {"accuracy_rate": 0, "average_efficiency": 0, "total_questions": 0}}

    print("Calculating Accuracy...")
    accuracy_results = calculate_accuracy(ground_truth_data, predicted_data)

    efficiency_results = {}
    if model_tool_calls_file:
        print(f"Loading Tool Calls: {model_tool_calls_file}")
        model_tool_calls_data = load_model_tool_calls(model_tool_calls_file)
        print("Calculating Efficiency...")
        efficiency_results = calculate_efficiency_with_tool_calls(ground_truth_data, model_tool_calls_data)

    return {
        "accuracy": accuracy_results,
        "efficiency": efficiency_results,
        "summary": {
            "total_questions": accuracy_results["total_questions"],
            "accuracy_rate": accuracy_results["accuracy"],
            "average_efficiency": efficiency_results.get("average_efficiency", 0.0)
        }
    }


def print_evaluation_summary(results: Dict, model_name: str = ""):
    accuracy = results.get("accuracy", {})
    efficiency = results.get("efficiency", {})

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {model_name}")
    print("=" * 60)

    if accuracy:
        print(f"Correct: {accuracy['correct_answers']} / {accuracy['evaluated_questions']}")
        print(f"Accuracy: {accuracy['accuracy']:.2%}")

    if efficiency:
        print(f"Avg Efficiency: {efficiency.get('average_efficiency', 0):.4f}")
        print("(Ratio of Model Tool Calls / GT Tool Calls)")


def find_model_directories(root_dir: str) -> List[str]:
    model_dirs = []
    if not os.path.exists(root_dir):
        print(f"Error: Root directory {root_dir} does not exist.")
        return model_dirs

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            results_file = os.path.join(item_path, "results_summary_polished.json")
            if os.path.exists(results_file):
                model_dirs.append(item_path)
    return sorted(model_dirs)


def main():
    print(f"Starting Evaluation in: {ROOT_DIR}")

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Error: Ground Truth file not found at {GROUND_TRUTH_FILE}")
        return

    model_dirs = find_model_directories(ROOT_DIR)

    if not model_dirs:
        print(
            "No model directories found. Make sure you have the structure: ./evaluate_langchain/model_name/results_summary_polished.json")
        return

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        pred_file = os.path.join(model_dir, "results_summary_polished.json")
        tool_file = os.path.join(model_dir, "extracted_tool_calls.json")

        if not os.path.exists(tool_file):
            tool_file = None  # Efficiency check skipped if file missing

        try:
            results = run_end_to_end_evaluation(GROUND_TRUTH_FILE, pred_file, tool_file)
            print_evaluation_summary(results, model_name)

            # Save results
            out_path = os.path.join(model_dir, "eval_results.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved results to: {out_path}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()