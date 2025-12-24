import json
import os
from typing import Dict, List, Any, Tuple

# ================= 路径配置 (自动获取) =================
# 获取当前脚本所在目录 (即 evaluate 文件夹)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 标准答案文件路径
GROUND_TRUTH_FILE = os.path.join(CURRENT_DIR, "extracted_tool_calls_GT.json")
# 搜索模型的根目录 (设为当前目录，以便扫描 qwen3-max)
SEARCH_ROOT_DIR = CURRENT_DIR


# =====================================================

def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON file data"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_tool_name(tool_call: Dict) -> str:
    """Safely extract tool name handling different keys"""
    return tool_call.get("tool_name") or tool_call.get("name") or ""


def get_tool_args(tool_call: Dict) -> Any:
    """Safely extract arguments handling different keys"""
    return tool_call.get("arguments") or tool_call.get("args") or tool_call.get("input") or {}


def extract_tool_names_from_calls(tool_calls: List[Dict]) -> List[str]:
    """Extract tool names from tool calls list"""
    return [get_tool_name(call) for call in tool_calls]


def find_tool_calls_from_data(data: Dict) -> List[str]:
    """Extract tool call name list from data"""
    tool_calls = data.get("tool_calls", [])
    return extract_tool_names_from_calls(tool_calls)


def contains_all_tool_calls_any_order(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """Check if all expected tool calls are contained (order not considered)"""
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)

    if not expected_tools:
        return {
            "score": 1.0,
            "key": "contains_all_tool_calls_any_order",
            "expected": expected_tools,
            "actual": actual_tools,
            "details": {"matched_tools": 0, "total_expected": 0}
        }

    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    matched_tools = expected_set.intersection(actual_set)

    # Avoid division by zero
    score = len(matched_tools) / len(expected_set) if expected_set else 0

    return {
        "score": score,
        "key": "contains_all_tool_calls_any_order",
        "expected": expected_tools,
        "actual": actual_tools,
        "details": {
            "matched_tools": len(matched_tools),
            "total_expected": len(expected_set),
            "matched_tool_names": list(matched_tools)
        }
    }


def contains_all_tool_calls_in_order(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """Check if all expected tool calls are contained (considering order)"""
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)

    if not expected_tools:
        return {
            "score": 1.0,
            "key": "contains_all_tool_calls_in_order",
            "expected": expected_tools,
            "actual": actual_tools,
            "details": {"matched_in_order": 0, "total_expected": 0}
        }

    matched_count = 0
    actual_iter = iter(actual_tools)

    for expected_tool in expected_tools:
        found = False
        for actual_tool in actual_iter:
            if actual_tool == expected_tool:
                matched_count += 1
                found = True
                break
        if not found:
            break

    score = matched_count / len(expected_tools) if expected_tools else 0

    return {
        "score": score,
        "key": "contains_all_tool_calls_in_order",
        "expected": expected_tools,
        "actual": actual_tools,
        "details": {
            "matched_in_order": matched_count,
            "total_expected": len(expected_tools)
        }
    }


def trajectory_step_wise_score(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """Calculate trajectory step-wise score (strict order matching)"""
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)

    if not expected_tools:
        return {"score": 0, "key": "trajectory_step_wise"}

    correct_steps = 0
    min_length = min(len(expected_tools), len(actual_tools))

    for i in range(min_length):
        if expected_tools[i] == actual_tools[i]:
            correct_steps += 1
        else:
            break

    score = correct_steps / len(expected_tools) if len(expected_tools) > 0 else 0

    return {
        "score": score,
        "key": "trajectory_step_wise",
        "details": {
            "correct_steps": correct_steps,
            "total_expected": len(expected_tools),
            "expected_sequence": expected_tools,
            "actual_sequence": actual_tools[:len(expected_tools)]
        }
    }


def check_parameter_accuracy(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """Check the accuracy of tool call parameters (Updated for 'tool_name' and 'arguments')"""
    predicted_calls = predicted_data.get("tool_calls", [])
    expected_calls = ground_truth_data.get("tool_calls", [])

    if not expected_calls:
        return {
            "score": 1.0,
            "key": "parameter_accuracy",
            "details": {"expected_count": 0, "actual_count": len(predicted_calls)}
        }

    total_expected_steps = len(expected_calls)
    matched_steps = 0
    parameter_details = []

    min_length = min(len(predicted_calls), len(expected_calls))

    for i in range(min_length):
        pred_call = predicted_calls[i]
        exp_call = expected_calls[i]

        # Extract name and args using helper to handle variable key names
        p_name = get_tool_name(pred_call)
        p_args = get_tool_args(pred_call)

        e_name = get_tool_name(exp_call)
        e_args = get_tool_args(exp_call)

        call_detail = {
            "step": i + 1,
            "expected_tool_name": e_name,
            "actual_tool_name": p_name,
            "is_correct": False
        }

        # Name and Args must both match
        # Note: dict comparison (p_args == e_args) handles key order automatically
        if p_name == e_name and p_args == e_args:
            matched_steps += 1
            call_detail["is_correct"] = True
            parameter_details.append(call_detail)
        else:
            call_detail["is_correct"] = False
            parameter_details.append(call_detail)
            # Strict step-by-step: stop at first mismatch?
            # The previous logic had a 'break' here. Keeping strict logic.
            break

    score = matched_steps / total_expected_steps if total_expected_steps > 0 else 0

    return {
        "score": score,
        "key": "parameter_accuracy",
        "details": {
            "matched_steps": matched_steps,
            "total_expected_steps": total_expected_steps,
            "call_details": parameter_details
        }
    }


def evaluate_single_question(predicted_data: Dict, ground_truth_data: Dict) -> Dict:
    results = {}
    results["contains_all_tool_calls_any_order"] = contains_all_tool_calls_any_order(predicted_data, ground_truth_data)
    results["contains_all_tool_calls_in_order"] = contains_all_tool_calls_in_order(predicted_data, ground_truth_data)
    results["trajectory_step_wise_score"] = trajectory_step_wise_score(predicted_data, ground_truth_data)
    results["parameter_accuracy"] = check_parameter_accuracy(predicted_data, ground_truth_data)
    return results


def run_step_by_step_evaluation(predicted_file: str, ground_truth_file: str) -> Dict:
    """Run complete step-by-step evaluation"""
    predicted_data = load_json_data(predicted_file)
    ground_truth_data = load_json_data(ground_truth_file)

    if not ground_truth_data:
        print(" Error: Ground Truth data is empty or not found.")
        return {}
    if not predicted_data:
        print(" Error: Prediction data is empty or not found.")
        return {}

    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    pred_dict = {}
    for item in predicted_data:
        key = str(item["question_index"])
        if key.isdigit(): key = f"question{key}"
        pred_dict[key] = item

    all_results = {}
    summary_stats = {
        "total_questions": 0,
        "evaluated_questions": 0,
        "missing_predictions": [],
        "metrics_summary": {
            "contains_all_tool_calls_any_order": {"total_score": 0, "count": 0},
            "contains_all_tool_calls_in_order": {"total_score": 0, "count": 0},
            "trajectory_step_wise_score": {"total_score": 0, "count": 0},
            "parameter_accuracy": {"total_score": 0, "count": 0}
        }
    }

    # Iterate through ALL ground truth items
    for question_index, gt_item in gt_dict.items():
        summary_stats["total_questions"] += 1

        if question_index not in pred_dict:
            # summary_stats["missing_predictions"].append(question_index)
            continue

        pred_item = pred_dict[question_index]
        summary_stats["evaluated_questions"] += 1

        question_results = evaluate_single_question(pred_item, gt_item)
        all_results[question_index] = question_results

        for metric_name, metric_result in question_results.items():
            if metric_name in summary_stats["metrics_summary"]:
                summary_stats["metrics_summary"][metric_name]["total_score"] += metric_result["score"]
                summary_stats["metrics_summary"][metric_name]["count"] += 1

    for metric_name, metric_stats in summary_stats["metrics_summary"].items():
        if metric_stats["count"] > 0:
            metric_stats["average_score"] = metric_stats["total_score"] / metric_stats["count"]
        else:
            metric_stats["average_score"] = 0

    return {
        "individual_results": all_results,
        "summary": summary_stats
    }


def print_evaluation_summary(results: Dict, model_name: str = ""):
    if not results: return
    summary = results["summary"]
    print("=" * 60)
    print(f"Step-by-Step Evaluation: {model_name}")
    print("=" * 60)
    print(f"Evaluated questions: {summary['evaluated_questions']} / {summary['total_questions']}")

    print("\nAverage scores (0.00 - 1.00):")
    print("-" * 60)
    for metric_name, metric_stats in summary["metrics_summary"].items():
        avg_score = metric_stats.get("average_score", 0)
        print(f"{metric_name:<35}: {avg_score:.4f}")
    print("=" * 60)


def find_model_directories(root_dir: str) -> List[str]:
    """Find all model directories next to this script"""
    model_dirs = []
    if not os.path.exists(root_dir):
        return model_dirs

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        # Scan for directories that are NOT hidden and NOT __pycache__
        if os.path.isdir(item_path) and not item.startswith(".") and item != "__pycache__":
            tool_calls_file = os.path.join(item_path, "extracted_tool_calls.json")
            if os.path.exists(tool_calls_file):
                model_dirs.append(item_path)

    return sorted(model_dirs)


def main():
    print(f"Working Directory: {CURRENT_DIR}")
    print(f"Ground Truth File: {GROUND_TRUTH_FILE}")

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f" Error: Ground Truth file not found at {GROUND_TRUTH_FILE}")
        return

    # Find model directories in the CURRENT_DIR (e.g., qwen3-max)
    model_dirs = find_model_directories(SEARCH_ROOT_DIR)

    if not model_dirs:
        print(" No model directories found containing 'extracted_tool_calls.json'.")
        return

    print(f"\nFound {len(model_dirs)} model directories.")

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"\nProcessing: {model_name}...")

        predicted_file = os.path.join(model_dir, "extracted_tool_calls.json")

        try:
            results = run_step_by_step_evaluation(predicted_file, GROUND_TRUTH_FILE)

            if results:
                print_evaluation_summary(results, model_name)

                output_file = os.path.join(model_dir, "step_by_step_evaluation_results.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Saved results to: {output_file}")
            else:
                print("Skipping (No results generated).")

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()