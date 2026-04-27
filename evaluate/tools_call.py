import json
import os
from collections import defaultdict
from typing import List, Dict, Any
from pathlib import Path

# ================= 核心计算逻辑 =================

def calculate_any_order_score(gt_list: List[str], pred_list: List[str]) -> float:
    """指标 1: 任意顺序召回率 (Set Intersection)"""
    if not gt_list:
        return 1.0 if not pred_list else 0.0
    gt_set = set(gt_list)
    pred_set = set(pred_list)
    matched = gt_set.intersection(pred_set)
    return len(matched) / len(gt_set)

def calculate_in_order_score(gt_list: List[str], pred_list: List[str]) -> float:
    """指标 2: 顺序子序列匹配率 (Subsequence)"""
    if not gt_list: return 1.0
    matched_count = 0
    pred_iter = iter(pred_list)
    for gt_tool in gt_list:
        for pred_tool in pred_iter:
            if pred_tool == gt_tool:
                matched_count += 1
                break
    return matched_count / len(gt_list)

def calculate_levenshtein_similarity(gt_list: List[str], pred_list: List[str]) -> float:
    """
    指标 3: 编辑距离相似度 (Levenshtein Similarity)
    替代严格匹配。计算两个列表的相似程度，容忍插入、删除和替换。
    Score = 1 - (Distance / MaxLen)
    """
    len_gt = len(gt_list)
    len_pred = len(pred_list)

    # 边界情况处理
    if len_gt == 0:
        return 1.0 if len_pred == 0 else 0.0
    if len_pred == 0:
        return 0.0

    # 初始化矩阵
    # matrix[i][j] 表示 gt[:i] 和 pred[:j] 的距离
    matrix = [[0] * (len_pred + 1) for _ in range(len_gt + 1)]

    for i in range(len_gt + 1):
        matrix[i][0] = i
    for j in range(len_pred + 1):
        matrix[0][j] = j

    # 动态规划计算距离
    for i in range(1, len_gt + 1):
        for j in range(1, len_pred + 1):
            cost = 0 if gt_list[i - 1] == pred_list[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1, 
                matrix[i - 1][j - 1] + cost 
            )

    distance = matrix[len_gt][len_pred]
    max_len = max(len_gt, len_pred)
    
    return 1.0 - (distance / max_len)

# ================= 主评估流程 =================

def evaluate_tools_by_difficulty(gt_file: str, pred_file: str, verbose=True) -> Dict[str, Dict[str, float]]:
    """
    评估工具调用性能，按难度级别分类
    
    Returns:
        dict: 按难度分类的结果 {difficulty: {metric: avg_score}}
    """
    
    if verbose:
        print(f"Loading GT: {gt_file}")
        print(f"Loading Pred: {pred_file}")

    if not os.path.exists(gt_file) or not os.path.exists(pred_file):
        print("Error: File not found.")
        return {}

    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    # 统计容器
    categories = ["Simple", "Medium", "Complex", "Overall"]
    # 将 Strict 替换为 Levenshtein
    metrics = ["Any Order", "In Order", "Levenshtein"] 
    stats = {cat: {m: [] for m in metrics} for cat in categories}

    if verbose:
        print("-" * 115)
        print(f"{'ID':<6} | {'Diff':<8} | {'Any Order':<10} | {'In Order':<10} | {'Levenshtein':<12} | {'GT vs Pred Len'}")
        print("-" * 115)

    sorted_keys = sorted(gt_data.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for key in sorted_keys:
        gt_item = gt_data[key]
        
        # 1. 解析 GT
        if isinstance(gt_item, dict):
            difficulty = gt_item.get("difficulty", "Unknown")
            gt_tools = gt_item.get("tools", [])
        else:
            difficulty = "Unknown"
            gt_tools = gt_item
        
        # 2. 解析 Pred
        pred_tools = pred_data.get(key, [])
        if pred_tools is None: pred_tools = []

        # 3. 计算指标
        s_any = calculate_any_order_score(gt_tools, pred_tools)
        s_in = calculate_in_order_score(gt_tools, pred_tools)
        
        # 使用新的 Levenshtein 算法
        s_lev = calculate_levenshtein_similarity(gt_tools, pred_tools)

        # 4. 存储
        if difficulty in ["Simple", "Medium", "Complex"]:
            stats[difficulty]["Any Order"].append(s_any)
            stats[difficulty]["In Order"].append(s_in)
            stats[difficulty]["Levenshtein"].append(s_lev)
        
        # Overall 包含所有问题
        stats["Overall"]["Any Order"].append(s_any)
        stats["Overall"]["In Order"].append(s_in)
        stats["Overall"]["Levenshtein"].append(s_lev)

        # 打印详情
        if verbose:
            print(f"{key:<6} | {difficulty:<8} | {s_any:.4f}     | {s_in:.4f}     | {s_lev:.4f}       | {len(gt_tools)} vs {len(pred_tools)}")

    # ================= 打印汇总 =================
    if verbose:
        print("-" * 115)
        print("Evaluation Summary by Difficulty (Avg Scores):")
        print("-" * 115)
        print(f"{'Category':<10} | {'Count':<5} | {'Any Order':<12} | {'In Order':<12} | {'Levenshtein':<12}")
        print("-" * 115)

    # 计算平均值并保存结果
    result = {}
    for cat in categories:
        data = stats.get(cat)
        count = len(data["Any Order"])
        
        if count == 0:
            if verbose:
                print(f"{cat:<10} | {0:<5} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12}")
            result[cat] = None
            continue
        
        avg_any = sum(data["Any Order"]) / count
        avg_in = sum(data["In Order"]) / count
        avg_lev = sum(data["Levenshtein"]) / count
        
        if verbose:
            print(f"{cat:<10} | {count:<5} | {avg_any:.4f}       | {avg_in:.4f}       | {avg_lev:.4f}")
        
        result[cat] = {
            "count": count,
            "any_order": avg_any,
            "in_order": avg_in,
            "levenshtein": avg_lev
        }
    
    if verbose:
        print("-" * 115)

        # 保存结果
        output_path = pred_file.replace(".json", "_metrics.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {output_path}\n")
    
    return result


def evaluate_baselines_by_folder(base_folder: str, gt_file: str):
    """
    评估一个文件夹中的所有基线，并输出按难度级别分类的对比表
    
    Args:
        base_folder: 包含所有基线子文件夹的路径
        gt_file: ground truth 文件路径
    """
    base_folder = Path(base_folder)
    
    if not base_folder.exists():
        print(f"错误: 文件夹 {base_folder} 不存在")
        return
    
    # 获取所有子文件夹（基线）
    baseline_folders = sorted([d for d in base_folder.iterdir() if d.is_dir()])
    
    if not baseline_folders:
        print(f"错误: 在 {base_folder} 中找不到任何子文件夹")
        return
    
    print(f"发现 {len(baseline_folders)} 个基线:")
    for baseline_dir in baseline_folders:
        print(f"  - {baseline_dir.name}")
    print("\n" + "="*120 + "\n")
    
    # 收集所有基线结果
    all_baseline_results = {}
    
    # 评估每个基线
    for baseline_dir in baseline_folders:
        baseline_name = baseline_dir.name
        pred_file = baseline_dir / "tool_call.json"
        
        if not pred_file.exists():
            print(f"警告: {baseline_name} 中找不到 tool_call.json，跳过")
            continue
        
        print(f"{'='*115}")
        print(f"评估基线: {baseline_name}")
        print(f"{'='*115}")
        
        # 评估此基线
        result = evaluate_tools_by_difficulty(gt_file, str(pred_file), verbose=True)
        all_baseline_results[baseline_name] = result
    
    # ================= 输出最终的对比表（按难度级别分类）=================
    print("\n\n")
    print("="*150)
    print("所有基线对比汇总表 - 按难度级别分类")
    print("="*150)
    
    categories = ["Simple", "Medium", "Complex", "Overall"]
    metrics = ["any_order", "in_order", "levenshtein"]
    metric_names = ["Any Order", "In Order", "Levenshtein"]
    
    # 为每个难度类别输出一个表
    for category in categories:
        print(f"\n\n{category.upper()} 难度级别:")
        print("-"*180)
        
        # 构建表头
        header = f"{'Baseline':<20}"
        for metric_name in metric_names:
            header += f" | {metric_name:<15}"
        header += f" | {'Overall':<15}"
        print(header)
        print("-"*180)
        
        # 为每个基线输出一行
        for baseline_name in sorted(all_baseline_results.keys()):
            result = all_baseline_results[baseline_name]
            cat_result = result.get(category)
            
            if not cat_result:
                print(f"{baseline_name:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
                continue
            
            # 输出一行
            row = f"{baseline_name:<20}"
            metric_values = []
            for metric in metrics:
                value = cat_result.get(metric, 0.0)
                metric_values.append(value)
                row += f" | {value:.4f}        "
            
            # 计算Overall：三个指标的平均值
            overall = sum(metric_values) / len(metric_values)
            row += f" | {overall:.4f}        "
            print(row)
        
        print("-"*180)
        
        # 输出该难度级别下所有基线的加权平均值（按问题数量加权）
        row = f"{'AVERAGE':<20}"
        for metric in metrics:
            weighted_sum = 0.0
            total_count = 0
            for baseline_name in all_baseline_results.keys():
                result = all_baseline_results[baseline_name]
                cat_result = result.get(category)
                if cat_result:
                    value = cat_result.get(metric, 0.0)
                    count = cat_result.get("count", 0)
                    # 按问题数量加权
                    weighted_sum += value * count
                    total_count += count
            
            if total_count > 0:
                avg = weighted_sum / total_count
                row += f" | {avg:.4f}        "
            else:
                row += f" | {'N/A':<15}"
        
        # 计算Overall列的平均值（三个指标平均值的加权平均）
        overall_weighted_sum = 0.0
        total_count = 0
        for baseline_name in all_baseline_results.keys():
            result = all_baseline_results[baseline_name]
            cat_result = result.get(category)
            if cat_result:
                count = cat_result.get("count", 0)
                metric_values = []
                for metric in metrics:
                    value = cat_result.get(metric, 0.0)
                    metric_values.append(value)
                overall = sum(metric_values) / len(metric_values)
                overall_weighted_sum += overall * count
                total_count += count
        
        if total_count > 0:
            overall_avg = overall_weighted_sum / total_count
            row += f" | {overall_avg:.4f}        "
        else:
            row += f" | {'N/A':<15}"
        
        print(row)
        print("-"*180)

if __name__ == "__main__":
    # 请修改为你的实际路径
    # 注意：GT 文件必须是包含 {"difficulty": ...} 结构的新文件
    GROUND_TRUTH_FILE = "/media/csudxy0218/ZL/AgentToolmem/API-Bank/test-data/level-gt.json" 
    ""
    
    # # 方式 1: 使用文件夹评估所有基线
    BASE_FOLDER = "/media/csudxy0218/ZL/AgentToolmem/API-Bank/outputs/deepseek-v3.2/level-3"
    evaluate_baselines_by_folder(BASE_FOLDER, GROUND_TRUTH_FILE)
    
    # 方式 2: 如果要单独评估特定文件，可以使用
    # plan_sgc = "/media/csudxy0218/ZL/AgentToolmem/geoplan-bench/outputs/deepseek-v3.2/graph/tool_call.json"
    # evaluate_tools_by_difficulty(GROUND_TRUTH_FILE, plan_sgc)