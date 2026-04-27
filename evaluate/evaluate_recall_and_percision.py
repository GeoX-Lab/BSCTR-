import json
import math
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

def calculate_ndcg(retrieved_list, gt_set, k):
    top_k_list = retrieved_list[:k]
    dcg = 0.0
    for i, item in enumerate(top_k_list):
        if item in gt_set:
            dcg += 1.0 / math.log2(i + 2)

    idcg = 0.0
    num_relevant_items = min(len(gt_set), k)
    for i in range(num_relevant_items):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0:
        return 0.0
    return dcg / idcg

def calculate_retrieval_metrics(gt_file_path, retrieved_file_path, top_k=5, verbose=True):
    # 1. 加载数据
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(retrieved_file_path, 'r', encoding='utf-8') as f:
        retrieved_data = json.load(f)
    
    # 2. 检查是否有 difficulty 字段
    has_difficulty = False
    for item in gt_data.values():
        if isinstance(item, dict) and "difficulty" in item:
            has_difficulty = True
            break
    
    if has_difficulty:
        # 如果有 difficulty，按难度分类统计
        categories = ["Simple", "Medium", "Complex"]
        stats = {cat: {'recall': [], 'precision': [], 'f1': [], 'ndcg': []} for cat in categories}
    else:
        # 否则统一统计
        categories = []
        stats = {"Overall": {'recall': [], 'precision': [], 'f1': [], 'ndcg': []}}
    
    if verbose:
        print(f"Calculating metrics @ Top-{top_k}")
        print(f"GT数据样本 (前3个): {list(gt_data.items())[:3]}")
        print(f"检索数据样本 (前3个): {list(retrieved_data.items())[:3]}")
        print(f"GT总数: {len(gt_data)}, 检索总数: {len(retrieved_data)}")
        if has_difficulty:
            print("检测到 difficulty 字段，按难度分类统计")
        else:
            print("未检测到 difficulty 字段，统一统计所有问题")
        print(f"{'ID':<5} | {'Diff':<8} | {'Recall':<8} | {'Prec':<8} | {'F1':<8} | {'NDCG':<8}")
        print("-" * 105)

    matched_count = 0
    sorted_items = sorted(retrieved_data.items(), key=lambda x: str(x[0]))

    for ret_key, ret_list in sorted_items:
        gt_key = ret_key
        if gt_key not in gt_data: 
            continue

        matched_count += 1
        gt_item = gt_data[gt_key]
        
        # 解析 GT 数据
        if isinstance(gt_item, dict):
            if "tools" in gt_item:
                gt_set = set(gt_item["tools"])
                difficulty = gt_item.get("difficulty", "Unknown")
            else:
                # 如果是字典但没有tools，可能整个字典就是工具列表的其他格式
                gt_set = set(gt_item.values()) if gt_item else set()
                difficulty = "Unknown"
        else:
            # 直接是列表
            gt_set = set(gt_item)
            difficulty = "Unknown"

        top_k_ret_list = ret_list[:top_k]
        top_k_ret_set = set(top_k_ret_list)
        hit_count = len(gt_set.intersection(top_k_ret_set))

        # 计算基础指标
        recall = hit_count / len(gt_set) if len(gt_set) > 0 else 0.0
        retrieved_count = len(top_k_ret_list)
        precision = hit_count / retrieved_count if retrieved_count > 0 else 0.0
        ndcg = calculate_ndcg(ret_list, gt_set, top_k)
        f1 = 2 * (recall * ndcg) / (recall + ndcg) if (recall + ndcg) > 0 else 0.0

        # 存储数据
        if has_difficulty and difficulty in ["Simple", "Medium", "Complex"]:
            stats[difficulty]['recall'].append(recall)
            stats[difficulty]['precision'].append(precision)
            stats[difficulty]['f1'].append(f1)
            stats[difficulty]['ndcg'].append(ndcg)
        elif not has_difficulty:
            # 没有difficulty时，所有数据都放入Overall
            stats["Overall"]['recall'].append(recall)
            stats["Overall"]['precision'].append(precision)
            stats["Overall"]['f1'].append(f1)
            stats["Overall"]['ndcg'].append(ndcg)

    if verbose:
        print(f"成功匹配的问题数: {matched_count}")
    
    # 3. 计算各类别平均值
    final_results = {}
    metrics_list = ['recall', 'precision', 'f1', 'ndcg']
    
    if has_difficulty:
        # 有difficulty的情况
        for cat in categories:
            count = len(stats[cat]['recall'])
            if count > 0:
                final_results[cat] = {m: sum(stats[cat][m]) / count for m in metrics_list}
                final_results[cat]['count'] = count
            else:
                final_results[cat] = {m: 0.0 for m in metrics_list}
                final_results[cat]['count'] = 0

        # 计算 Overall: 求 Simple, Medium, Complex 的加权平均
        overall = {}
        for m in metrics_list:
            weighted_sum = 0
            total_count = 0
            for cat in categories:
                if final_results[cat]['count'] > 0:
                    weighted_sum += final_results[cat][m] * final_results[cat]['count']
                    total_count += final_results[cat]['count']
            overall[m] = weighted_sum / total_count if total_count > 0 else 0.0
        overall['count'] = sum(final_results[cat]['count'] for cat in categories)
        final_results['Overall'] = overall
    else:
        # 没有difficulty的情况，直接统计Overall
        count = len(stats["Overall"]['recall'])
        final_results['Overall'] = {m: (sum(stats["Overall"][m]) / count if count > 0 else 0.0) for m in metrics_list}
        final_results['Overall']['count'] = count

    if verbose:
        print("-" * 110)
        print(f"{'Category':<10} | {'Count':<5} | {'Avg Recall':<12} | {'Avg Precision':<15} | {'Avg F1':<10} | {'Avg NDCG':<10}")
        print("-" * 110)
        
        if has_difficulty:
            for cat in ["Simple", "Medium", "Complex", "Overall"]:
                res = final_results.get(cat, {})
                print(f"{cat:<10} | {res.get('count', 0):<5} | {res.get('recall', 0.0):.4f}       | {res.get('precision', 0.0):.4f}          | {res.get('f1', 0.0):.4f}     | {res.get('ndcg', 0.0):.4f}")
        else:
            res = final_results['Overall']
            print(f"{'Overall':<10} | {res['count']:<5} | {res['recall']:.4f}       | {res['precision']:.4f}          | {res['f1']:.4f}     | {res['ndcg']:.4f}")
    
    # 为了保持兼容性，返回格式稍作调整或由外部处理
    return final_results

def evaluate_baselines_in_folder(base_folder, gt_path, json_filename="tools_pool.json", top_k=20):
    base_folder = Path(base_folder)
    if not base_folder.exists(): return
    
    baseline_folders = sorted([d for d in base_folder.iterdir() if d.is_dir()])
    all_baseline_results = {}
    
    for baseline_dir in baseline_folders:
        baseline_name = baseline_dir.name
        json_file = baseline_dir / json_filename
        if not json_file.exists(): continue
        
        print(f"\n{'='*120}\n评估基线: {baseline_name}\n{'='*120}")
        # 获取该基线在各个难度下的平均分
        final_results = calculate_retrieval_metrics(gt_path, str(json_file), top_k=top_k, verbose=True)
        all_baseline_results[baseline_name] = final_results
    
    # --- 输出汇总对比表 ---
    print("\n\n" + "="*150 + "\n所有基线对比汇总表\n" + "="*150)
    
    # 检查是否有difficulty字段
    has_difficulty = False
    for results in all_baseline_results.values():
        if "Simple" in results:
            has_difficulty = True
            break
    
    if has_difficulty:
        categories = ["Simple", "Medium", "Complex", "Overall"]
    else:
        categories = ["Overall"]
    
    metrics = ["recall", "precision", "f1", "ndcg"]
    
    for category in categories:
        print(f"\n\n{category.upper()} 难度级别:")
        print("-"*150)
        header = f"{'Baseline':<25}"
        for metric in metrics:
            header += f" | {metric.upper():<15}"
        print(header)
        print("-"*150)
        
        for baseline_name in sorted(all_baseline_results.keys()):
            res = all_baseline_results[baseline_name].get(category)
            if res is None:
                continue
            row = f"{baseline_name:<25}"
            for metric in metrics:
                val = res.get(metric, 0.0)
                row += f" | {val:.4f}        "
            print(row)
        print("-"*150)

if __name__ == "__main__":
    base_folder = "/media/csudxy0218/ZL/AgentToolmem/geoplan-bench/outputs/deepseek-v3.2"
    gt_path = "/media/csudxy0218/ZL/AgentToolmem/geoplan-bench/data/gt_with_difficulty.json"
    top_k = 15
    evaluate_baselines_in_folder(base_folder, gt_path, json_filename="tools_pool.json", top_k=top_k)