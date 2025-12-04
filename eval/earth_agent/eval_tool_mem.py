from ToolMem import ToolMem
import json
from typing import List, Optional
import yaml
import requests
import numpy as np
import matplotlib.pyplot as plt
def get_embedding(text: str) -> np.ndarray:
    with open("/media/csudxy0218/ZL/AgentToolmem/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        url = cfg["ollama"]["embedding_url"]
        model_name = cfg["ollama"]["model_name"]

        data = {"model": model_name, "prompt": text}
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        embedding = response.json()["embedding"]
        return np.array(embedding, dtype=float)

    except Exception as e:
        dim = cfg["ollama"].get("embedding_dim", 768)
        print(f"Error getting embedding: {e}")
        return np.zeros(dim, dtype=float)


def cos_similarity(vec1: Optional[np.ndarray], vec2: Optional[np.ndarray]) -> float:
    if vec1 is None or vec2 is None:
        return 0.0

    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.dot(v1, v2) / denom)

def run_tool_rag_experiment(
    groundtruth_path: str,
    tools_path: str,
    topk_list: List[int] = [4, 5, 6],
):
    tool_mem = ToolMem()

    tools_list = tool_mem.get_node_from_doc()
    print(f"[INFO] Loaded {len(tools_list)} tools from {tools_path}")

    # 检查 vector 是否存在
    missing_vec = [name for name, t in tools_list.items() if t.get("vector") is None]
    print(f"[DEBUG] Tools missing vectors: {len(missing_vec)}")

    with open(groundtruth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # 保存结果
    recall_results = {str(k): [] for k in topk_list}
    precision_results = {str(k): [] for k in topk_list}

    # 遍历每个问题
    for qid, item in gt_data.items():
        query = item["question"]
        gt_tools = set(item["tools_used"])

        sims = tool_mem.get_similar_tools(
            query,
            top_k=max(topk_list),
            use_graph=True,
            seed_k=3,
            depth=2,
            graph_lambda=0.3,
        )
        predicted_order = [name for name, _, _ in sims]

        print("\n==============================")
        print(f"[Q{qid}] Query: {query}")
        print(f"Ground Truth Tools: {gt_tools}")

        print("Retrieved (ranked):")
        for rank, (name, score, _) in enumerate(sims[:10], start=1):
            hit_mark = "✅" if name in gt_tools else "❌"
            print(f"  {rank}. {name:<40} score={score:.4f} {hit_mark}")

        for k in topk_list:
            predicted = set(predicted_order[:k])

            hits = len(predicted & gt_tools)

            recall_k = hits / len(gt_tools)
            precision_k = hits / k

            recall_results[str(k)].append(recall_k)
            precision_results[str(k)].append(precision_k)

            hits = predicted & gt_tools
            missed = gt_tools - predicted
            false_positive = predicted - gt_tools

            print("\nGround Truth vs Retrieval:")
            print(f"  Hits (正确检索到): {hits if hits else 'None'}")
            print(f"  Missed (未检索到): {missed if missed else 'None'}")
            print(f"  False Positives:  {false_positive if false_positive else 'None'}")

            print(f"  Recall = {len(hits)}/{len(gt_tools)} = {len(hits) / len(gt_tools):.3f}")
            print(f"  Precision = {len(hits)}/{len(predicted)} = {len(hits) / len(predicted):.3f}")

    return (
        recall_results,
        precision_results
    )

def plot_recall_precision(recall_results, precision_results, topk_list, out_dir="./"):
    avg_recall = [np.mean(recall_results[str(k)]) for k in topk_list]
    avg_precision = [np.mean(precision_results[str(k)]) for k in topk_list]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(topk_list, avg_recall, marker='o')
    plt.title("Recall@K Curve")
    plt.xlabel("K")
    plt.ylabel("Recall")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(topk_list, avg_precision, marker='o', color="green")
    plt.title("Precision@K Curve")
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.grid(True)

    plt.tight_layout()
    save_path = out_dir.rstrip("/") + "/recall_precision_curve.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")

    plt.close()

def plot_pr_curve(recall_results, precision_results, topk_list, out_dir="./"):
    recall_values = [np.mean(recall_results[str(k)]) for k in topk_list]
    precision_values = [np.mean(precision_results[str(k)]) for k in topk_list]

    plt.figure(figsize=(6, 5))
    plt.plot(recall_values, precision_values, marker='o')

    for i, k in enumerate(topk_list):
        plt.text(recall_values[i], precision_values[i], f"k={k}")

    plt.title("Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)

    save_path = out_dir.rstrip("/") + "/pr_curve.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")

    plt.close()
