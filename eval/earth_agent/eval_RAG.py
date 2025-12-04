import numpy as np
import yaml
import json
import requests
from typing import Optional, List
from rank_bm25 import BM25Okapi
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


def get_tool_embedding_weighted(tool: dict, w_desc=0.6, w_inputs=0.3, w_outputs=0.1):
    desc_emb = get_embedding(tool["description"])
    inputs_emb = get_embedding(tool["inputs"])
    outputs_emb = get_embedding(tool["outputs"])

    combined = (
        w_desc * desc_emb +
        w_inputs * inputs_emb +
        w_outputs * outputs_emb
    )

    return combined / (np.linalg.norm(combined) + 1e-8)

def build_bm25(tools: List[dict]):
    corpus = [
        f"{tool['description']} {tool['inputs']} {tool['outputs']}"
        for tool in tools
    ]
    tokenized = [text.lower().split() for text in corpus]
    return BM25Okapi(tokenized)


def bm25_score(query: str, bm25):
    return bm25.get_scores(query.lower().split())


def hybrid_scores(query, tool_embeddings, bm25, tool_names, alpha=0.7):
    q_embed = get_embedding(query)

    semantic_scores = np.array([
        cos_similarity(q_embed, tool_embeddings[name])
        for name in tool_names
    ])

    bm25_scores = np.array(bm25_score(query, bm25))
    bm25_scores = (bm25_scores - bm25_scores.min()) / (np.ptp(bm25_scores) + 1e-8)

    final = alpha * semantic_scores + (1 - alpha) * bm25_scores
    return final

def run_tool_rag_experiment(
    groundtruth_path: str,
    tools_path: str,
    topk_list: List[int] = [3, 5, 7],
    weighted: bool = True,
    hybrid: bool = True,
    alpha: float = 0.7
):
    # Load data
    with open(groundtruth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(tools_path, "r", encoding="utf-8") as f:
        tools_list = json.load(f)

    tool_names = [tool["name"] for tool in tools_list]

    # Build tool embeddings
    tool_embeddings = {}
    print("Computing tool embeddings...")

    if weighted:
        for tool in tools_list:
            tool_embeddings[tool["name"]] = get_tool_embedding_weighted(tool)
    else:
        for tool in tools_list:
            tool_embeddings[tool["name"]] = get_embedding(tool["description"])

    # Build BM25
    if hybrid:
        bm25 = build_bm25(tools_list)
    else:
        bm25 = None

    # 保存结果
    recall_results = {str(k): [] for k in topk_list}
    precision_results = {str(k): [] for k in topk_list}

    # 遍历每个问题
    for qid, item in gt_data.items():
        query = item["question"]
        gt_tools = set(item["tools_used"])

        if hybrid:
            scores = hybrid_scores(query, tool_embeddings, bm25, tool_names, alpha)
        else:
            q_emb = get_embedding(query)
            scores = np.array([
                cos_similarity(q_emb, tool_embeddings[name])
                for name in tool_names
            ])

        sorted_idx = np.argsort(scores)[::-1]

        for k in topk_list:
            predicted = set(tool_names[i] for i in sorted_idx[:k])

            hits = len(predicted & gt_tools)

            recall_k = hits / len(gt_tools)
            precision_k = hits / k

            recall_results[str(k)].append(recall_k)
            precision_results[str(k)].append(precision_k)

            print(f"[Q{qid}] Recall@{k}={recall_k:.3f}, Precision@{k}={precision_k:.3f}")

    return (
        recall_results,
        precision_results,
        gt_data,
        tools_list,
        tool_embeddings,
        bm25,
        tool_names
    )

def show_examples(
    gt_data,
    tools_list,
    tool_embeddings,
    bm25,
    tool_names,
    alpha=0.7,
    topk=5,
    num_examples=3
):
    """
    展示 num_examples 个 Tool RAG 的检索案例：
      - query
      - ground truth tools
      - predicted Top-K tools
    """

    print("\n========== 示例案例 (Top-K Tool Retrieval) ==========")

    # 抽样几个问题
    qids = list(gt_data.keys())[:num_examples]

    for qid in qids:
        query = gt_data[qid]["question"]
        gt_tools = set(gt_data[qid]["tools_used"])

        # Hybrid scoring
        scores = hybrid_scores(query, tool_embeddings, bm25, tool_names, alpha)
        sorted_idx = np.argsort(scores)[::-1][:topk]
        pred_tools = [tool_names[i] for i in sorted_idx]

        print("\n-----------------------------------")
        print(f"Q{qid}: {query}")
        print(f"Ground Truth Tools: {list(gt_tools)}")
        print(f"Predicted Top-{topk}: {pred_tools}")

        # 命中情况
        hit = set(pred_tools) & gt_tools
        print(f"Hits: {list(hit)}  (Hit {len(hit)}/{len(gt_tools)})")

        # 显示每个 tool 的得分（可选）
        print("\nScores:")
        for i in sorted_idx:
            print(f"  {tool_names[i]}: {scores[i]:.4f}")

    print("\n=====================================================\n")

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

if __name__ == "__main__":
    (recall_results,
     precision_results,
     gt_data,
     tools_list,
     tool_embeddings,
     bm25,
     tool_names) = run_tool_rag_experiment(
        "./toolRAG_ground_truth.json",
        "./earth_tools.json",
        topk_list=[4, 5, 6],
        weighted=True,
        hybrid=True,
        alpha=0.7
    )

    show_examples(
        gt_data=gt_data,
        tools_list=tools_list,
        tool_embeddings=tool_embeddings,
        bm25=bm25,
        tool_names=tool_names,
        alpha=0.7,
        topk=5,
        num_examples=5
    )
    plot_recall_precision(recall_results, precision_results, [4, 5, 6], out_dir="./plots")
    plot_pr_curve(recall_results, precision_results, [4, 5, 6], out_dir="./plots")
