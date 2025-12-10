import torch
import torch.nn.functional as F
from collections import defaultdict


class SGCRetriever:

    def __init__(self, graph_manager, tool_names, raw_embeddings, alpha=0.2):
        """
        graph_manager: GraphManager 实例
        raw_embeddings: (N, D)
        alpha: SGC 融合比例
        """
        self.graph = graph_manager
        self.tool_names = tool_names
        self.raw_embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        self.alpha = alpha

        self.final_embeddings = self.compute_sgc_embeddings()

    # -------------------------------------------------
    # 生成 SGC 特征
    # -------------------------------------------------
    def compute_sgc_embeddings(self):
        adj = self.graph.get_sgc_adj()                 # child × parent
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        adj_norm = adj / row_sum                       # 归一化

        context = adj_norm @ self.raw_embeddings       # (N,N) × (N,D)

        has_parent = (adj.sum(dim=1, keepdim=True) > 0).float()

        z = (1 - self.alpha * has_parent) * self.raw_embeddings + \
            (self.alpha * has_parent) * context

        return F.normalize(z, p=2, dim=1)

    # -------------------------------------------------
    # Stage 1: 基于 SGC 的粗排
    # -------------------------------------------------
    def search(self, query_vec, top_k=5, conflict_threshold=0.1, avoid_names=None):

        avoid_names = avoid_names or []
        query_vec = F.normalize(query_vec, p=2, dim=1)

        # 计算所有工具的相似度分数
        scores = (self.final_embeddings @ query_vec.T).squeeze()

        if avoid_names:
            for bad_name in avoid_names:
                try:
                    bad_idx = self.tool_names.index(bad_name)
                    scores[bad_idx] = -float('inf')

                except ValueError:
                    pass

        top_scores, top_idx = torch.topk(scores, k=top_k)

        candidates = []
        for s, i in zip(top_scores, top_idx):
            candidates.append({
                "id": i.item(),
                "name": self.tool_names[i],
                "score": s.item(),
                "vec": self.final_embeddings[i]
            })

        print("\n[Stage 1] SGC top candidates:")
        for rank, c in enumerate(candidates):
            print(f"  {rank+1}. {c['name']} ({c['score']:.4f})")

        # Stage 2：兄弟抑制
        final = self.apply_sibling_inhibition(query_vec, candidates, conflict_threshold)

        return final

    # -------------------------------------------------
    # Stage 2: 兄弟竞争 → 抑制
    # -------------------------------------------------
    def apply_sibling_inhibition(self, query_vec, candidates, threshold):

        # 1. 获取 child → parents 映射
        child_to_parents = self.graph.get_parent_map()

        clusters = defaultdict(list)
        # [优化] 使用 set 记录已经处理过的 (candidate_id_A, candidate_id_B)
        # 避免同一个 A, B 因为共享多个父节点而被多次 PK
        processed_pairs = set()

        # 2. 构建父节点簇
        for c in candidates:
            parents = child_to_parents.get(c["id"], [])
            # 如果是孤立节点，跳过
            if not parents:
                continue
            for p in parents:
                clusters[p].append(c)

        rerank_needed = False

        # 3. 对每个父节点簇执行兄弟 PK
        for pid, group in clusters.items():
            if len(group) < 2:
                continue

            # 必须排序！确保 group[0] 是当前最高分
            group.sort(key=lambda x: x["score"], reverse=True)

            leader = group[0]
            challenger = group[1]

            # [优化] 简化去重逻辑：只看两个工具 ID，不看父节点 ID
            # 无论它们共享几个父亲，向量差异是固定的，PK一次就够了
            pair_sig = tuple(sorted((leader["id"], challenger["id"])))
            if pair_sig in processed_pairs:
                continue
            processed_pairs.add(pair_sig)

            diff = leader["score"] - challenger["score"]
            if diff > threshold:
                continue

            print(f"\n[Conflict Detected] Parent {self.tool_names[pid]}")
            print(f"  Leader: {leader['name']} ({leader['score']:.4f})")
            print(f"  Challenger: {challenger['name']} ({challenger['score']:.4f})")

            # [Fix] 移除归一化
            # 我们需要真实的差异幅度。如果差异极小(噪声)，归一化会放大噪声导致随机反转。
            v_diff = leader["vec"] - challenger["vec"]
            # v_diff = F.normalize(v_diff, p=2, dim=0) <--- DELETED

            # 计算投影
            # query_vec: [1, D], v_diff: [D]
            # 这里的 .item() 会自动处理标量转换
            dir_score = (query_vec @ v_diff).item()
            print(f"  Sim(Query, V_diff) = {dir_score:.4f}")

            if dir_score < 0:
                print(f"  -> Challenger {challenger['name']} WINS, boosting score!")
                # 赋予微小的优势，完成逆转
                challenger["score"] = leader["score"] + 1e-4
                rerank_needed = True
            else:
                print(f"  -> Leader holds position.")

        # 4. 若发生变化 → 全局重排
        if rerank_needed:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            print("\n[Re-ranking Final Result]:")
            for i, c in enumerate(candidates):
                print(f"  {i + 1}. {c['name']} ({c['score']:.4f})")

        return candidates
