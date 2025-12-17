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

    def compute_sgc_embeddings(self):
        adj = self.graph.get_sgc_adj()  # child × parent
        adj = torch.log1p(adj)  # 平滑频次
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        adj_norm = adj / row_sum                       # 归一化

        context = adj_norm @ self.raw_embeddings       # (N,N) × (N,D)

        has_parent = (adj.sum(dim=1, keepdim=True) > 0).float()

        z = (1 - self.alpha * has_parent) * self.raw_embeddings + \
            (self.alpha * has_parent) * context

        return F.normalize(z, p=2, dim=1)

    def search(self, query_vec, top_k=5, conflict_threshold=0.1, avoid_names=None, pre_tool=None):

        avoid_names = avoid_names or []
        query_vec = F.normalize(query_vec, p=2, dim=1)

        z = self.compute_sgc_embeddings()
        # 计算所有工具的相似度分数
        sim_query = (z @ query_vec.T).squeeze()
        # 计算先前工具的子节点的相似分数
        sim_pre_tool = torch.zeros_like(sim_query)

        if pre_tool is not None:
            prev_vec = z[pre_tool]
            sim_pre_tool = (z @ prev_vec.unsqueeze(1)).squeeze()

        w_q = 0.7
        w_c = 0.3 if pre_tool is not None else 0.0

        scores = w_q * sim_query + w_c * sim_pre_tool

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
                "vec": z[i]
            })

        print("\n[Stage 1] SGC top candidates:")
        for rank, c in enumerate(candidates):
            print(f"  {rank+1}. {c['name']} ({c['score']:.4f})")

        # Stage 2：兄弟抑制
        # final = self.apply_sibling_inhibition(query_vec, candidates, conflict_threshold)
        final = candidates
        return final

    # -------------------------------------------------
    # Stage 2: 兄弟竞争 → 抑制
    # -------------------------------------------------
    def apply_sibling_inhibition(self, query_vec, candidates, threshold=0.1):
        """
        方向增强的兄弟抑制策略：
        - 只在兄弟分数差 >= threshold 时触发
        - 使用 query 与 (leader - challenger) 的余弦方向性判断
        - 根据方向判断翻转顺序（模拟语义优先级）
        """

        child_to_parents = self.graph.get_parent_map()

        # 构建兄弟簇 {parent -> [child_candidates]}
        clusters = defaultdict(list)
        for c in candidates:
            cid = c["id"]
            parents = child_to_parents.get(cid, [])

            for p in parents:
                clusters[p].append(c)

        rerank_needed = False

        print("\n[Stage 2] Sibling inhibition (direction-based):")

        # 遍历每个父节点簇
        for parent_id, group in clusters.items():

            if len(group) < 2:
                continue

            # 按当前分数排序
            group.sort(key=lambda x: x["score"], reverse=True)

            leader = group[0]
            challenger = group[1]

            diff = abs(leader["score"] - challenger["score"])

            print(f"\n  Parent: {self.tool_names[parent_id]}")
            print(f"    Leader:     {leader['name']} ({leader['score']:.4f})")
            print(f"    Challenger: {challenger['name']} ({challenger['score']:.4f})")
            print(f"    Score diff = {diff:.4f}")

            # ---- 触发条件：兄弟差值 >= 阈值 ----
            if diff < threshold:
                print("    -> diff < threshold，保持原顺序")
                continue

            print("    -> diff >= threshold，进行方向判定")

            # ===== 核心逻辑：方向判断 =====
            v_diff = (leader["vec"] - challenger["vec"]).unsqueeze(1)  # (D,1)
            directional_score = (query_vec @ v_diff).item()

            print(f"    direction(query, leader - challenger) = {directional_score:.4f}")

            # 如果方向说明 challenger 更适合任务 → 翻转顺序
            if directional_score < 0:
                print(f"    -> Challenger '{challenger['name']}' wins, flip scores")

                # 翻转得分（更稳定的方式）
                leader["score"], challenger["score"] = (
                    challenger["score"],
                    leader["score"]
                )

                rerank_needed = True

            else:
                print("    -> Leader 保持领先")

        # 全局重新排序
        if rerank_needed:
            candidates.sort(key=lambda x: x["score"], reverse=True)

            print("\n[Re-ranked result after sibling inhibition]:")
            for i, c in enumerate(candidates):
                print(f"  {i + 1}. {c['name']} ({c['score']:.4f})")

        return candidates

