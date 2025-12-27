import torch
import torch.nn.functional as F


# 向量SGC
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
        # adj = torch.log1p(adj)  # 平滑频次
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        adj_norm = adj / row_sum                       # 归一化

        context = adj_norm @ self.raw_embeddings       # (N,N) × (N,D)

        has_parent = (adj.sum(dim=1, keepdim=True) > 0).float()

        z = (1 - self.alpha * has_parent) * self.raw_embeddings + \
            (self.alpha * has_parent) * context

        return F.normalize(z, p=2, dim=1)

    def search(self, query_vec, top_k=10, pre_tool=None):

        query_vec = F.normalize(query_vec, p=2, dim=1)

        z = self.compute_sgc_embeddings()
        # 计算所有工具的相似度分数
        sim_query = (z @ query_vec.T).squeeze()
        # 计算先前工具的子节点的相似分数
        # sim_pre_tool = torch.zeros_like(sim_query)

        # if pre_tool is not None:
        #     prev_vec = z[pre_tool]
        #     sim_pre_tool = (z @ prev_vec.unsqueeze(1)).squeeze()
        #
        # w_q = 0.7
        # w_c = 0.3 if pre_tool is not None else 0.0

        # scores = w_q * sim_query + w_c * sim_pre_tool

        top_scores, top_idx = torch.topk(sim_query, k=top_k)

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

        final = candidates
        return final

# 分数SGC

# import torch
# import torch.nn.functional as F
#
#
# class SGCRetriever:
#     def __init__(self, graph_manager, tool_names, raw_embeddings, alpha=0.3):
#         """
#         alpha: 父节点分数对子节点的影响权重 (0.0 - 1.0)
#                alpha 越大，检索到父节点时，子节点排位升得越高
#         """
#         self.graph = graph_manager
#         self.tool_names = tool_names
#         self.raw_embeddings = F.normalize(raw_embeddings, p=2, dim=1)
#         self.alpha = alpha
#
#     def search(self, query_vec, top_k):
#         # --- 步骤 1: 计算所有节点的“原始”相似度 ---
#         query_vec = F.normalize(query_vec, p=2, dim=1)
#
#         # sim_raw: (N, )  每个工具自己的单打独斗得分
#         sim_raw = (self.raw_embeddings @ query_vec.T).squeeze()
#
#         # 确保分数为正数（ReLU），防止父节点的负相关性（不相似）拉低子节点
#         sim_raw = F.relu(sim_raw)
#
#         # --- 步骤 2: 获取父子关系矩阵 ---
#         # adj 形状为 (N_child, N_parent)，adj[i][j]=1 表示 j 是 i 的父亲
#         adj = self.graph.get_sgc_adj()
#
#         # --- 步骤 3: 分数传播 (Parent -> Child) ---
#         if adj.shape[0] > 0:
#             # 计算每个节点从父节点那里继承来的分数
#             # 矩阵乘法逻辑：对于每个 Child，把它的所有 Parent 的 sim_raw 加起来
#             inherited_score = adj @ sim_raw  # (N, )
#
#             # 归一化（可选）：如果一个孩子有多个爹，取平均值而不是累加，防止分数爆炸
#             degree = adj.sum(dim=1)
#             degree[degree == 0] = 1.0
#             inherited_score = inherited_score / degree
#
#             # --- 步骤 4: 最终融合 ---
#             # 这里的逻辑是：你自己的分 + 你爹的分 * 权重
#             # 这样既保留了子节点的独立性，又实现了“父节点召回 -> 子节点跟上”
#             final_scores = sim_raw + self.alpha * inherited_score
#
#         else:
#             final_scores = sim_raw
#
#         # --- 步骤 5: 排序与返回 ---
#         top_scores, top_idx = torch.topk(final_scores, k=top_k)
#
#         candidates = []
#         for s, i in zip(top_scores, top_idx):
#             candidates.append({
#                 "id": i.item(),
#                 "name": self.tool_names[i],
#                 "score": s.item(),
#                 "raw_score": sim_raw[i].item(),
#                 "vec": self.raw_embeddings[i]
#             })
#
#         # 打印调试信息，验证你的想法是否实现
#         print("\n[Stage 1] Retrieval Result (Score Propagation):")
#         for rank, c in enumerate(candidates):
#             boost = c['score'] - c['raw_score']
#             print(
#                 f"  {rank + 1}. {c['name']} | Final: {c['score']:.4f} (Base: {c['raw_score']:.4f} + ParentBoost: {boost:.4f})")
#
#         return candidates
