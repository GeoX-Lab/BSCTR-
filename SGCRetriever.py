import torch
import torch.nn.functional as F


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
