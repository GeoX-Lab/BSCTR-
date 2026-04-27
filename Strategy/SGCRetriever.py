import torch
import torch.nn.functional as F
from Strategy.Base import BaseRetriever

class SGCRetriever(BaseRetriever):
    
    def __init__(self, graph_manager, tool_names, raw_embeddings, alpha=0.2, mode='f'):
        """
        SGC 检索
        :param mode: 聚合模式, 'f' (forward, 父聚合到子), 'b' (backward, 子聚合到父), 's' (symmetric, 双向)
        """
        self.graph = graph_manager
        self.tool_names = tool_names
        self.raw_embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        self.alpha = alpha
        self.mode = mode

    def compute_sgc_embeddings(self, k=1):

        A = self.graph.get_sgc_adj()
        num_nodes = A.shape[0]
        identity = torch.eye(num_nodes, device=A.device)

        if self.mode == 'f':
            adj_hat = A + identity
        elif self.mode == 'b':
            adj_hat = A.t() + identity
        elif self.mode == 's':
            adj_hat = A + A.t() + identity
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from 'f', 'b', 's'.")

        row_sum = adj_hat.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        adj_norm = adj_hat / row_sum  
        

        context = self.raw_embeddings
        for _ in range(k):
            context = adj_norm @ context

        z = (1 - self.alpha) * self.raw_embeddings + (self.alpha) * context

        return F.normalize(z, p=2, dim=1)

    def search(self, query_vec, top_k=10):

        query_vec = F.normalize(query_vec, p=2, dim=1)

        z = self.compute_sgc_embeddings()
        
        sim_query = (z @ query_vec.T).squeeze()

        top_scores, top_idx = torch.topk(sim_query, k=top_k)

        candidates = []
        for s, i in zip(top_scores, top_idx):
            candidates.append({
                "id": i.item(),
                "name": self.tool_names[i],
                "score": s.item()
            })

        print(f"\n[Stage 1] SGC top candidates (Mode: {self.mode}):")
        for rank, c in enumerate(candidates):
            print(f"  {rank+1}. {c['name']} ({c['score']:.4f})")

        return candidates
    