import torch
import torch.nn.functional as F
from Strategy.Base import BaseRetriever

class DenseRetriever(BaseRetriever):
    
    def __init__(self, tool_names, raw_embeddings):
        """
        Dense 检索
        """
        self.tool_names = tool_names
        self.raw_embeddings = F.normalize(raw_embeddings, p=2, dim=1)

    def search(self, query_vec, top_k):
        
        query_vec = F.normalize(query_vec, p=2, dim=1)

        z = self.raw_embeddings
        sim_query = (z @ query_vec.T).squeeze()

        top_scores, top_idx = torch.topk(sim_query, k=top_k)

        candidates = []
        for s, i in zip(top_scores, top_idx):
            candidates.append({
                "id": i.item(),
                "name": self.tool_names[i],
                "score": s.item()
            })

        print("\n[Stage 1] Dense top candidates:")
        for rank, c in enumerate(candidates):
            print(f"  {rank+1}. {c['name']} ({c['score']:.4f})")

        return candidates
