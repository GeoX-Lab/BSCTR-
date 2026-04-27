import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from Strategy.Base import BaseRetriever

class GraphFusionRetriever(BaseRetriever):
    """
    基于 Graph RAG-Tool Fusion 的混合图检索器
    """
    def __init__(self, dense_retriever: BaseRetriever, tool_graph, tool_names: List[str], d_limit: int = 3, final_top_k: int = 10):

        self.dense_retriever = dense_retriever
        self.tool_graph = tool_graph
        self.tool_names = tool_names
        self.d_limit = d_limit
        self.final_top_k = final_top_k
        self.name_to_id = {name: idx for idx, name in enumerate(tool_names)}

    def search(self, query_vec: torch.Tensor, top_k: int) -> List[Dict[str, Any]]:

        initial_candidates = self.dense_retriever.search(query_vec, top_k)
        final_candidates =[]
        seen_names = set() 
        
        print("\n[Stage 2] Graph Traversal Expansion:")
        
        for candidate in initial_candidates:
            c_name = candidate["name"]
            c_score = candidate["score"]
            
            if c_name not in seen_names:
                final_candidates.append(candidate)
                seen_names.add(c_name)

            dependencies = self.tool_graph.dfs_search(start_tool=c_name, d_limit=self.d_limit)
            
            if dependencies:
                print(f"  -> '{c_name}' expanded dependencies: {dependencies}")
                
            for dep_name in dependencies:
                if dep_name not in seen_names and dep_name in self.name_to_id:
                    dep_score = c_score * 0.95 
                    
                    final_candidates.append({
                        "id": self.name_to_id[dep_name],
                        "name": dep_name,
                        "score": dep_score
                    })
                    seen_names.add(dep_name)

            if len(final_candidates) >= self.final_top_k:
                break
        final_candidates = final_candidates[:self.final_top_k]
        
        print("\n[Final Stage] Graph Fusion top candidates:")
        for rank, c in enumerate(final_candidates):
            print(f"  {rank+1}. {c['name']} (id: {c['id']}, score: {c['score']:.4f})")
            
        return final_candidates