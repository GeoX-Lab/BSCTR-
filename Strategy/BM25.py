import numpy as np
from rank_bm25 import BM25Okapi
from Strategy.Base import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, tool_names, tool_descriptions):
        """
        BM25 检索
        """
        self.tool_names = tool_names
        self.tool_descriptions = tool_descriptions
        tokenized_corpus = [doc.lower().split() for doc in tool_descriptions]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query_text, top_k):

        tokenized_query = query_text.lower().split()

        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_idx]

        candidates = []
        for s, i in zip(top_scores, top_idx):
            candidates.append({
                "id": int(i),                 
                "name": self.tool_names[i],   
                "score": float(s)        
            })

        print(f"\n[Baseline: BM25] Top candidates for query: '{query_text[:20]}...'")
        for rank, c in enumerate(candidates):
            print(f"  {rank+1}. {c['name']} ({c['score']:.4f})")

        return candidates