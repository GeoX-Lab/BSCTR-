import torch
from typing import List
from ToolRetriever import ToolGraphRetriever


class SubtaskToolGraphTrainer:
    """
    基于 subtask 的工具检索训练器：
    - 每个 subtask 作为单独的训练样本
    - 使用 N-Pair Loss (from compute_loss)
    - 正样本是正确工具
    - 负样本是兄弟节点（若无则跳过）
    - 不再更新图结构
    """

    def __init__(
        self,
        model: ToolGraphRetriever,
        tool_embeddings: torch.Tensor,
        adj_matrix: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.tool_embeddings = tool_embeddings  # (N, D) 工具基础语义（冻结）
        self.adj_matrix = adj_matrix            # (N, N) 图结构（只用于找兄弟，不训练）
        self.optimizer = optimizer

    def get_siblings(self, node_idx: int) -> List[int]:
        """
        获取当前工具的兄弟节点：与 node_idx 具有相同父节点的工具。
        假设 adj[row, col] = 1 表示 col -> row (col 是父节点)
        """
        parents = (self.adj_matrix[:, node_idx] > 0).nonzero(as_tuple=True)[0]

        siblings = set()
        for p in parents.tolist():
            children = (self.adj_matrix[p, :] > 0).nonzero(as_tuple=True)[0]
            for c in children.tolist():
                if c != node_idx:
                    siblings.add(c)

        return list(siblings)

    def train_step_from_subtasks(
        self,
        subtask_embs: torch.Tensor,   # (B, D)
        tool_indices: torch.Tensor,   # (B,)
    ):
        """
        使用一组 (subtask_emb, tool_idx) 做一次训练。
        每个 subtask 是一个训练样本。
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 计算当前图结构下的工具表征（SGC + Adapter）
        with torch.no_grad():
            current_embeddings = self.model(self.tool_embeddings, self.adj_matrix)

        total_loss = 0.0
        steps = 0

        B = tool_indices.size(0)

        for i in range(B):
            q_vec = subtask_embs[i]             # (D,)
            pos_idx = tool_indices[i].item()    # int

            # 找到全部兄弟节点作为负样本
            siblings = self.get_siblings(pos_idx)

            # ⭐没有兄弟 → 不训练这个样本
            if len(siblings) == 0:
                continue

            neg_idx = torch.tensor(
                siblings, dtype=torch.long, device=current_embeddings.device
            )

            # 调用 retriever 内部的 N-Pair Loss
            loss = self.model.compute_loss(
                query_vec=q_vec,                 # (D,)
                pos_node_idx=pos_idx,            # int
                neg_node_idx=neg_idx,            # (K,)
                all_embeddings=current_embeddings,
            )

            total_loss += loss
            steps += 1

        if steps > 0:
            avg_loss = total_loss / steps
            avg_loss.backward()
            self.optimizer.step()
            return avg_loss.item()

        return 0.0
