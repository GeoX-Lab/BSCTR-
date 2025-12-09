import torch
from collections import defaultdict


class GraphManager:
    """
    统一图管理器：
    - 方向：adj[parent, child] = 1
    - SGC 聚合时自动取转置：adj_sgc = adj.T
    """

    def __init__(self, num_nodes: int, device="cpu"):
        self.device = device
        self.num_nodes = num_nodes
        self.adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)

    # -----------------------
    # 图更新逻辑
    # -----------------------
    def add_edge(self, parent: int, child: int):
        if parent != child:
            self.adj[parent, child] = 1.0

    def update_from_trajectory(self, trajectory: list):
        """
        trajectory = [2,5,7] → 2→5, 5→7
        """
        if len(trajectory) < 2:
            return
        # [Fix] 转换为 tensor 以确保索引操作的安全性
        parents = torch.tensor(trajectory[:-1], dtype=torch.long, device=self.device)
        children = torch.tensor(trajectory[1:], dtype=torch.long, device=self.device)
        self.adj[parents, children] = 1.0

    # -----------------------
    # 图数据输出（供 SGC + 检索）
    # -----------------------
    def get_sgc_adj(self):
        """
        SGC 需要 child × parent 的入度矩阵
        """
        return self.adj.T

    def get_parent_map(self):
        """
        [Fix] 新增此方法，供 Retriever 的 apply_sibling_inhibition 使用
        返回 {child_id: [parent_id1, ...]}
        """
        # adj.T 的行索引是 child，列索引是 parent
        # nonzero 返回 (child_indices, parent_indices)
        child_indices, parent_indices = torch.nonzero(self.adj.T, as_tuple=True)

        child_to_parents = defaultdict(list)
        for c, p in zip(child_indices.tolist(), parent_indices.tolist()):
            child_to_parents[c].append(p)

        return child_to_parents

    def get_sibling_map(self):
        """
        (备用) 如果两个 child 拥有同一个 parent，则它们互为兄弟。
        """
        sibling_map = defaultdict(set)
        parent_indices = torch.nonzero(self.adj.sum(dim=1) > 0, as_tuple=True)[0]

        for p in parent_indices:
            children = torch.nonzero(self.adj[p], as_tuple=True)[0].tolist()
            if len(children) > 1:
                for c in children:
                    sibs = set(children)
                    sibs.remove(c)
                    sibling_map[c].update(sibs)

        return {k: list(v) for k, v in sibling_map.items()}