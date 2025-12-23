import torch


class GraphManager:
    """
    统一图管理器：
    - 方向：adj[parent, child] = 1
    - SGC 聚合时自动取转置：adj_sgc = adj.T
    """

    def __init__(self, num_nodes: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.num_nodes = num_nodes
        self.adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)

    # -----------------------
    # 图更新逻辑
    # -----------------------
    def add_edge(self, parent, child, weight=1.0):
        if parent != child:
            # self.adj[parent, child] += weight
            self.adj[parent, child] = weight


    def update_from_trajectory(self, trajectory: list):
        """
        trajectory = [2,5,7] → 2→5, 5→7
        """
        if len(trajectory) < 2:
            return
        # 转换为 tensor 以确保索引操作的安全性
        parents = torch.tensor(trajectory[:-1], dtype=torch.long, device=self.device)
        children = torch.tensor(trajectory[1:], dtype=torch.long, device=self.device)

        # 取消自环
        mask = parents != children
        if mask.any():
            # self.adj[parents[mask], children[mask]] += 1.0
            self.adj[parents[mask], children[mask]] = 1.0

    # -----------------------
    # 图数据输出（供 SGC + 检索）
    # -----------------------
    def get_sgc_adj(self):
        """
        SGC 需要 child × parent 的入度矩阵
        """
        return self.adj.T

    def get_child_tools(self, parent_tool_id: int):
        """
        返回给定 parent tool 的所有 child tool ids
        """
        adj = self.get_sgc_adj()  # (N, N), child × parent
        child_ids = torch.nonzero(adj[:, parent_tool_id]).squeeze(-1)
        return child_ids.tolist()
