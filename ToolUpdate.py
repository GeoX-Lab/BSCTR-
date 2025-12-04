import torch

class GraphUpdater:
    """
    基于成功的工具调用轨迹更新工具图结构：
    - trajectory = [t1, t2, t3, ...]
    - 添加边：t1 -> t2, t2 -> t3, ...
    - 自动 dedupe（不重复加边）
    - 自动建立兄弟节点（共享父节点）
    - 保持图稀疏性（不交叉添加错误边）
    - retriever 训练阶段不调用本类，训练期间图固定
    """

    def __init__(self, adj_matrix: torch.Tensor):
        """
        adj_matrix: (N, N) 工具图邻接矩阵
                    adj[i, j] = 1 表示 i -> j（i 是父节点）
        """
        self.adj_matrix = adj_matrix

    def add_edge(self, parent: int, child: int):
        """
        添加有向边 parent -> child
        避免重复写入
        """
        if self.adj_matrix[parent, child] == 0:
            self.adj_matrix[parent, child] = 1

    def update_from_trajectory(self, trajectory):
        """
        根据成功工具链更新图结构，例如：
        trajectory = [2, 5, 7, 9]
        添加边：
            2->5, 5->7, 7->9
        """
        if len(trajectory) < 2:
            return

        for i in range(len(trajectory) - 1):
            parent = trajectory[i]
            child = trajectory[i + 1]
            self.add_edge(parent, child)

    def get_parents(self, node_idx: int):
        """
        返回所有指向 node_idx 的父节点下标
        """
        return (self.adj_matrix[:, node_idx] > 0).nonzero(as_tuple=True)[0].tolist()

    def get_siblings(self, node_idx: int):
        """
        所有与 node_idx 共享父节点的工具，即兄弟节点
        """
        parents = self.get_parents(node_idx)
        siblings = set()
        for p in parents:
            children = (self.adj_matrix[p, :] > 0).nonzero(as_tuple=True)[0]
            for c in children.tolist():
                if c != node_idx:
                    siblings.add(c)
        return list(siblings)