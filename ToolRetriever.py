import torch
import torch.nn as nn
import torch.nn.functional as F


class ToolGraphRetriever(nn.Module):
    """
    工具图检索模块：
    - SGC 拓扑特征传播 + 残差
    - 线性适配层 W_adapter (初始化为单位阵) 实现兄弟抑制 / 动态路径增强
    """

    def __init__(self, input_dim, alpha=0.2, lambda_reg=0.01):
        """
        Args:
            input_dim: 工具/查询 Embedding 维度 (e.g., 768)
            alpha: SGC 平滑系数 (保留自身语义 vs 聚合父节点)
            lambda_reg: W_adapter 正则化系数
        """
        super().__init__()
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)
        self.input_dim = input_dim

        # W_adapter: 线性投影层，初始化为单位矩阵，保证初始状态空间不变
        self.W_adapter = nn.Parameter(torch.eye(input_dim))

        # 缓存归一化后的邻接矩阵 D^{-1}(A + I)
        self.normalized_adj = None

    @torch.no_grad()
    def compute_normalized_adj(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算行归一化矩阵: D_{in}^{-1}(A+I)
        adj_matrix: (N, N) 0/1 矩阵
        假设 adj[row, col] = 1 表示 col -> row (col 是 row 的父节点/前驱)
        """
        device = adj_matrix.device
        dtype = adj_matrix.dtype
        N = adj_matrix.shape[0]

        # 添加自环 (A + I)
        A_hat = adj_matrix + torch.eye(N, device=device, dtype=dtype)

        # 入度 d_in[i] = sum_j A_hat[i, j]
        d_in = A_hat.sum(dim=1)  # (N,)
        d_in_inv = d_in.pow(-1)
        d_in_inv[torch.isinf(d_in_inv)] = 0.0  # 对于全 0 行，设为 0

        D_inv = torch.diag(d_in_inv)

        # D^{-1}(A + I)
        self.normalized_adj = D_inv @ A_hat
        return self.normalized_adj

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 原始工具 Embeddings (N, D)
            adj_matrix: 邻接矩阵 (N, N)，当图结构发生变化时需要传入以重新计算
        Returns:
            z_final: 经过 SGC + Adapter 的工具表征 (N, D)
        """
        if self.normalized_adj is None or adj_matrix is not None:
            if adj_matrix is None:
                raise ValueError("Initial adjacency matrix required at first forward.")
            self.compute_normalized_adj(adj_matrix)

        # 1. SGC Propagation
        # Z_sgc = (1 - alpha) X + alpha * D^{-1}(A + I) X
        prop_feat = self.normalized_adj @ x
        z_sgc = (1.0 - self.alpha) * x + self.alpha * prop_feat

        # 2. Linear Adapter Projection
        # z_final = z_sgc W_adapter
        z_final = z_sgc @ self.W_adapter

        return z_final

    def compute_loss(
            self,
            query_vec: torch.Tensor,
            pos_node_idx,
            neg_node_idx,
            all_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 N-Pair Loss + 正则项
        Args:
            query_vec: (D,) 或 (B, D) 查询向量
            pos_node_idx: int / LongTensor[(B,)] 正样本
            neg_node_idx: List[int] 或 LongTensor[(B, K)] 负样本
            all_embeddings: (N, D)
        """

        device = all_embeddings.device

        # --- 处理 positive index ---
        pos_idx = torch.as_tensor(pos_node_idx, device=device, dtype=torch.long).view(-1)

        # --- 处理 negative index (允许多个负样本) ---
        neg_idx = torch.as_tensor(neg_node_idx, device=device, dtype=torch.long)
        if neg_idx.dim() == 1:
            neg_idx = neg_idx.unsqueeze(0)  # (1, K) 或 (B, K)

        # --- 处理 query_vec 维度 ---
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)  # (1, D)

        B = pos_idx.size(0)
        if query_vec.size(0) == 1 and B > 1:
            query_vec = query_vec.expand(B, -1)

        # --- 获取 embeddings ---
        pos_emb = all_embeddings[pos_idx]  # (B, D)
        neg_emb = all_embeddings[neg_idx]  # (B, K, D)

        # --- L2 normalize ---
        q = F.normalize(query_vec, p=2, dim=-1)  # (B, D)
        p = F.normalize(pos_emb, p=2, dim=-1)  # (B, D)
        n = F.normalize(neg_emb, p=2, dim=-1)  # (B, K, D)

        # --- 计算相似度 ---
        # 正样本： q · p
        pos_sim = torch.sum(q * p, dim=-1, keepdim=True)  # (B, 1)

        # 负样本： q · n_j
        neg_sim = torch.sum(q.unsqueeze(1) * n, dim=-1)  # (B, K)

        # --- 拼接成 logits: [pos | neg1 | neg2 | ...] ---
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+K)

        # CrossEntropyLoss 默认会对 dim=1 做 softmax
        labels = torch.zeros(B, dtype=torch.long, device=device)
        loss_npairs = F.cross_entropy(logits, labels)

        # --- 正则项: ||W - I||^2 ---
        identity = torch.eye(self.input_dim, device=self.W_adapter.device, dtype=self.W_adapter.dtype)
        loss_reg = ((self.W_adapter - identity) ** 2).sum()

        total_loss = loss_npairs + self.lambda_reg * loss_reg
        return total_loss
