import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import requests
import yaml
from collections import deque
from Base import ToolNode, ToolEdge


class ToolMem:
    """
    Agent 的 Tool-memory（图：节点=工具；边=逻辑/经验）
    """

    def __init__(self, tool_node_path = "./tools_graph/node.json", tool_edge_path = "./tools_graph/edge.json"):
        self.tool_node: Dict[str, Any] = {}
        self.tool_edge: Dict[str, Any] = {}
        self.tool_node_path = tool_node_path
        self.tool_edge_path = tool_edge_path

    # ---------- 基础 IO ----------
    def load_tools_from_json_list(self, path: str):

        with open(path, "r", encoding="utf-8") as f:
            json_list = json.load(f)
        for item in json_list:
            name = item.get("name") or item.get("tool_name")
            if not name:
                continue
            vector = self._get_embedding(
                item.get("description", "")
            )
            self.tool_node[name] = {
                "name": name,
                "description": item.get("description", ""),
                "inputs": item.get("inputs", ""),
                "outputs": item.get("outputs", ""),
                "feedback": "",
                "vector": vector
            }

        if self.tool_node_path:
            self.save_nodes_to_file()
        return f"{len(json_list)} tools loaded and saved to {self.tool_node_path}"

    def save_nodes_to_file(self) -> str:
        """保存工具节点到文件（以工具名为 key）"""
        serializable_nodes = {}
        for tool_name, node in self.tool_node.items():
            if isinstance(node, ToolNode):
                vector = node.vector.tolist() if isinstance(node.vector, np.ndarray) else node.vector
                serializable_nodes[tool_name] = {
                    "name": tool_name,
                    "description": node.description,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "feedback": node.feedback,
                    "vector": vector,
                }
            else:
                # dict 形态，确保 vector 可被序列化
                vector = node.get("vector", None)
                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                node_copy = dict(node)
                node_copy["vector"] = vector
                # name 字段补齐为 key
                node_copy["name"] = tool_name
                serializable_nodes[tool_name] = node_copy

        with open(self.tool_node_path, "w", encoding="utf-8") as f:
            json.dump(serializable_nodes, f, ensure_ascii=False, indent=2)

        return f"Tool nodes saved to {self.tool_node_path}"

    def save_edges_to_file(self) -> str:
        """保存工具边到文件"""
        serializable_edges = {}
        for edge_key, e in self.tool_edge.items():
            if isinstance(e, ToolEdge):
                serializable_edges[edge_key] = {
                    "start_node": e.start_node,
                    "end_node": e.end_node,
                    "messages": e.messages,
                    "states": e.status,
                    "timestamp": e.timestamp,
                    "weights": e.weights,
                }
            else:
                # dict 形态
                serializable_edges[edge_key] = {
                    "start_node": e.get("start_node"),
                    "end_node": e.get("end_node"),
                    "messages": e.get("messages", []),
                    "states": e.get("status", e.get("states", 0)),
                    "timestamp": e.get("timestamp"),
                    "weights": e.get("weights"),
                }

        with open(self.tool_edge_path, "w", encoding="utf-8") as f:
            json.dump(serializable_edges, f, ensure_ascii=False, indent=2)

        return f"Tool edges saved to {self.tool_edge_path}"

    def _to_name_indexed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一成 {tool_name: node_dict}
        """
        name_indexed: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                name = v.get("name") or v.get("tool_name")
                if not name:
                    # 若没有 name，退化用 k；但强烈建议你的节点里都有 name 字段
                    name = k
                # 向量统一成 np.ndarray，便于相似度计算（保存时再转 list）
                vec = v.get("vector")
                if vec is not None and not isinstance(vec, np.ndarray):
                    try:
                        vec = np.array(vec, dtype=float)
                    except Exception:
                        vec = None
                node = dict(v)
                node["name"] = name
                node["vector"] = vec
                name_indexed[name] = node
            else:
                # 如果是 ToolNode 对象
                if isinstance(v, ToolNode):
                    name_indexed[v.name] = v
        return name_indexed

    def get_node_from_doc(self) -> Dict[str, Any]:
        """
        从文件中读取 tool_node 信息，并标准化为 {tool_name: node}
        """
        with open(self.tool_node_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.tool_node = self._to_name_indexed(raw)
        return self.tool_node

    def get_edge_from_doc(self) -> Dict[str, Any]:
        """
        从文件中读取 tool_edge 信息
        """
        with open(self.tool_edge_path, "r", encoding="utf-8") as f:
            self.tool_edge = json.load(f)
        return self.tool_edge

    # ---------- 节点/边管理 ----------
    def add_node(self, new_node: ToolNode) -> str:
        """
        添加 tool 节点（以工具名为 key）
        """
        node_dict = {
            "name": new_node.name,
            "description": new_node.description,
            "inputs": new_node.inputs,
            "outputs": new_node.outputs,
            "feedback": new_node.feedback,
            "vector": new_node.vector.tolist() if isinstance(new_node.vector, np.ndarray) else new_node.vector,
        }

        # 先读出文件已有节点
        with open(self.tool_node_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

        # 统一成以工具名为 key 的结构后更新
        existing_name_indexed = self._to_name_indexed(existing)
        existing_name_indexed[new_node.name] = node_dict

        # 写回
        with open(self.tool_node_path, "w", encoding="utf-8") as f:
            json.dump(existing_name_indexed, f, ensure_ascii=False, indent=2)

        # 同步内存
        self.tool_node = self._to_name_indexed(existing_name_indexed)
        return f"New Tool Node '{new_node.name}' saved to {self.tool_node_path}!"

    def _edge_weights(self, e: Any) -> Optional[float]:
        if isinstance(e, ToolEdge):
            return e.weights
        if isinstance(e, dict):
            return e.get("weights")
        return None

    def add_edge(self, start_node: str, end_node: str, messages: List[str], timestamp: Optional[str] = None) -> str:
        """
        添加逻辑边（由 LLM 生成）
        """
        if start_node not in self.tool_node:
            return f"Source tool '{start_node}' not found!"
        if end_node not in self.tool_node:
            return f"Target tool '{end_node}' not found!"

        edge_key = f"{start_node}->{end_node}"
        if edge_key in self.tool_edge:
            return f"Edge '{edge_key}' already exists!"

        new_edge = ToolEdge(start_node=start_node, end_node=end_node, messages=messages, timestamp=timestamp, status=0)
        self.tool_edge[edge_key] = new_edge

        self.save_edges_to_file()
        return f"Logical edge '{edge_key}' added successfully!"

    def update_edge(self, start_node: str, end_node: str, timestamp: str, weights: float) -> str:
        """
        添加/更新经验边（基于执行反馈）
        """
        if start_node not in self.tool_node:
            return f"Source tool '{start_node}' not found!"
        if end_node not in self.tool_node:
            return f"Target tool '{end_node}' not found!"

        edge_key = f"{start_node}->{end_node}"
        w = float(np.mean(weights)) if isinstance(weights, (list, tuple, np.ndarray)) else float(weights)

        if edge_key in self.tool_edge:
            e = self.tool_edge[edge_key]
            # 简单的累积/平滑策略
            old_w = self._edge_weights(e) or 0.0
            new_w = 0.5 * old_w + 0.5 * w
            if isinstance(e, ToolEdge):
                e.weights = new_w
                e.status = (e.status or 0) + 1
                e.timestamp = timestamp
            else:
                e["weights"] = new_w
                e["status"] = e.get("status", e.get("states", 0)) + 1
                e["timestamp"] = timestamp
        else:
            # 不存在则创建一条带经验权重的边
            new_edge = ToolEdge(start_node=start_node, end_node=end_node, messages=[], timestamp=timestamp, status=1)
            new_edge.weights = w
            self.tool_edge[edge_key] = new_edge

        self.save_edges_to_file()
        return f"Experience edge '{edge_key}' updated with weights {w}"

    def add_node_feedback(self, tool_name: str, feedback: str) -> str:
        """更新节点的反馈"""
        if tool_name not in self.tool_node:
            return f"Tool '{tool_name}' not found!"

        node = self.tool_node[tool_name]
        if isinstance(node, ToolNode):
            node.feedback = feedback
        else:
            node["feedback"] = feedback
        self.save_nodes_to_file()
        return f"Tool '{tool_name}' feedback updated to {feedback}"

    # ---------- 检索 ----------
    def get_similar_tools(
        self,
        query: str,
        top_k: int,
        use_graph: bool = True,
        seed_k: Optional[int] = None,
        depth: int = 2,
        graph_lambda: float = 0.3,
    ) -> List[tuple]:
        """
        检索与 query 最相关的工具。

        默认策略：优先深度检索（向量 + 图结构混合）：
        1）先用向量相似度全局检索出若干种子工具（seed_k）
        2）从这些种子出发，在工具图上做 BFS 扩展 depth 层，得到局部子图候选
        3）对子图中的工具，用：combined_score = cos_sim + graph_lambda * graph_bonus 重新打分
            - graph_bonus = 1 / (1 + hop_distance)，种子=1，1跳=0.5，2跳≈0.33
        4）按 combined_score 排序，返回前 top_k

        如果 use_graph=False，则退化为纯向量检索。
        返回: [(tool_name, score, node_dict_or_ToolNode), ...]
        """
        if not self.tool_node:
            return []

        # 纯向量模式
        if not use_graph:
            return self._get_similar_tools_flat(query, top_k)

        # 1) 先做一次全局向量检索，拿到种子
        base_scores = self._get_similar_tools_flat(query, top_k=None)  # 全量排序
        if not base_scores:
            return []

        if seed_k is None:
            seed_k = max(top_k, 5)

        seeds = [name for name, _, _ in base_scores[:seed_k]]

        # 2) 从种子出发，在图中 BFS 扩展 depth 层，得到候选及最短 hop 距离
        dist_map = self._graph_expand_from_seeds(seeds, depth)

        # 3) 重新打分：向量相似度 + 图结构 bonus
        candidates: List[Tuple[str, float, Any, float, Optional[int]]] = []

        # 先把 base_scores 转成 dict，方便查 base_sim
        base_score_dict = {name: (sim, node) for name, sim, node in base_scores}

        for name, (base_sim, node) in base_score_dict.items():
            if name not in dist_map:
                # 不在扩展子图中的工具可以忽略，优先子图
                continue
            hop_dist = dist_map[name]  # 0, 1, 2, ...
            graph_bonus = 1.0 / (1.0 + hop_dist)  # 种子=1, 1-hop=0.5, 2-hop≈0.33
            combined = base_sim + graph_lambda * graph_bonus
            candidates.append((name, combined, node, base_sim, hop_dist))

        # 如果图太稀疏，候选数量不足 top_k，则用剩余的纯向量结果补齐
        if len(candidates) < top_k:
            used = {c[0] for c in candidates}
            for name, base_sim, node in base_scores:
                if name in used:
                    continue
                candidates.append((name, base_sim, node, base_sim, None))
                if len(candidates) >= top_k:
                    break

        # 4) 最终排序并截断
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]

        # 对外只暴露 (tool_name, combined_score, node)
        return [(name, score, node) for name, score, node, _, _ in top]

    def _get_similar_tools_flat(self, query: str, top_k: Optional[int] = 5) -> List[tuple]:
        """
        纯向量相似度检索（原始实现提炼到这里）。
        当 top_k=None 时返回全量排序结果。
        """
        if not self.tool_node:
            return []

        query_vec = self._get_embedding(query)

        sims: List[Tuple[str, float, Any]] = []
        for tool_name, node in self.tool_node.items():
            if isinstance(node, ToolNode):
                vec = node.vector
            else:
                vec = node.get("vector")
            sim = self._cos_similarity(query_vec, vec)
            sims.append((tool_name, sim, node))

        sims.sort(key=lambda x: x[1], reverse=True)

        if top_k is None:
            return sims
        return sims[:top_k]

    def _graph_expand_from_seeds(
        self,
        seeds: List[str],
        depth: int = 2,
    ) -> Dict[str, int]:
        """
        从多个种子工具出发，在工具图上做 BFS 扩展，得到：
            {tool_name: 最短 hop 距离}

        depth: 最大 BFS 深度（0=只包含种子本身，1=再加一层邻居，以此类推）
        """
        dist_map: Dict[str, int] = {}

        for seed in seeds:
            if seed not in self.tool_node:
                continue

            # 如果这个种子已经有更短距离记录，就跳过
            if seed in dist_map and dist_map[seed] <= 0:
                continue

            q = deque()
            q.append((seed, 0))

            while q:
                node, d = q.popleft()
                # 如果已有更短路径，跳过
                if node in dist_map and dist_map[node] <= d:
                    continue
                dist_map[node] = d

                if d >= depth:
                    continue

                neighbors = self.get_connected_tools(node)
                for nb, _, _ in neighbors:
                    # 下一层
                    nd = d + 1
                    if nb not in dist_map or nd < dist_map[nb]:
                        q.append((nb, nd))

        return dist_map

    def get_connected_tools(self, tool_name: str) -> List[Tuple[str, str, Optional[float]]]:
        """
        获取与指定工具连接的工具
        返回：(对端工具名, 方向[outgoing/incoming], 权重)
        """
        if tool_name not in self.tool_node:
            return []

        connected = []
        for edge_key, e in self.tool_edge.items():
            try:
                start_node, end_node = edge_key.split("->", 1)
            except ValueError:
                continue
            w = self._edge_weights(e)
            if start_node == tool_name:
                connected.append((end_node, "outgoing", w))
            if end_node == tool_name:
                connected.append((start_node, "incoming", w))
        return connected

    def delete_node(self, tool_name: str) -> str:
        """删除节点及相关边"""
        if tool_name not in self.tool_node:
            return f"Tool '{tool_name}' not found!"

        del self.tool_node[tool_name]
        edges_to_delete = [ek for ek in list(self.tool_edge.keys()) if tool_name in ek]
        for ek in edges_to_delete:
            del self.tool_edge[ek]

        self.save_nodes_to_file()
        self.save_edges_to_file()
        return f"Tool '{tool_name}' and related edges deleted successfully!"

    # ---------- 向量工具 ----------
    @staticmethod
    def _cos_similarity(vec1: Optional[np.ndarray], vec2: Optional[np.ndarray]) -> float:
        if vec1 is None or vec2 is None:
            return 0.0
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.dot(v1, v2) / denom)

    @staticmethod
    def _get_embedding(text: str) -> np.ndarray:
        """使用本地 Ollama 获取向量；失败则回退为零向量"""
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        try:
            url = cfg["ollama"]["embedding_url"]
            model_name = cfg["ollama"]["model_name"]

            data = {"model": model_name, "prompt": text}
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding, dtype=float)
        except Exception as e:
            dim = cfg["ollama"].get("embedding_dim", 768)
            print(f"Error getting embedding: {e}")
            return np.zeros(dim, dtype=float)
