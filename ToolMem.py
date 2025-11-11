import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import requests
import yaml
from Base import ToolNode, ToolEdge

# -------------------------------
# ToolMem：统一用“工具名”为 key 的节点表；修复 connected_tools；兼容 dict/对象
# -------------------------------
class ToolMem:
    """
    Agent 的 Tool-memory（图：节点=工具；边=逻辑/经验）
    """

    def __init__(self, tool_node_path: Optional[str] = None, tool_edge_path: Optional[str] = None):
        self.tool_node: Dict[str, Any] = {}   # 统一：key=工具名，value=dict 或 ToolNode
        self.tool_edge: Dict[str, Any] = {}   # key="A->B"，value=ToolEdge 或 dict
        self.tool_node_path = tool_node_path
        self.tool_edge_path = tool_edge_path

    # ---------- 基础 IO ----------
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
    def get_similar_tools(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        基于向量相似度检索相似工具（兼容 ToolNode/dict）
        """
        if not self.tool_node:
            return []

        query_vec = self._get_embedding(query)

        sims = []
        for tool_name, node in self.tool_node.items():
            if isinstance(node, ToolNode):
                vec = node.vector
            else:
                vec = node.get("vector")
            sim = self._cos_similarity(query_vec, vec)
            sims.append((tool_name, sim, node))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def get_subgraph(self, tool_name: str, depth: int = 2) -> Dict[str, List[str]]:
        """
        BFS 子图
        """
        subgraph = {tool_name: {"connected_tools": [], "edges": []}}
        visited = set()
        to_visit = [tool_name]
        current_depth = 0

        while to_visit and current_depth < depth:
            nxt = []
            for tool in to_visit:
                if tool in visited:
                    continue
                visited.add(tool)
                connected_tools = self.get_connected_tools(tool)
                subgraph.setdefault(tool, {"connected_tools": [], "edges": []})
                subgraph[tool]["connected_tools"] = connected_tools

                for connected_tool, direction, w in connected_tools:
                    edge_key = f"{tool}->{connected_tool}" if direction == "outgoing" else f"{connected_tool}->{tool}"
                    if edge_key in self.tool_edge:
                        subgraph[tool]["edges"].append(self.tool_edge[edge_key])
                    nxt.append(connected_tool)
            to_visit = nxt
            current_depth += 1

        return subgraph

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
