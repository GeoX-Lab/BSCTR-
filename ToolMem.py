from Base import ToolNode, ToolEdge
from typing import Dict, List
import numpy as np
import json
import requests
import yaml

class ToolMem:

    """
    创建一个 Agent的 Tool-memory，该Tool-memory可以被Tool-manager调用
    memory以图的形式构建，图中的节点为工具，边具有两个主要的参数，一是逻辑参数，二是经验参数。
        逻辑参数：工具与工具之间的逻辑联系。如果 Tool1的输出是 Tool2的输入，则 Tool1 -> Tool2。
        经验参数：一系列 Tool被调用后，memory会记录被调用的 Tool链与任务执行结果的反馈。
    memory的检索也分别分为逻辑边的检索与经验边的检索。
    """

    def __init__(self, tool_node_path=None, tool_edge_path=None):
        self.tool_node: Dict[str, ToolNode] = {}
        self.tool_edge: Dict[str, ToolEdge] = {}
        self.tool_node_path = tool_node_path
        self.tool_edge_path = tool_edge_path

    def save_nodes_to_file(self) -> str:
        """保存工具节点到文件"""
        serializable_nodes = {}
        for tool_name, tool_node in self.tool_node.items():
            serializable_nodes[tool_name] = {
                'name': tool_name,
                'description': tool_node.description,
                'inputs': tool_node.inputs,
                'outputs': tool_node.outputs,
                'feedback': tool_node.feedback,
                'vector': tool_node.vector
            }

        with open(self.tool_node_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_nodes, f, ensure_ascii=False, indent=2)

        return f"Tool nodes saved to {self.tool_node_path}"
    def save_edges_to_file(self) -> str:
        """保存工具边到文件"""
        serializable_edges = {}
        for edge_key, tool_edge in self.tool_edge.items():
            serializable_edges[edge_key] = {
                "start_node": tool_edge.start_node,
                "end_node": tool_edge.end_node,
                "messages": tool_edge.messages,
                "states": tool_edge.status,
                "timestamp": tool_edge.timestamp,
                "weights": tool_edge.weights
            }

        with open(self.tool_edge_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_edges, f, ensure_ascii=False, indent=2)

        return f"Tool edges saved to {self.tool_edge_path}"

    def get_node_from_doc(self):
        """
        从文件中读取 tool_node信息
        """
        with open(self.tool_node_path, 'r', encoding='utf-8') as f:
            self.tool_node = json.load(f)
        return self.tool_node

    def get_edge_from_doc(self):
        """
        从文件中读取 tool_edge信息
        """
        with open(self.tool_edge_path, 'r', encoding='utf-8') as f:
            self.tool_edge = json.load(f)
            return self.tool_edge

    def add_node(self, new_node):
        """
        添加 tool节点到 tool_list
        """
        node_dict = {
            'name': new_node.name,
            'description': new_node.description,
            'inputs': new_node.inputs,
            'outputs': new_node.outputs,
            'feedback': new_node.feedback,
            'vector': new_node.vector.tolist() if isinstance(new_node.vector, np.ndarray) else new_node.vector
        }

        with open(self.tool_node_path, 'r', encoding='utf-8') as f:
            tool_node = json.load(f)
        i = len(tool_node)
        tool_node[str(i+1)] = node_dict

        with open(self.tool_node_path, 'w', encoding='utf-8') as f:
            json.dump(tool_node, f, ensure_ascii=False, indent=2)

        return f"New Tool Node is saved to {self.tool_node_path}!"

    def add_edge(self, start_node: str, end_node: str, messages: List[str], timestamp=None) -> str:
        """
        添加逻辑边（由LLM生成）
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

        # 保存到文件
        self.save_edges_to_file()

        return f"Logical edge '{edge_key}' added successfully!"

    def update_edge(self, start_node: str, end_node: str, timestamp: str, weights: float) -> str:
        """
        添加经验边（基于执行反馈）
        """
        if start_node not in self.tool_node:
            return f"Source tool '{start_node}' not found!"
        if end_node not in self.tool_node:
            return f"Target tool '{end_node}' not found!"

        edge_key = f"{start_node}->{end_node}"

        if edge_key in self.tool_edge:
            # 更新现有边
            edge = self.tool_edge[edge_key]
            edge.weights = np.mean(weights) if edge.weights else 0.01
            edge.status += 1
            edge.timestamp = timestamp

        # 保存到文件
        self.save_edges_to_file()

        return f"Experience edge '{edge_key}' updated with weights {weights}"

    def add_node_feedback(self, tool_name: str, feedback: str) -> str:
        """更新节点的反馈"""
        if tool_name not in self.tool_node:
            return f"Tool '{tool_name}' not found!"

        self.tool_node[tool_name].feedback = feedback
        self.save_nodes_to_file()

        return f"Tool '{tool_name}' feedback updated to {feedback}"

    def get_similar_tools(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        基于向量相似度检索相似工具
        """
        if not self.tool_node:
            return []

        # 生成查询向量（简化版）
        query_vec = self._get_embedding(query)  # 实际应该使用相同的嵌入模型

        similarities = []
        for tool_name, node in self.tool_node.items():
            similarity = self._cos_similarity(query_vec, node.vector)
            similarities.append((tool_name, similarity, node))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_subgraph(self, tool_name: str, depth: int = 2) -> Dict[str, List[str]]:
        """
        根据指定工具及其深度，检索相关的子图。
        depth 控制递归的深度，表示从目标工具开始，向外扩展的层级。

        返回一个字典，其中包括该工具及其相关工具和边的信息。
        """
        # 初始化相关子图
        subgraph = {tool_name: {'connected_tools': [], 'edges': []}}
        visited = set()  # 防止重复访问
        to_visit = [tool_name]  # 待访问的工具列表

        # 执行广度优先搜索，直到达到指定深度
        current_depth = 0
        while to_visit and current_depth < depth:
            next_to_visit = []
            for tool in to_visit:
                if tool not in visited:
                    visited.add(tool)
                    # 获取当前工具与其他工具的连接关系
                    connected_tools = self.get_connected_tools(tool)
                    subgraph[tool]['connected_tools'] = connected_tools
                    # 将当前工具的边信息添加到子图中
                    for connected_tool, direction, _ in connected_tools:
                        edge_key = f"{tool}->{connected_tool}"
                        if edge_key in self.tool_edge:
                            subgraph[tool]['edges'].append(self.tool_edge[edge_key])

                        next_to_visit.append(connected_tool)
            to_visit = next_to_visit
            current_depth += 1

        return subgraph

    def get_connected_tools(self, tool_name: str):
        """
        获取与指定工具连接的工具
        """
        if tool_name not in self.tool_node:
            return []

        connected = []
        for edge_key, edge in self.tool_edge.items():
            start_node, end_node = edge_key.split("->")
            if start_node == tool_name:
                connected.append((end_node, "outgoing", edge.weights))
            elif edge == tool_name:
                connected.append((start_node, "incoming", edge.weights))

        return connected

    def delete_node(self, tool_name: str) -> str:
        """删除节点及相关边"""
        if tool_name not in self.tool_node:
            return f"Tool '{tool_name}' not found!"

        # 删除节点
        del self.tool_node[tool_name]

        # 删除相关边
        edges_to_delete = []
        for edge_key in self.tool_edge.keys():
            if tool_name in edge_key:
                edges_to_delete.append(edge_key)

        for edge_key in edges_to_delete:
            del self.tool_edge[edge_key]

        # 保存更改
        self.save_nodes_to_file()
        self.save_edges_to_file()

        return f"Tool '{tool_name}' and related edges deleted successfully!"

    @staticmethod
    def _cos_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1 is None or vec2 is None:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

    @staticmethod
    def _get_embedding(text):
        """
        使用本地 Ollama模型获取文本的向量表示
        """
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        try:
            # Ollama的API端点，假设运行在本地默认端口
            url = cfg["ollama"]["embedding_url"]
            model_name = cfg["ollama"]["model_name"]

            data = {
                "model": model_name,
                "prompt": text
            }
            response = requests.post(url, json=data)
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding)

        except Exception as e:
            dim = cfg["ollama"]["embedding_dim"]
            print(f"Error getting embedding: {e}")
            return np.zeros(dim)
