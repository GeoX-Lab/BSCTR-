from Base import ToolNode, ToolEdge
from typing import Dict
import numpy as np
from gta_dataset.extract_tool_from_GTA import ExtractTool
from gta_dataset.gta_logic_edge import GTALogicEdge

import json
class ToolMem:

    """
    创建一个 Agent的 Tool-memory
    memory以图的形式构建，图中的节点为工具，边分为两种：其一是逻辑边，其二是经验边。
        逻辑边：工具与工具之间的逻辑联系。如果 Tool1的输出是 Tool2的输入，则 Tool1 -> Tool2。
        经验边：一系列 Tool被调用后，memory会记录被调用的 Tool链与任务执行结果的反馈。
    memory的检索也分别分为逻辑边的检索与经验边的检索。
    """

    def __init__(self, tool_node_path=None, tool_edge_path=None):
        self.tool_node: Dict[str, ToolNode] = {}
        self.tool_exp_edge: Dict[str, ToolEdge] = {}
        self.tool_node_path = tool_node_path
        self.tool_edge_path = tool_edge_path
        self.gta_tool_edge = GTALogicEdge()

    def read_gta_tool_doc(self):
        """
        提取 Tool的描述信息
        """
        extract_tool = ExtractTool()
        self.tool_node = extract_tool.extract_tool_from_gta()
        serializable_nodes = {}

        for tool_id, tool_node in self.tool_node.items():
            node_dict = {
                'name': tool_node.name,
                'description': tool_node.description,
                'inputs': tool_node.inputs,
                'outputs': tool_node.outputs,
                'state': tool_node.state,
                'feedback': tool_node.feedback,
                'vector': tool_node.vector.tolist() if isinstance(tool_node.vector, np.ndarray) else tool_node.vector
            }
            serializable_nodes[tool_id] = node_dict

        # 写入JSON文件
        with open(self.tool_node_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_nodes, f, ensure_ascii=False, indent=2)

        return f"Tool nodes saved to {self.tool_node_path}"

    def get_node_from_doc(self):
        """
        从文件中读取 tool_node信息
        """
        with open(self.tool_node_path, 'r', encoding='utf-8') as f:
            self.tool_node = json.load(f)
        return self.tool_node

    def add_node(self, new_node: ToolNode):
        """
        添加 tool节点到 tool_list
        """
        node_dict = {
            'name': new_node.name,
            'description': new_node.description,
            'inputs': new_node.inputs,
            'outputs': new_node.outputs,
            'state': new_node.state,
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

    def add_edge(self):
        """
        添加 tool边
        大模型完成
        """
        self.gta_tool_edge.build_log_edges_from_file()
        self.gta_tool_edge.save_log_edges()

    def get_edge_from_doc(self):
        return self.gta_tool_edge.load_log_edges()

    def expend_retrieve(self):
        """
        检索图拓展
        """

        pass

    def delete_node(self):
        """
        删除节点
        """
        pass

    def delete_log_edge(self):
        """
        删除逻辑边
        """
        pass

    def delete_exp_edge(self):
        """
        删除经验边
        """
        pass

    def init_graph(self):
        """
        初始化图
        """
        pass

    def _update_weights(self):
        pass

    def _update_feedback(self):
        pass

    def _cos_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1 is None or vec2 is None:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

    def _get_embedding(self):
        pass

if __name__ == '__main__':
    tool_mem = ToolMem("gta_dataset/gta_tool_node.json")
    tool_mem.add_edge()
    tool_mem.get_edge_from_doc()