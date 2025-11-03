from Base import ToolEdge
from typing import Dict
import json,re

class GTALogicEdge:
    def __init__(self, logic_edge_txt_path="./gta_dataset/gta_graph.txt", logic_edge_save_path="./gta_dataset/gta_logic_edge.json"):
        self.logic_edge_txt_path = logic_edge_txt_path
        self.logic_edge_save_path = logic_edge_save_path

    def _get_edge_key(self, start_node: str, end_node: str) -> str:
        """生成唯一的边键"""
        return f"{start_node}->{end_node}"

    def build_log_edges_from_file(self) -> Dict[str, ToolEdge]:
        """
        从txt文件读取逻辑边并构建self.tool_log_edge
        """
        self.tool_log_edge = {}
        try:
            with open(self.logic_edge_txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释
                    continue

                # 解析逻辑边
                edge = self._parse_logic_edge(line, line_num)
                if edge:
                    edge_key = self._get_edge_key(edge.start_node, edge.end_node)
                    self.tool_log_edge[edge_key] = edge

            print(f"成功从 {self.logic_edge_txt_path} 加载 {len(self.tool_log_edge)} 条逻辑边")
            print(self.tool_log_edge)
            return self.tool_log_edge

        except FileNotFoundError:
            print(f"错误：找不到文件 {self.logic_edge_txt_path}")
            return {}
        except Exception as e:
            print(f"读取文件时出错：{e}")
            return {}

    def _parse_logic_edge(self, line: str, line_num: int):
        """
        解析单行逻辑边描述
        """
        # 使用正则表达式匹配格式：数字 -> 数字：描述
        pattern = r'(\d+)\s*->\s*(\d+)\s*：\s*(.+)'
        match = re.match(pattern, line)

        if not match:
            print(f"第{line_num}行格式错误: {line}")
            return None

        start_node = match.group(1)
        end_node = match.group(2)
        description = match.group(3).strip()

        # 创建LogicEdge对象
        edge = ToolEdge(
            start_node=start_node,
            end_node=end_node,
            messages=[description],
            weights=float(0)
        )
        return edge
    def add_log_edge(self, start_node: str, end_node: str, message: str):
        """添加单个逻辑边"""
        edge_key = self._get_edge_key(start_node, end_node)

        if edge_key in self.tool_log_edge:
            # 如果边已存在，添加新的消息
            existing_edge = self.tool_log_edge[edge_key]
            if message not in existing_edge.messages:
                existing_edge.messages.append(message)
        else:
            # 创建新的逻辑边
            new_edge = ToolEdge(
                start_node=start_node,
                end_node=end_node,
                messages=[message],
                weights=float(0)
            )
            self.tool_log_edge[edge_key] = new_edge

    def save_log_edges(self):
        """保存逻辑边到JSON文件"""
        try:
            edges_dict = {}
            for edge_key, edge in self.tool_log_edge.items():
                edges = {
                    "start_node": edge.start_node,
                    "end_node": edge.end_node,
                    "messages": edge.messages,
                    "weights": edge.weights
                }
                edges_dict[edge_key] = edges

            with open(self.logic_edge_save_path, 'w', encoding='utf-8') as f:
                json.dump(edges_dict, f, ensure_ascii=False, indent=2)

            print(f"逻辑边已保存到 {self.logic_edge_save_path}")
            return True

        except Exception as e:
            print(f"保存逻辑边时出错：{e}")
            return False

    def load_log_edges(self):
        """从JSON文件加载逻辑边"""
        try:
            with open(self.logic_edge_save_path, 'r', encoding='utf-8') as f:
                edges_dict = json.load(f)

            self.tool_log_edge = {}
            for edge_key, edge_data in edges_dict.items():
                self.tool_log_edge[edge_key] = ToolEdge(**edge_data)

            print(f"从 {self.logic_edge_save_path} 加载了 {len(self.tool_log_edge)} 条逻辑边")
            return self.tool_log_edge

        except FileNotFoundError:
            print(f"文件不存在：{self.logic_edge_save_path}")
            return {}
        except Exception as e:
            print(f"加载逻辑边时出错：{e}")
            return {}