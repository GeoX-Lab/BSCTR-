import json
import networkx as nx
from Toolregistry import ToolRegistry
from typing import List

class DataDrivenToolGraph:

    def __init__(self):
        self.graph = nx.DiGraph()

    def init_nodes_from_registry(self, registry: ToolRegistry):

        for tool_name in registry.tools.keys():
            self.graph.add_node(tool_name)
        print(f"[*] Initialized {len(registry.tools)} nodes from ToolRegistry.")

    def build_edges_from_json(self, json_file_path: str, direction: str = "forward"):

        with open(json_file_path, 'r', encoding='utf-8') as f:
            trajectories_dict = json.load(f)
            
        edge_count = 0
        
        for traj_id, trajectory in trajectories_dict.items():
            if not trajectory or len(trajectory) < 2:
                continue

            for i in range(len(trajectory) - 1):
                tool_current = trajectory[i]
                tool_next = trajectory[i+1]
                
                if tool_current not in self.graph:
                    self.graph.add_node(tool_current)
                if tool_next not in self.graph:
                    self.graph.add_node(tool_next)
  
                if direction == "forward":

                    if not self.graph.has_edge(tool_current, tool_next):
                        self.graph.add_edge(tool_current, tool_next, weight=1)
                        edge_count += 1
                        
                elif direction == "backward":

                    if not self.graph.has_edge(tool_next, tool_current):
                        self.graph.add_edge(tool_next, tool_current, weight=1)
                        edge_count += 1
                else:
                    raise ValueError("direction 必须是 'forward' 或 'backward'")
                    
        print(f"[*] Unidirectional Graph ({direction}) complete! Added {edge_count} directed edges.")

    def dfs_search(self, start_tool: str, d_limit: int) -> List[str]:

        if start_tool not in self.graph:
            return []
        
        dependencies =[]

        dfs_nodes = nx.dfs_preorder_nodes(self.graph, source=start_tool)
        
        for node in dfs_nodes:
            if node == start_tool:
                continue
            
            dependencies.append(node)
            if len(dependencies) >= d_limit:
                break
                
        return dependencies