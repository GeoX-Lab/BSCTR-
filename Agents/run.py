import torch
import asyncio
from typing import List, Dict
from GraphManager import GraphManager
from Strategy.Dense import DenseRetriever
from Strategy.BM25 import BM25Retriever
from Strategy.SGCRetriever import SGCRetriever
from Strategy.Graph import GraphFusionRetriever
from Graph_RAG_Tool_Fusion import DataDrivenToolGraph
from Agents.agent import ToolAgent
from Working_memory import WorkingMemory
from collections import OrderedDict

class Wrapper(ToolAgent):
    def __init__(self, 
                 mode: str,
                 initial_model: str, 
                 output_dir: str, 
                 tool_dir: str,
                 trajectory_path: str = None,
                 config_path: str=None):
        
        super().__init__(initial_model, output_dir, tool_dir, yaml_path=config_path)
        self.mode = mode.lower()
        self.trajectory_path = trajectory_path
        self.yaml_path = config_path
        
        print(f"\n[***] Initializing Experiment Agent in Mode: [{self.mode.upper()}]")
        self.use_planning = self.mode in ['plan', 'plan_sgc']
        self.use_sgc_embedding = self.mode in ['sgc', 'plan_sgc']
        self.use_bm25 = self.mode == 'bm25'
        self.use_graph_fusion = self.mode == 'graph'

    def init_retrieval_system(self):
        """
        初始化不同的 Retriever
        """
        tools = self.tool_registry.tools
        self.tool_names = list(tools.keys())
        self.tool_map = {name: i for i, name in enumerate(self.tool_names)}
        
        descriptions = [
            self.tool_registry.get_unified_tool_info(name)["description"] 
            for name in self.tool_names
        ]

        if not self.use_bm25:
            print("[*] Encoding tool descriptions...")
            raw_embeds_list = [self.get_text_embedding(d) for d in descriptions]
            if raw_embeds_list:
                self.raw_embeddings = torch.cat(raw_embeds_list, dim=0)
            else:
                self.raw_embeddings = torch.empty(0, device=self.device)
        
        if self.use_sgc_embedding:
            print("[*] Initializing SGC Graph...")
            num_nodes = len(self.tool_names)
            self.graph_manager = GraphManager(num_nodes=num_nodes, device=self.device)
            
            if self.trajectory_path:
                trajectories = self.load_trajectory_from_file(self.trajectory_path, 248)
                for traj in trajectories:
                    indices = [self.tool_map[t] for t in traj if t in self.tool_map]
                    if len(indices) >= 2:
                        self.graph_manager.update_from_trajectory(indices)
            print("[*] SGC Embeddings Computed.")

        if self.use_bm25:
            self.retriever = BM25Retriever(self.tool_names, descriptions)
            print("[*] Retriever: BM25 (Keyword)")
            
        elif self.use_sgc_embedding:
            self.retriever = SGCRetriever(
                graph_manager=self.graph_manager, 
                tool_names=self.tool_names,
                raw_embeddings=self.raw_embeddings,
                alpha=0.5,
                mode='f')
            print(f"[*] Retriever: SGC")
        
        elif self.use_graph_fusion:
            print("[*] Initializing Graph Fusion Retriever (Dense + Trajectory Graph)...")
            tool_graph = DataDrivenToolGraph()

            tool_graph.init_nodes_from_registry(self.tool_registry)
            
            if self.trajectory_path:
                print(f"    -> Loading trajectories from: {self.trajectory_path}")
                tool_graph.build_edges_from_json(self.trajectory_path, direction="backward")
            else:
                print("    [!] Warning: No trajectory_path provided for Graph Fusion! Graph will be empty.")

            base_dense = DenseRetriever(self.tool_names, self.raw_embeddings)
            
            self.retriever = GraphFusionRetriever(
                dense_retriever=base_dense,
                tool_graph=tool_graph,
                tool_names=self.tool_names,
                d_limit=3,
                final_top_k=10  
            )
            print(f"[*] Retriever: Graph RAG-Tool Fusion (Backward DFS)")
            
        else: 
            self.retriever = DenseRetriever(self.tool_names, self.raw_embeddings)
            print(f"[*] Retriever: Dense (Raw BERT/Ada)")

    def build_tool_pool(self, tasks: List[Dict], top_k: int = 5, rrf_k: int = 60):

        print(f"\n>>> [Retrieval ({self.mode})] Processing {len(tasks)} new tasks (Incremental Update)...")

        tool_rrf_scores = {}
        tool_metadata = {}

        if self.tool_set:
            for name, data in self.tool_set.items():
                tool_rrf_scores[name] = data.get('score', 0.0)
                tool_metadata[name] = {
                    "id": data.get("id"),
                    "name": name
                }
            print(f"    [Merge] Loaded {len(self.tool_set)} existing tools into candidate pool.")

        for task in tasks:
            if self.use_planning:
                query_text = f"{task.get('action', '')} {task.get('tool_search', '')}"
            else:
                query_text = task.get('query', '')

            candidates = []
            if self.use_bm25:
                candidates = self.retriever.search(query_text, top_k=top_k)
            else:
                query_vec = self.get_text_embedding(query_text)
                candidates = self.retriever.search(query_vec, top_k=top_k)

            if not candidates:
                print(f"[Warning] No tools found for: {query_text[:20]}...")
                continue

            for rank, c in enumerate(candidates):
                tool_name = c["name"]
                rrf_score = 1.0 / (rrf_k + rank + 1)

                if tool_name in tool_rrf_scores:
                    tool_rrf_scores[tool_name] += rrf_score
                else:
                    tool_rrf_scores[tool_name] = rrf_score
                    tool_metadata[tool_name] = c

        sorted_tools = sorted(tool_rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        self.tool_set = OrderedDict() 

        for tool_name, rrf_score in sorted_tools:
            if tool_name in tool_metadata:
                original_info = tool_metadata[tool_name]
                
                self.tool_set[tool_name] = {
                    "id": original_info["id"],
                    "name": tool_name,
                    "score": rrf_score,
                }

        print(f"[Retrieval Done] Final Pool Size: {len(self.tool_set)} (Merged & Sorted)")

    async def run(self, user_query: str, choices=None, trajectory_path=None):
        self.history = []
        self.tool_set = OrderedDict()
        self.attempt_tool_chain = []
        
        self.working_memory = WorkingMemory(original_query=user_query)
        self.working_memory.recent_steps = []
        if trajectory_path:
            self.trajectory_path = trajectory_path

        if self.retriever is None:
            self.init_retrieval_system()
            
        if not self.use_bm25:
             self._assert_tool_index_alignment()

        print(f"\n====== Running Experiment Mode: {self.mode.upper()} ======")
        print(f"User Query: {user_query}")

        try:
            task_queue = []
            self.history.append({"role": "user", "content": user_query})
            
            if self.use_planning:
                print(f"\n>>> [Planning] Decomposing...")
                task_queue = await self._decompose_query(user_query)
                print(f"    Initial Plan: {len(task_queue)} steps.")
            else:
                print(f"\n>>> [No-Planning] Using raw query directly.")
                task_queue = [{
                    "step": 1,
                    "action": "execute",
                    "query": user_query,        
                    "tool_search": user_query   
                }]

            self.build_tool_pool(task_queue, top_k=5) 

            # TODO
            # API-Bank need final summary
            # final_res = await self._execute_tasks_logic(task_queue, user_query, choices)
            final_res = "Task Finished!"
            
            return final_res

        except asyncio.CancelledError:
            print("\n[!] Agent execution was cancelled due to timeout!")
            return "TIMEOUT"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {e}"
        finally:
             self._save_tool()

    async def _execute_tasks_logic(self, task_queue, user_query, choices):

        task_idx = 0
        global_retry_count = 0
        MAX_GLOBAL_RETRIES = 3
        MAX_LOCAL_RETRIES = 14

        final_summary = "Task Failed."

        while task_idx < len(task_queue):
            current_task = task_queue[task_idx]
            self.working_memory.start_task(current_task['query'])

            print(f"\n=== Step {task_idx + 1}: {current_task['query']} ===")

            subtask_success = False
            local_attempt = 0
            
            tools_ids = []

            while local_attempt <= MAX_LOCAL_RETRIES:
                local_attempt += 1
                print(f"   [Attempt {local_attempt}/{MAX_LOCAL_RETRIES + 1}] Processing...")

                response = await self._tool_call(current_task)
                
                raw_status = response.get("status")
                final_status = None

                if isinstance(raw_status, dict):
                    final_status = raw_status.get("status")
                elif isinstance(raw_status, str):
                    final_status = raw_status

                if final_status == "SUCCESS":
                    reason = response.get("reason", "No reason provided") 
                    print(f"   [Decision] Subtask Success: {reason}")
                    subtask_success = True
                    if tools_ids:
                        self.attempt_tool_chain.extend(tools_ids)
                    break

                elif final_status == "FAILURE":
                    reason = response.get("reason", "Unknown failure")
                    print(f"   [Decision] Subtask Failed: {reason}")
                    self.working_memory.add_feedback_message(f"Subtask Failed: {reason}")
                    break 
                
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    print(f"   [Decision] Calling {len(tool_calls)} tools...")

                    for call in tool_calls:
                        tool_name = call.get('tool_name')
                        args = call.get('arguments', {})
                        
                        try:
                            print(f"       -> Executing: {tool_name}")
                            is_success, result = await self.call_tool(tool_name, args)
                            
                            outcome = "SUCCESS" if is_success else "FAIL"
                            if is_success and tool_name in self.tool_map:
                                tools_ids.append(self.tool_map[tool_name])

                            if not is_success:
                                print(f"       -> Tool execution failed: {result}")

                        except Exception as e:
                            result = f"Error: {str(e)}"
                            outcome = "FAIL"
                            print(f"       -> Error: {result}")

                        self.working_memory.record_step(tool=tool_name, args=args, result=result, outcome=outcome)
                    continue
                else:
                    print(f"   [Warning] Empty response. Retrying...")
                    self.working_memory.record_step(tool="No tool calls", args={}, result="Error: Empty response.", outcome="FAIL")
                    continue
                
            if subtask_success:
                task_idx += 1
                self.working_memory.finished_tasks.append(current_task['query'])
            else:
                print(f"[!] Step {task_idx + 1} failed after {local_attempt} attempts.")
                
                if self.use_planning and global_retry_count < MAX_GLOBAL_RETRIES:
                    print("    >>> Triggering Re-plan...")
                    new_sub_tasks = await self._re_plan()

                    if new_sub_tasks:
                        print(f"    [Re-plan Success] Generated {len(new_sub_tasks)} new steps.")

                        task_queue = task_queue[:task_idx] + new_sub_tasks

                        self.build_tool_pool(new_sub_tasks)
                        
                        global_retry_count += 1
                        continue
                    else:
                        print("[!] Re-planning returned empty tasks. Aborting.")
                        break
                else:
                    reason = "Max global retries reached" if self.use_planning else "Planning disabled in this mode"
                    print(f"[!] Task Failed. Reason: {reason}.")
                    break

        if self.use_sgc_embedding and self.graph_manager is not None:
            if len(self.attempt_tool_chain) > 1:
                print(f"\n>>> [Graph Update] Batch updating tool graph...")
                update_count = 0
                for i in range(len(self.attempt_tool_chain) - 1):
                    src = self.attempt_tool_chain[i]
                    dst = self.attempt_tool_chain[i + 1]
                    if src != dst:
                        self.graph_manager.add_edge(src, dst, weight=1.0)
                        update_count += 1
                print(f"     [Graph Update] Added {update_count} edges.")
            else:
                print("\n>>> [Graph Update] Chain too short.")
        
        # TODO
        # API-Bank need final summary
        # final_summary = await self._final_summary(user_query, choices)
        final_summary = "Task finished"
        return final_summary
    