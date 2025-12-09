import json
import requests
import yaml
import torch
from typing import Any, Optional, List, Dict
import inspect
from model import LLM
from Toolregistry import ToolRegistry
from GraphManager import GraphManager
from SGCRetriever import SGCRetriever


class BaseAgent:
    def __init__(self, initial_model: str, sys_prompt_template: str, output_dir: str = "./outputs/outputs.json"):

        self.initial_model = initial_model
        self.sys_prompt_template = sys_prompt_template
        self.output_dir = output_dir
        self.llm = LLM(initial_model)

        self.tool_registry = ToolRegistry()
        self.tool_list = []
        self.history = []

    def _normalize_chunk(self, chunk: Any) -> str:
        if isinstance(chunk, dict):
            if chunk.get("type") == "error":
                raise RuntimeError(chunk.get("error", "unknown tool error"))
            return chunk.get("text", json.dumps(chunk, ensure_ascii=False))
        if isinstance(chunk, (bytes, bytearray)):
            return chunk.decode("utf-8", errors="ignore")
        return str(chunk)

    async def call_tool(self, tool_name: str, arguments: Optional[dict] = None):
        arguments = arguments or {}
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return f"Tool '{tool_name}' not found."

        fn = tool["callable"]

        try:
            out = fn(**arguments)
            if inspect.isawaitable(out):
                out = await out

            if hasattr(out, "__aiter__"):
                acc = ""
                async for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc

            elif hasattr(out, "__iter__") and not isinstance(out, (str, bytes, dict)):
                acc = ""
                for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc

            else:
                text = self._normalize_chunk(out)

            # 记录工具调用轨迹
            self.tool_list.append(tool_name)

            # 写入历史
            self.history.append({
                "role": "tool",
                "name": tool_name,
                "content": text
            })

            return text

        except Exception as e:
            return f"Exception calling tool {tool_name}: {e}"

    async def chat(self, prompt, llm_name=None, image_path=None):

        if llm_name:
            self.llm = LLM(llm_name)

        # 写入用户消息（保持最简单历史）
        self.history.append({"role": "user", "content": prompt})

        # 流式输出
        result = None
        acc = []

        async for chunk in self.llm.generate_stream_res(
            prompt=prompt,
            history=self.history,
            image_path=image_path
        ):
            if chunk.get("type") == "error":
                return f"Error: {chunk.get('error')}"

            if chunk.get("type") in ("text", "stream"):
                acc.append(chunk.get("text", ""))

            if chunk.get("type") == "final":
                result = chunk.get("text", "")

        result = result if result is not None else "".join(acc)

        self.history.append({"role": "assistant", "content": result})

        return result

class SGCAgent(BaseAgent):
    def __init__(self,
                 initial_model: str,
                 sys_prompt_template: str,
                 output_dir: str = "./outputs/outputs.json",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        # 调用基类初始化
        super().__init__(initial_model, sys_prompt_template, output_dir)

        self.device = device
        print(f"[*] SGCAgent initialized on device: {self.device}")

        # 2. SGC 系统组件占位
        self.tool_names = []
        self.tool_map = {}  # name -> id
        self.graph_manager = None
        self.retriever = None
        self.raw_embeddings = None

        # 3. 轨迹缓冲区 (用于记录当前会话的任务链: [id1, id2, ...])
        self.trajectory_buffer = []

    def get_text_embedding(self, text: str) -> torch.Tensor:

        with open("../config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        try:
            url = cfg["ollama"]["embedding_url"]
            model_name = cfg["ollama"]["model_name"]
            data = {"model": model_name, "prompt": text}

            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()

            embedding_list = response.json()["embedding"]
            tensor = torch.tensor(embedding_list, dtype=torch.float32)
            tensor = tensor.unsqueeze(0)
            return tensor.to(self.device)

        except Exception as e:
            dim = cfg["ollama"].get("embedding_dim", 768)
            print(f"Error getting embedding: {e}")
            return torch.zeros((1, dim), dtype=torch.float32, device=self.device)
    def init_sgc_system(self):
        """
        初始化图检索系统。
        注意：必须在 ToolRegistry 注册完所有工具后调用此方法。
        """
        tools = self.tool_registry.tools  # 获取字典 {name: tool_conf}
        if not tools:
            print("[!] No tools registered in ToolRegistry.")
            return

        self.tool_names = list(tools.keys())
        # 提取工具描述用于初始化向量
        tool_descs = [t.get('description', '') or t.get('name') for t in tools.values()]
        self.tool_map = {name: i for i, name in enumerate(self.tool_names)}

        num_nodes = len(self.tool_names)

        print(f"[*] Initializing SGC Graph for {num_nodes} tools...")

        # 1. 初始化图管理器 (传入 device，矩阵将创建在 GPU)
        # 初始状态：离散节点，无边
        self.graph_manager = GraphManager(num_nodes=num_nodes, device=self.device)

        # 2. 批量生成初始工具嵌入 (Raw Embeddings)
        raw_embeds_list = []
        for desc in tool_descs:
            # get_text_embedding 已经处理了 device
            raw_embeds_list.append(self.get_text_embedding(desc))

        if raw_embeds_list:
            # torch.cat 在 GPU 上执行拼接
            self.raw_embeddings = torch.cat(raw_embeds_list, dim=0)  # (N, D)
        else:
            self.raw_embeddings = torch.empty(0, device=self.device)

        # 3. 初始化 SGC 检索器
        # 计算将利用 GPU 加速
        self.retriever = SGCRetriever(
            graph_manager=self.graph_manager,
            tool_names=self.tool_names,
            raw_embeddings=self.raw_embeddings,
            alpha=0.2  # 调节 SGC 聚合强度
        )
        print("[*] SGC System ready.")

    async def _llm_generate_text(self, prompt: str) -> str:
        """辅助方法：非流式获取 LLM 完整响应"""
        acc = []
        async for chunk in self.llm.generate_stream_res(prompt=prompt, history=self.history):
            if chunk.get("type") in ("text", "stream", "final"):
                acc.append(chunk.get("text", ""))
        return "".join(acc).strip()

    async def decompose_query(self, query: str) -> List[Dict]:
        """
        Step 1: 任务分解
        将用户查询分解为带有明确 'action' 的子问题序列。
        """
        prompt = f"""
        You are a generic planning agent. Break down the User Query into a sequence of sub-tasks.
        Each sub-task MUST have a distinct 'action' (verb) and a specific 'query'.

        User Query: "{query}"

        Output format: A pure JSON list of objects.
        Example:
        [
            {{"step": 1, "action": "search", "query": "find stock price of Apple"}},
            {{"step": 2, "action": "calculate", "query": "calculate the PE ratio"}}
        ]
        Do not output markdown code blocks, just the JSON string.
        """
        resp = await self._llm_generate_text(prompt)

        # 简单的 JSON 清洗逻辑
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0]
        elif "```" in resp:
            resp = resp.split("```")[1].split("```")[0]

        try:
            tasks = json.loads(resp)
            return tasks if isinstance(tasks, list) else []
        except json.JSONDecodeError:
            print(f"[!] Decomposition JSON error. Raw: {resp}")
            # 降级策略：作为单步任务
            return [{"step": 1, "action": "execute", "query": query}]

    async def generate_tool_args(self, tool_name: str, task_query: str, context: str) -> Dict:
        """根据上下文生成工具参数"""
        tool_info = self.tool_registry.get_tool(tool_name)
        schema = tool_info.get("args", {})

        prompt = f"""
        Construct arguments for the tool '{tool_name}'.
        Task: {task_query}
        Tool Schema: {json.dumps(schema)}
        Preceding Execution Context: {context}

        Return ONLY a JSON object with the arguments.
        """
        resp = await self._llm_generate_text(prompt)
        try:
            if "```" in resp: resp = resp.split("```")[1].replace("json", "")
            return json.loads(resp)
        except:
            return {}

    async def reflection(self, task_query: str, tool_name: str, tool_result: str) -> bool:
        """Step 3: 反思机制，判断任务是否完成"""
        prompt = f"""
        Goal: {task_query}
        Tool Used: {tool_name}
        Result: {str(tool_result)[:300]}...

        Did the tool result satisfy the goal? Answer YES or NO.
        """
        resp = await self._llm_generate_text(prompt)
        return "YES" in resp.upper()

    async def run(self, user_query: str):
        """
        智能体主执行流：
        分解 -> 批量检索 -> 执行 & 动态图更新
        """
        # 1. 记录用户输入
        self.history.append({"role": "user", "content": user_query})
        self.trajectory_buffer = []  # 清空当前轨迹

        # 2. 任务分解
        print(f"\n>>> Decomposing: {user_query}")
        sub_tasks = await self.decompose_query(user_query)
        print(f">>> Plan: {json.dumps(sub_tasks, indent=2, ensure_ascii=False)}")

        # 3. 批量检索 (Simultaneous Retrieval)
        # 根据题目要求，在分解完后对所有子任务进行检索。
        # 此时基于的是“当前”SGC图状态（如果是第一次运行，就是离散图）。
        execution_plan = []

        print("\n>>> Batch Retrieving Tools (GPU Accelerated)...")
        for task in sub_tasks:
            # 构造检索 Query: 动作 + 内容
            search_str = f"{task.get('action', '')} {task.get('query', '')}"

            # 向量化 (GPU)
            query_vec = self.get_text_embedding(search_str)

            # SGC 检索 (GPU)
            # search 返回最佳工具信息字典
            best_tool = self.retriever.search(query_vec, top_k=5)

            execution_plan.append({
                "task": task,
                "tool": best_tool
            })

        # 4. 逐步执行与图更新
        context_str = ""

        for idx, step in enumerate(execution_plan):
            task = step['task']
            tool_data = step['tool']

            tool_name = tool_data['name']
            tool_id = tool_data['id']

            print(f"\n[Step {idx + 1}] Goal: {task['query']}")
            print(f"         Tool: {tool_name} (SGC Score: {tool_data['score']:.4f})")

            # 4.1 参数生成
            args = await self.generate_tool_args(tool_name, task['query'], context_str)
            print(f"         Args: {args}")

            # 4.2 工具调用
            result = await self.call_tool(tool_name, args)

            # 4.3 反思
            success = await self.reflection(task['query'], tool_name, result)

            if success:
                print(f"         [Status] Success. Updating Graph Context.")
                context_str += f"\nStep {idx + 1} ({tool_name}): {result}"

                # --- SGC 图更新逻辑 ---
                # 要求：任务完成后，父子节点关系才写入 SGC 图
                if len(self.trajectory_buffer) > 0:
                    parent_id = self.trajectory_buffer[-1]  # 上一个成功的工具 ID
                    child_id = tool_id  # 当前成功的工具 ID

                    if parent_id != child_id:
                        # 1. 更新邻接矩阵 (GPU Tensor 操作)
                        self.graph_manager.add_edge(parent_id, child_id)

                        # 2. 重新计算 SGC Embeddings
                        # 这一步会利用 GPU 上的 adj 矩阵和 raw_embeddings 进行矩阵乘法
                        # 从而使下一次检索（或下一轮对话）能感知到这个工具链关系
                        self.retriever.final_embeddings = self.retriever.compute_sgc_embeddings()
                        print(
                            f"         [Graph] Edge added: {self.tool_names[parent_id]} -> {self.tool_names[child_id]}")

                # 将当前 ID 加入轨迹
                self.trajectory_buffer.append(tool_id)

            else:
                print(f"         [Status] Failed. Graph not updated.")
                context_str += f"\nStep {idx + 1} ({tool_name}) Failed."

        # 5. 生成最终回复
        final_prompt = f"User Query: {user_query}\nExecution History:\n{context_str}\n\nPlease verify if the user's question is answered and provide a summary."
        final_response = await self._llm_generate_text(final_prompt)

        self.history.append({"role": "assistant", "content": final_response})
        return final_response
