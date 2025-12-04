import os
import json
import time
from collections import deque
from typing import Any, Optional, Dict, List
import concurrent.futures
import inspect
import asyncio
from datetime import datetime
from model import LLM
from ToolMem import ToolMem
from Toolregistry import ToolRegistry
from prompt import tool_agent_prompt, graph_build_prompt

class BaseAgent:
    def __init__(self,
                 initial_model: str,
                 sys_prompt_template: str,
                 output_dir: str = "outputs"):

        self.initial_model = initial_model
        self.sys_prompt_template = sys_prompt_template
        self.output_dir = output_dir
        self.llm = LLM(initial_model)
        self.tool_registry = ToolRegistry()
        self.tool_mem = ToolMem()
        self.history = []
        self.tool_list = []

    def history_save(self, filename: str):
        if filename is None:
            filename = os.path.join(self.output_dir, "history.json")
        with open(filename, "w") as f:
            for chunk in self.history:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    @staticmethod
    def _get_timestamp(format_type: str = "default") -> str:
        """
        获取当前时间戳
        """
        now = datetime.now()

        if format_type == "iso":
            return now.isoformat()
        elif format_type == "compact":
            return now.strftime("%Y%m%d_%H%M%S")
        elif format_type == "timestamp":
            return str(int(time.time()))
        else:  # default
            return now.strftime("%Y-%m-%d %H:%M:%S")

    def _append_msg(self, role: str, text: str):
        self.history.append({
            "role": role,
            "content": [{"type": "text", "text": text}],
            "ts": self._get_timestamp("iso")
        })

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
            return f"Tool '{tool_name}' not found in the registry."

        fn = tool["callable"]
        try:
            out = fn(**arguments)
            # awaitable（协程对象）
            if inspect.isawaitable(out):
                out = await out
            # 异步可迭代（async generator / async iterable）
            if hasattr(out, "__aiter__"):
                acc = ""
                async for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc
            # 同步可迭代（generator/iterable，但非字符串/字节/dict）
            elif hasattr(out, "__iter__") and not isinstance(out, (str, bytes, dict)):
                acc = ""
                for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc
            else:
                text = self._normalize_chunk(out)

            self.history.append({
                "role": "tool",
                "name": tool_name,
                "arguments": arguments,
                "content": [{"type": "text", "text": text}],
                "ts": self._get_timestamp("iso")
            })
            return text

        except Exception as e:
            err = f"Exception while running tool {tool_name}: {e}"
            self.history.append({
                "role": "tool",
                "name": tool_name,
                "arguments": arguments,
                "content": [{"type": "error", "text": str(e)}],
                "ts": self._get_timestamp("iso")
            })
            return err

    def _ensure_system_in_history(self):
        """
        将 sys_prompt_template写入history
        """
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, {"role": "system", "text": self.sys_prompt_template})
    async def chat(self, prompt, llm_name, image_path = None):
        """
        带有历史记录的对话函数
        """
        if llm_name:
            self.llm = LLM(llm_name)
        self._ensure_system_in_history()

        result = None
        acc = []
        if llm_name is not None:
            self.llm = LLM(llm_name)
        async for chunk in self.llm.generate_stream_res(prompt, self.history, image_path):
            if chunk.get("type") == "error":
                return f"Error: {chunk.get('error')}"
            if chunk.get("type") in ("text", "stream"):
                 acc.append(chunk.get("text", ""))
            if chunk.get("type") == "final":
                result = chunk.get("text", "")

        result = result if result is not None else "".join(acc)
        self._append_msg("user", prompt)
        self._append_msg("text", result)
        return result

class ToolAgent(BaseAgent):
    """
    ToolAgent：
    """
    def __init__(self, initial_model: str, sys_prompt_template: str, output_dir: str = "outputs"):
        super().__init__(initial_model, sys_prompt_template, output_dir)
        self.tool_history: List[Dict[str, Any]] = []

    def retrieve_tools(self, task: str) -> List[Dict[str, Any]]:
        """从 ToolMem 获取候选工具集合"""
        return self.tool_mem.get_similar_tools(task)

    async def execute_decided_tool(self, tool_decision: Dict[str, Any]):
        """执行 LLM 决策出的工具，并返回执行结果"""
        if not tool_decision or "tool_name" not in tool_decision:
            return "No valid tool decision."

        tool_name = tool_decision["tool_name"]
        arguments = tool_decision.get("arguments", {})

        result = await self.call_tool(tool_name, arguments)

        # 记录执行历史
        self.history.append({
            "role": "tool",
            "name": tool_name,
            "arguments": arguments,
            "content": [{"type": "text", "text": result}],
            "feedback": "Executed by ToolAgent decision",
            "ts": self._get_timestamp("iso")
        })

        self._extract_tool_history()
        return result

    async def execute_task(self, instruction: str) -> Dict[str, Any]:
        """
        处理单个任务：
        1) 检索候选工具
        2) 让 LLM 决策（decide_tool）
        3) 执行被选工具（execute_decided_tool）
        """
        candidates = self.retrieve_tools(instruction)
        if not candidates:
            return {"task": instruction, "result": "No tools found"}

        tool_list_str = json.dumps([t["name"] for t in candidates], ensure_ascii=False)
        prompt = tool_agent_prompt.format(tool_list=tool_list_str)
        llm_response = await self.chat(prompt, self.initial_model)
        tool_choice = self._parse_llm_response(llm_response)
        result = await self.execute_decided_tool(tool_choice)

        return {
            "task": instruction,
            "tool_choice": tool_choice,
            "result": result
        }

    async def execute_tasks(self, tasks: List[Dict[str, Any]]):
        """
        """
        all_results = []
        for task in tasks:
            instruction = task.get("instruction", "")
            try:
                out = await self.execute_task(instruction)
                all_results.append(out)
            except Exception as e:
                all_results.append({"task": instruction, "error": str(e)})
        return all_results

    def _extract_tool_history(self):
        """从历史中提取工具使用记录"""
        self.tool_history = [h for h in self.history if h.get("role") == "tool"]
    @staticmethod
    def _parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 输出中的工具选择结果"""
        try:
            response = response.strip()
            if response.startswith("{"):
                return json.loads(response)
            for line in response.splitlines():
                if "tool_name" in line:
                    return json.loads(line)
        except Exception as e:
            print(f"Failed to parse tool decision: {e}")
        return None

class ToolManager(BaseAgent):
    """
    负责工具图的初始化，与更新
    """
    def __init__(self, initial_model: str, sys_prompt_template: str, output_dir: str = "outputs"):
        super().__init__(initial_model, sys_prompt_template, output_dir)
        self.initial_model = initial_model
        self.sys_prompt_template = sys_prompt_template

    def tool_manager(self, prompt, llm_name=None):
        self.llm = LLM(llm_name)
        res = self.llm.generate_res(prompt)
        return res

    @staticmethod
    def evaluate_feedback(result) -> str:
        """
        根据任务执行结果评估工具节点的反馈
        例如，根据执行结果返回描述性反馈：“任务执行成功”、“任务执行失败”或“结果不明确”
        """
        if isinstance(result, str) and "error" in result.lower():
            # 如果执行结果包含 "error"，说明执行失败，返回失败的描述
            return "Task execution failed due to an error."
        elif result:
            # 如果执行结果非空且无错误，说明执行成功，返回成功的描述
            return "Task executed successfully."
        else:
            # 如果执行结果为空或不明确，返回结果不明确的描述
            return "Task executed with unclear results."
    @staticmethod
    def _prepare_arguments(sig: inspect.Signature, arguments: dict) -> inspect.BoundArguments:
        """准备函数参数，处理参数验证和默认值"""
        try:
            return sig.bind(**arguments)
        except TypeError as e:
            # 处理参数不匹配的情况
            raise ValueError(f"Argument mismatch for function: {str(e)}")

    @staticmethod
    def _parse_llm_response(llm_response: str) -> Optional[Dict[str, Any]]:
        """
        解析LLM响应，提取工具调用信息
        期望的格式示例:
        {
            "tool_name": "add_memory",
            "arguments": {
                "key": "user_preference",
                "value": "dark_mode"
            }
        }
        或者:
        TOOL: add_memory
        ARGUMENTS: {"key": "user_preference", "value": "dark_mode"}
        """
        try:
            # 尝试解析JSON格式
            if llm_response.strip().startswith('{'):
                return json.loads(llm_response)

            # 尝试解析键值对格式
            lines = llm_response.strip().split('\n')
            tool_info = {}

            for line in lines:
                if line.startswith('TOOL:'):
                    tool_info['tool_name'] = line.replace('TOOL:', '').strip()
                elif line.startswith('ARGUMENTS:'):
                    args_str = line.replace('ARGUMENTS:', '').strip()
                    tool_info['arguments'] = json.loads(args_str)

            return tool_info if tool_info else None

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse LLM response: {e}")
            return None

    def get_available_tools(self) -> list:
        """获取所有可用的工具列表"""
        tools = []
        for name in dir(self.tool_mem):
            if not name.startswith('_') and callable(getattr(self.tool_mem, name)):
                tools.append({
                    'name': name,
                    'doc': getattr(self.tool_mem, name).__doc__ or 'No documentation'
                })
        return tools

    def get_tool_history(self) -> list:
        """获取工具调用历史"""
        return self.history

class Mem_Builder:
    def __init__(self, initial_model: str, sys_prompt: str = tool_agent_prompt):
        """
        initial_model: 传给你自己的 LLM 封装的 model 名称
        sys_prompt:    系统提示词，默认用 BASE_SYS_PROMPT，也可以外部覆盖
        """
        self.initial_model = initial_model
        self.sys_prompt = sys_prompt
        self.batch_size = 1
        self.tool_mem = ToolMem()
        self.tool_node: Dict[str, Any] = {}
        self.llm = LLM(initial_model)

        # 记录多次无法建立连接的“待定”工具
        self.pending_tools = set()

    # ---------- 节点 ----------
    def get_node_doc(self) -> Dict[str, Any]:
        """
        从 ToolMem 的 node.json 读入所有工具，标准化为 {name: node}
        """
        self.tool_node = self.tool_mem.get_node_from_doc()
        return self.tool_node

    # ---------- 初始化工具筛选 ----------
    def _select_initial_tools(self, tools: List[Dict[str, Any]], first_n: int = 20) -> List[Dict[str, Any]]:
        """
        初始化建图阶段，优先选“基础 I/O / 数据源 / 预处理”工具作为起始节点。
        规则（打分）：
        - inputs 为空或很短 → 更可能是数据源 → +3
        - 描述中包含典型 I/O / 预处理关键词 → +3
        """
        keywords = [
            "load", "read", "open", "download",
            "raster", "tiff", "geotiff", "shapefile"
        ]

        scored: List[tuple] = []

        for tool in tools:
            desc = (tool.get("description") or "").lower()
            inputs = (tool.get("inputs") or "").lower()
            outputs = (tool.get("outputs") or "").lower()

            score = 0

            # 没有输入 / 输入很短：更像是 pipeline 的起点
            if not inputs or len(inputs) < 10:
                score += 3

            # 包含 I/O / 预处理关键词：进一步加分
            if any(k in desc for k in keywords):
                score += 3

            scored.append((score, tool))

        # 按分数从高到低排序
        scored.sort(key=lambda x: x[0], reverse=True)

        selected = [t for s, t in scored[:first_n] if s > 0]

        print(f"[INIT SELECT] selected {len(selected)} initial tools (score > 0):")
        for s, t in scored[:first_n]:
            print(f"  - {t['name']} (score={s})")

        if not selected:
            print("[INIT SELECT] no high-score IO tools found, fallback to first-N by name")
            selected = tools[:first_n]

        return selected

    # ---------- I/O 匹配子图 ----------
    def _find_subgraph_by_io(self, tool: Dict) -> List[str]:
        """
        Two-stage candidate search:
        1) I/O keyword matching (broad recall)
        2) semantic similarity filtering (precision boost)

        Output → Input 才可能建立边
        """

        outputs = (tool.get("outputs") or "").lower()
        if not outputs:
            return []

        tokens = outputs.split()

        # -------- Step 1: keyword I/O match --------
        initial_candidates = []
        for name, node in self.tool_mem.tool_node.items():
            if name == tool["name"]:
                continue

            inputs = (node.get("inputs") or "").lower()
            if not inputs:
                continue

            # output token 匹配 input
            if any(tok in inputs for tok in tokens):
                initial_candidates.append(name)

        if not initial_candidates:
            return []

        # -------- Step 2: semantic similarity filter --------
        q_vec = tool.get("vector")
        filtered = []

        for name in initial_candidates:
            node = self.tool_mem.tool_node[name]
            sim = self.tool_mem._cos_similarity(q_vec, node.get("vector"))

            # 阈值可调
            if sim > 0.25:
                filtered.append((name, sim))

        # semantic relevance sorting
        filtered.sort(key=lambda x: x[1], reverse=True)

        # 返回 Top-K 保证上下文不会爆炸
        return [name for name, _ in filtered[:5]]

    # ---------- Prompt 生成（单工具） ----------
    def _generate_batch_prompt(self, tool: Dict, candidates: List[str]) -> str:
        """
        针对单个工具和若干个候选下游工具生成 prompt。
        """
        name = tool.get("name")
        desc = tool.get("description") or ""
        inputs = tool.get("inputs")
        outputs = tool.get("outputs")

        prompt = f"""
            Tool Name: {name}
            Description: {desc}
            Inputs: {inputs}
            Outputs: {outputs}
            
            Candidate downstream tools based on I/O matching:
            {candidates}
            
            Now, following the rules, infer all valid workflow edges related to this tool.
            Remember:
            - Only connect when this tool's OUTPUT is required as another tool's INPUT.
            - Prefer realistic remote-sensing processing pipelines.
            - If no valid edges exist, return {{"edges": []}}.
        """

        return self.sys_prompt + "\n" + prompt

    # ---------- LLM ----------
    async def _llm_generate_edge(self, prompt: str) -> str:
        res = await self.llm.generate_res(prompt)
        return res

    def _run_async(self, batch_prompt: str) -> str:
        return asyncio.run(self._llm_generate_edge(batch_prompt))

    # ---------- 解析 ----------
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Dict]:
        edges: Dict[str, Dict[str, Any]] = {}
        try:
            parsed = json.loads(llm_response)
            for edge in parsed.get("edges", []):
                st = edge["start_tool"]
                ed = edge["end_tool"]
                edge_key = f"{st}->{ed}"
                edges[edge_key] = {
                    "start_node": st,
                    "end_node": ed,
                    "messages": edge.get("messages", ""),
                }
        except Exception as e:
            print(f"[ERROR] LLM parsing failed: {e}")
            print("Raw response:", llm_response)
        return edges

    def _add_edges_to_mem(self, edges: Dict[str, Dict]):
        for _, d in edges.items():
            self.tool_mem.add_edge(d["start_node"], d["end_node"], d.get("messages", []))

    # ---------- 初始化建图（不重试） ----------
    def _initial_build(self, tools: List[Dict[str, Any]]) -> None:
        """
        初始化阶段：只跑一遍，不做重试。
        目的是先建立一批基础 workflow 边，给后续工具提供上游支撑。
        """
        print("[INIT BUILD] Building base graph with initial tools...")

        for tool in tools:
            name = tool["name"]
            candidates = self._find_subgraph_by_io(tool)

            if not candidates:
                print(f"[INIT] {name} has no I/O match, skipping in init phase")
                continue

            # 控制候选数量，防止 prompt 过长
            candidates = candidates[:3]

            prompt = self._generate_batch_prompt(tool, candidates)
            llm_response = self._run_async(prompt)
            edges = self._parse_llm_response(llm_response)
            self._add_edges_to_mem(edges)

        print("[INIT BUILD] Done.")

    # ---------- 延迟重试建图 ----------
    def build_tool_graph(self, tool_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        对传入的工具列表做增量建图：
        - 没有 I/O 匹配的工具：放回队列，等其它工具建完边后重试
        - 每个工具最多重试 MAX_RETRY 次
        - 多次失败则加入 pending_tools，不强行连错边
        """
        if not tool_list:
            return {"nodes": self.tool_mem.tool_node, "edges": self.tool_mem.tool_edge}

        tool_queue = deque(tool_list)
        attempt_count = {t["name"]: 0 for t in tool_list}
        MAX_RETRY = 3

        while tool_queue:
            tool = tool_queue.popleft()
            name = tool["name"]

            candidates = self._find_subgraph_by_io(tool)

            if not candidates:
                attempt_count[name] += 1
                print(f"[DEFER] {name} no I/O match, retry {attempt_count[name]}/{MAX_RETRY}")

                if attempt_count[name] < MAX_RETRY:
                    # 放回队列，等待图结构更完整后再试
                    tool_queue.append(tool)
                    continue
                else:
                    # 多次失败，标记为 pending，但不强行连边
                    print(f"[PENDING] {name} stored as pending (no valid I/O match)")
                    self.pending_tools.add(name)
                    continue

            # 有候选 → 构造 prompt 调 LLM
            candidates = candidates[:3]  # 只取 Top-3，控制上下文长度
            prompt = self._generate_batch_prompt(tool, candidates)
            llm_response = self._run_async(prompt)
            edges = self._parse_llm_response(llm_response)
            self._add_edges_to_mem(edges)

        # 只在外部统一保存，这里不强制写盘
        return {"nodes": self.tool_mem.tool_node, "edges": self.tool_mem.tool_edge}

    # ---------- 从 node 文件构图 ----------
    def build_from_node_file(self, first_n: int = 20) -> Dict[str, Any]:
        """
        从 node.json 全量加载工具：
        - 第一步：筛选出 first_n 个“基础 I/O / 预处理”工具做初始化建图
        - 第二步：对剩余工具使用延迟重试机制补全边
        """
        all_nodes = self.get_node_doc()  # {name: node}
        tools: List[Dict[str, Any]] = []

        for name, node in all_nodes.items():
            node = dict(node)
            node["name"] = name
            tools.append(node)

        # 按名称排序，保证稳定性
        tools.sort(key=lambda x: x.get("name", ""))

        print(f"[LOAD] total tools: {len(tools)}")

        # Step 1: 初始化工具筛选 + 建图
        init_tools = self._select_initial_tools(tools, first_n=first_n)
        self._initial_build(init_tools)

        # Step 2: 对剩余工具做延迟重试建图
        init_names = {t["name"] for t in init_tools}
        remain_tools = [t for t in tools if t["name"] not in init_names]

        print(f"[REMAIN] {len(remain_tools)} tools left for deferred linking")
        self.build_tool_graph(remain_tools)

        # 最终持久化
        if self.tool_mem.tool_edge_path:
            self.tool_mem.save_edges_to_file()

        print(f"[PENDING] total pending tools: {len(self.pending_tools)}")

        return {"nodes": self.tool_mem.tool_node, "edges": self.tool_mem.tool_edge}


if __name__ == "__main__":
    def build_graph_from_node_file(
            initial_model: str,
            sys_prompt: str,
            node_path: str,
            edge_path: str,
            first_n: int = 20,
            batch_size: int = 5,
    ) -> Dict[str, Any]:
        builder = Mem_Builder(initial_model=initial_model, sys_prompt=sys_prompt)
        builder.batch_size = batch_size
        builder.tool_mem = ToolMem(tool_node_path=node_path, tool_edge_path=edge_path)
        result = builder.build_from_node_file(first_n=first_n)
        return result

    # tool_mem = ToolMem()
    # tool_mem.load_tools_from_json_list("./eval/earth_agent/earth_tools.json")

    build_graph_from_node_file(initial_model="qwen3-max", sys_prompt=graph_build_prompt, node_path="./tools_graph/node.json", edge_path="./tools_graph/edge.json")
