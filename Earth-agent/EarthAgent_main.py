import asyncio
import json
from Agent import SGCAgent
from Toolregistry import ToolRegistry
from tools import Analysis, Index, Inversion, Perception, Statistics


def load_questions(test_json_path: str = "./question.json"):
    """Load evaluation questions (EarthAgent official style)"""

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_json = json.load(f)

    out = []

    for question_idx, question_info in test_json.items():
        evaluations = question_info.get("evaluation", [])
        if len(evaluations) < 2:
            continue

        # Autonomous Planning 优先
        if evaluations[0]["type"].lower().startswith("autonomous"):
            ap_index = 0
            other_index = 1
        else:
            ap_index = 1
            other_index = 0

        # data fallback
        data = evaluations[ap_index].get("data", None)
        if data is None:
            data = evaluations[other_index].get("data", None)

        if data is None:
            continue

        out.append({
            "question_id": question_idx,
            "auto": evaluations[ap_index]["question"],
            "instruct": evaluations[other_index]["question"],
            "data": data,
            "choices": question_info.get("choices", None)
        })

    return out

def build_prompt(sample: dict):
    blocks = []

    # Task（Autonomous Planning）
    blocks.append(f"""### Task\n{sample["auto"]}""")

    # Data
    if sample.get("data"):
        blocks.append(f"""### Available Data
The required data for this task is located at:
{sample["data"]}
You must inspect this data using appropriate tools before reasoning.""")

    choices = sample.get("choices")
    if choices:
        choice_lines = []
        for i, c in enumerate(choices):
            label = chr(ord("A") + i)
            choice_lines.append(f"{label}. {c}")

    return "\n\n".join(blocks).strip()
async def main():
    print(">>> Initializing System...")
    registry = ToolRegistry()

    tool_modules = [
        Analysis,
        Perception,
        Statistics,
        Index,
        Inversion
    ]

    print(f"[*] Loading MCP tools from {len(tool_modules)} modules...")
    for module in tool_modules:
        if hasattr(module, "mcp"):
            registry.load_from_fastmcp(module.mcp)
        else:
            print(f"[!] Warning: {module.__name__} has no mcp object")

    # TODO 初始化智能体
    agent = SGCAgent(initial_model="qwen3-max", output_dir="/media/csudxy0218/ZL/AgentToolmem/Earth-agent/evaluate/qwen3-max/qwen3-max_outputs.jsonl")
    agent.tool_registry = registry
    print("[*] Agent and tool registry initialized.")

    samples = load_questions("./question.json")
    print(f"[*] Loaded {len(samples)} benchmark questions.")

    if not samples:
        print("[!] No valid questions found.")
        return

    # sample = samples[7]
    # user_query = build_prompt(sample)
    # choices = sample.get("choices")
    # try:
    #     result = await agent.run(user_query, choices)
    #
    #     print("\n================ RESULT ================")
    #     print(result)
    #     print("========================================")
    #
    # except Exception as e:
    #     print(f"[!] Agent execution failed: {e}")
    #     import traceback
    #     traceback.print_exc()

    samples_half = samples[:123]
    for sample in samples_half:
        user_query = build_prompt(sample)
        choices = sample.get("choices")

        # print("\n================ PROMPT ================")
        # print(user_query)
        # print("========================================\n")

        try:
            result = await agent.run(user_query,choices)

            print("\n================ RESULT ================")
            print(result)
            print("========================================")

        except Exception as e:
            print(f"[!] Agent execution failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
