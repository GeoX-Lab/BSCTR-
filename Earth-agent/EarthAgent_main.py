import asyncio
import json
from Agent import SGCAgent
from Toolregistry import ToolRegistry
from tools import Analysis, Index, Inversion, Perception, Statistics


async def main():
    print(">>> Initializing System...")
    registry = ToolRegistry()

    # 定义要加载的工具模块列表
    tools = [
        Analysis,
        Perception,
        Statistics,
        Index,
        Inversion
    ]

    print(f"[*] Loading MCP tools from {len(tools)} files...")

    # 循环加载每个文件中的 'mcp' 对象
    for module in tools:
        if hasattr(module, 'mcp'):
            registry.load_from_fastmcp(module.mcp)
        else:
            print(f"[!] Warning: Module '{module.__name__}' does not have an 'mcp' object.")

    agent = SGCAgent(
        initial_model="qwen3-max"
    )

    agent.tool_registry = registry

    with open("./question.json", "r") as f:
        data = json.load(f)
    for item in data:
        user_query = item['dialogs'][0]['content']

        print(f"\n>>> Running User Query: {user_query}")
        print("-" * 50)

        # 运行 Agent
        try:
            final_result = await agent.run(user_query)

            print("-" * 50)
            print(f">>> Final Result:\n{final_result}")
            print("-" * 50)

        except Exception as e:
            print(f"\n[!] Agent Execution Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 使用 asyncio 运行主函数
    asyncio.run(main())