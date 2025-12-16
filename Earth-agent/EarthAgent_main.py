import asyncio
import json
import re
import os
from Agent import SGCAgent
from Toolregistry import ToolRegistry
import Analysis, Index, Inversion, Perception, Statistics


def parse_raster_filename(fname: str):
    """
    解析类似：
    Xinjiang_2019-02-02_LST.tif
    """
    name = os.path.splitext(fname)[0]

    m = re.match(r"(.+?)_(\d{4}-\d{2}-\d{2})_(.+)", name)
    if not m:
        return {
            "region": None,
            "date": None,
            "semantic": "unknown",
            "path": fname
        }

    region, date, semantic = m.groups()
    return {
        "region": region,
        "date": date,
        "semantic": semantic.upper(),
        "path": fname
    }

def build_image_assets(image_dir: str):
    assets = []

    if not os.path.isdir(image_dir):
        print(f"[!] Image directory not found: {image_dir}")
        return assets

    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".tif", ".tiff", ".jp2", ".png")):
            asset = parse_raster_filename(fname)
            asset["path"] = os.path.join(image_dir, fname)
            asset["type"] = "raster"
            assets.append(asset)

    return assets

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
    print("[*] Registry successfully initialized.")

    with open("./question.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 你当前用的是 "1" 这个 key
    question_id = "1"
    user_query = data[question_id]["dialogs"][0]["content"]

    # ---------- Load Image Assets (System Layer) ----------
    image_dir = os.path.join("./Earth-Bench", f"question{question_id}")
    assets = build_image_assets(image_dir)

    # ---------- Inject Assets into Prompt (EarthAgent-style) ----------
    if assets:
        asset_lines = []
        for a in assets:
            asset_lines.append(
                f"- {a['semantic']} raster for {a['region']} on {a['date']}: {a['path']}"
            )

        asset_block = "\n".join(asset_lines)

        augmented_query = f"""
    ### Available Data Assets (System Provided)

    The following raster files are already available for this task:

    {asset_block}

    You MUST use these file paths exactly when calling tools.

    ---

    ### Task
    {user_query}
    """.strip()
    else:
        augmented_query = user_query

    try:
        final_result = await agent.run(augmented_query)

        print("-" * 50)
        print(f">>> Final Result:\n{final_result}")
        print("-" * 50)

    except Exception as e:
        print(f"\n[!] Agent Execution Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # for item in data:
    #     user_query = item['dialogs'][0]['content']

    #     print(f"\n>>> Running User Query: {user_query}")
    #     print("-" * 50)

    #     # 运行 Agent
    #     try:
    #         final_result = await agent.run(user_query)

    #         print("-" * 50)
    #         print(f">>> Final Result:\n{final_result}")
    #         print("-" * 50)

    #     except Exception as e:
    #         print(f"\n[!] Agent Execution Failed: {e}")
    #         import traceback
    #         traceback.print_exc()

if __name__ == "__main__":
    # 使用 asyncio 运行主函数
    asyncio.run(main())