import os
import sys
import json
import glob
import inspect
import asyncio
import torch

print(f"当前工作目录: {os.getcwd()}")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Agents.run import Wrapper
from Toolregistry import ToolRegistry
from geoplan_bench import tools

BASE_PATH = "..."
CONFIG_PATH = os.path.join(BASE_PATH, "config.yaml")
TRAJECTORY_PATH = os.path.join(BASE_PATH, "geoplan-bench/data/train_trajectory.json")
TEST_DATA_PATH = os.path.join(BASE_PATH, "geoplan-bench/data/test_tasks")
EVAL_OUTPUT_PATH = os.path.join(BASE_PATH, "geoplan-bench/s")

def load_questions() -> list:
    """加载测试集问题"""
    if not os.path.exists(TEST_DATA_PATH):
        print(f"[Error] Test data path not found: {TEST_DATA_PATH}")
        return []

    json_files = sorted(glob.glob(os.path.join(TEST_DATA_PATH, "*.json")))
    questions = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            questions.append(data['question'])
    print(f"Loaded {len(questions)} questions from {TEST_DATA_PATH}")
    return questions

def registry_tools(registry: ToolRegistry, module):
    print(f"[*] Scanning module: {module.__name__}")
    
    members = inspect.getmembers(module, inspect.isfunction)
    if not members:
        print(f"[!] Warning: No functions found in module {module.__name__}. Check if tools.py is saved or empty.")
    
    for name, func in members:
        
        if func.__module__ != module.__name__:
            continue
            
        description = func.__doc__ if func.__doc__ else name
        description = registry.extract_short_description(description)
        
        parameters_schema = registry._infer_parameters_from_callable(func)
        
        meta = {
            "description": description,
            "parameters": parameters_schema,
            "source": "module_import"
        }
        
        registry.register_tool(name, func, meta)
        print(f"      -> [SUCCESS] Registered tool: {name}")


async def run_experiment(mode: str, model_name: str, questions: list):
    """运行单个模式的实验"""
    print(f"    STARTING EXPERIMENT MODE: [{mode.upper()}]")

    save_path = os.path.join(EVAL_OUTPUT_PATH, model_name, mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file = os.path.join(save_path, f"results")
    tool_log_file = os.path.join(save_path, f"tools_pool.json")

    agent = Wrapper(
        mode=mode,
        initial_model=model_name,
        output_dir=output_file,
        tool_dir=tool_log_file,
        trajectory_path=TRAJECTORY_PATH,
        config_path=CONFIG_PATH
    )

    registry = ToolRegistry()
    registry_tools(registry, tools)
    agent.tool_registry = registry
    TIMEOUT_SECONDS = 1200
    
    for idx, question in enumerate(questions):
        print(f"\n--- [Mode: {mode}] Question {idx + 1}/{len(questions)} ---")
        print(f"Q: {question}")
        
        try:
            result = await asyncio.wait_for(
                agent.run(question, trajectory_path=TRAJECTORY_PATH),
                timeout=TIMEOUT_SECONDS
            )
            print(f"Result: {result}")
            agent.save_data(query=question, final_result=result, id=idx)

        except asyncio.TimeoutError:
            print(f"[!] TASK TIMEOUT: Agent failed to finish within {TIMEOUT_SECONDS}s.")
            agent.save_data(query=question, final_result="TIMEOUT", id=idx, status="timeout")

        except Exception as e:
            print(f"[!] Error during agent run: {e}")
            import traceback
            traceback.print_exc()
            agent.save_data(query=question, final_result=f"ERROR: {e}", id=idx, status="error")

    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n[Finished] Experiment for mode '{mode}' completed.")

async def main():
    model_name = "deepseek-v3.2"

    # modes_to_run = ['bm25', 'dense', 'plan', 'sgc', 'plan_sgc']
    modes_to_run = ['plan_sgc']
    
    question_list = load_questions()
    questions = question_list
    if not questions:
        print("[!] No questions loaded. Exiting.")
        return
    
    for mode in modes_to_run:
        await run_experiment(mode, model_name, questions)

if __name__ == "__main__":
    asyncio.run(main())
    