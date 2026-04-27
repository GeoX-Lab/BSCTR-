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

BASE_PATH = "..."
CONFIG_PATH = os.path.join(BASE_PATH, "config.yaml")
API_BANK_PATH = os.path.join(BASE_PATH, "API-Bank/api_bank")
TRAJECTORY_PATH = None
TEST_DATA_PATH = os.path.join(BASE_PATH, "API-Bank/test-data")
EVAL_OUTPUT_PATH = os.path.join(BASE_PATH, "API-Bank/f")

def load_api_bank_questions(level: str = "level-3") -> list:
    """从 API-Bank 加载测试集问题"""
    json_file = os.path.join(TEST_DATA_PATH, f"{level}-api.json")
    
    if not os.path.exists(json_file):
        print(f"[Error] Test data not found: {json_file}")
        return []
    
    questions = []
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse JSON file: {e}")
            return []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if 'requirement' in item:
                        questions.append(item['requirement'])
                    elif 'query' in item:
                        questions.append(item['query'])
                    elif 'instruction' in item:
                        questions.append(item['instruction'])
        elif isinstance(data, dict) and 'queries' in data:
            questions.extend(data['queries'])
    
    print(f"Loaded {len(questions)} questions from {json_file}")
    if len(questions) == 0:
        print(f"[Warning] No questions found in {json_file}")

    return questions


def registry_api_bank_tools(registry: ToolRegistry):

    print(f"[*] Scanning API-Bank module...")
    
    api_bank_path = os.path.join(API_BANK_PATH, "apis")
    
    if not os.path.exists(api_bank_path):
        print(f"[!] ERROR: API path not found: {api_bank_path}")
        return
    
    api_files = sorted(glob.glob(os.path.join(api_bank_path, "*.py")))
    registered_count = 0
    
    for api_file in api_files:
        module_name = os.path.basename(api_file)[:-3]
        
        if module_name.startswith("_") or module_name == "api":
            continue
        
        try:
            import importlib.util
            spec_obj = importlib.util.spec_from_file_location(module_name, api_file)
            if spec_obj is None or spec_obj.loader is None:
                continue
                
            mod = importlib.util.module_from_spec(spec_obj)
            spec_obj.loader.exec_module(mod)
            
            members = inspect.getmembers(mod, inspect.isclass)
            
            for class_name, cls in members:
                if class_name.startswith("_"):
                    continue
                if class_name == "API" or cls.__module__ != module_name:
                    continue
                if not hasattr(cls, 'call'):
                    continue
                
                try:
                    api_instance = cls()
                    api_callable = api_instance.call
                    
                    description = getattr(cls, 'description', class_name)
                    description = registry.extract_short_description(description)
                    
                    input_params = getattr(cls, 'input_parameters', {})
                    
                    properties = {}
                    required = []
                    for param_name, param_info in input_params.items():
                        param_type = param_info.get('type', 'string')
                        if 'int' in param_type.lower():
                            json_type = 'number'
                        elif 'bool' in param_type.lower():
                            json_type = 'boolean'
                        elif 'list' in param_type.lower():
                            json_type = 'array'
                        else:
                            json_type = 'string'
                        
                        properties[param_name] = {
                            'type': json_type,
                            'description': param_info.get('description', '')
                        }
                        required.append(param_name)
                    
                    parameters = {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                    
                    # 注册工具
                    meta = {
                        "description": description,
                        "parameters": parameters,
                        "source": "api_bank"
                    }
                    
                    registry.register_tool(class_name, api_callable, meta)
                    registered_count += 1
                
                except Exception as e:
                    print(f"[!] Warning: Failed to register {class_name}: {str(e)[:60]}")
                    continue
        
        except Exception as e:
            print(f"[!] Warning: Failed to load {module_name}: {str(e)[:60]}")
            continue
    
    print(f"[*] Total APIs registered: {registered_count}")

async def run_experiment(mode: str, level: str, model_name: str, questions: list):
    print(f"    STARTING EXPERIMENT MODE: [{mode}] LEVEL: [{level.upper()}]")

    save_path = os.path.join(EVAL_OUTPUT_PATH, model_name, level, mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file = os.path.join(save_path, f"results")
    tool_log_file = os.path.join(save_path, f"tools_pool.json")

    agent = Wrapper(
        mode=mode,
        initial_model=model_name,
        output_dir=output_file,
        tool_dir=tool_log_file,
        trajectory_path=None,
        config_path=CONFIG_PATH
    )

    registry = ToolRegistry()
    registry_api_bank_tools(registry)
    
    if len(registry.tools) == 0:
        print(f"[!] ERROR: No tools registered! Aborting experiment for level {level}")
        return
    
    print(f"[*] Successfully loaded {len(registry.tools)} tools")
    
    agent.tool_registry = registry
    TIMEOUT_SECONDS = 1200
    
    for idx, question in enumerate(questions):
        print(f"\n{'='*80}")
        print(f"[Task {idx + 1}/{len(questions)}] [Level: {level}] [Mode: {mode}]")
        print(f"{'='*80}")
        print(f"[Query] {question}")
        print(f"{'='*80}")
        
        try:
            print(f"[Status] 开始执行任务...")
            result = await asyncio.wait_for(
                agent.run(question, trajectory_path=None),
                timeout=TIMEOUT_SECONDS
            )
            print(f"[Status] 任务执行完成")
            print(f"[Result] {result}")
            agent.save_data(query=question, final_result=result, id=idx)
            print(f"[Status] ✓ 结果已保存")

        except asyncio.TimeoutError:
            print(f"[Status] ✗ 任务超时")
            print(f"[Error] TASK TIMEOUT: Agent failed to finish within {TIMEOUT_SECONDS}s.")
            agent.save_data(query=question, final_result="TIMEOUT", id=idx, status="timeout")
            print(f"[Status] 超时结果已保存")

        except Exception as e:
            print(f"[Status] ✗ 任务执行出错")
            print(f"[Error] Error during agent run: {e}")
            import traceback
            traceback.print_exc()
            agent.save_data(query=question, final_result=f"ERROR: {e}", id=idx, status="error")
            print(f"[Status] 错误结果已保存")

    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n[Finished] Experiment for level '{level}' completed.")

async def main():
    model_name = "deepseek-v3.2"
    
    levels_to_run = ['level-3']
    
    # modes_to_run = ['bm25', 'dense', 'plan', 'sgc', 'plan_sgc']
    modes_to_run = ["plan_sgc"]
    
    for mode in modes_to_run:
        for level in levels_to_run:
            questions = load_api_bank_questions(level)
            if not questions:
                print(f"[!] No questions loaded for {level}. Skipping.")
                continue
            await run_experiment(mode, level, model_name, questions)

if __name__ == "__main__":
    asyncio.run(main())