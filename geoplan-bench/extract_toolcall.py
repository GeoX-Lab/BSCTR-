import json
import os

def extract_tool_names(input_folder, output_file):
    final_result = {}

    if not os.path.exists(input_folder):
        print(f"错误：目录 '{input_folder}' 不存在。")
        return

    files = os.listdir(input_folder)
    json_files = [f for f in files if f.endswith('.json')]
    
    print(f"找到 {len(json_files)} 个JSON文件，开始处理...")

    for filename in json_files:
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取 task_id
            task_id = str(data.get('task_id', os.path.splitext(filename)[0]))
            
            tools_set = []
            history = data.get('history', [])
            
            if isinstance(history, list):
                for turn in history:
                    if turn.get('role') == 'tool':
                        tool_name = turn.get('name')
                        if tool_name and tool_name not in tools_set:
                            tools_set.append(tool_name)
            
            final_result[task_id] = tools_set

        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")

    # ================= 关键修改：排序逻辑 =================
    print("正在对结果进行排序...")
    
    # 我们尝试将 key 转为整数进行排序（实现 0, 1, 2, 10 的自然顺序）
    # 如果转换失败（例如 key 是 "task_abc"），则回退到普通的字符串排序
    def sort_key(item):
        key, _ = item
        try:
            return int(key)
        except ValueError:
            return key

    # 对 final_result 的 items 进行排序，并生成新的字典
    sorted_result = dict(sorted(final_result.items(), key=sort_key))
    # ====================================================

    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # 写入排序后的字典
            json.dump(sorted_result, f_out, indent=2, ensure_ascii=False)
        print(f"处理完成！结果已保存至: {output_file}")
    except Exception as e:
        print(f"写入输出文件时发生错误: {e}")


if __name__ == "__main__":
    input_directory = "/media/csudxy0218/ZL/AgentToolmem/geoplan-bench/outputs/deepseek-v3.2/graph/results" 
    output_filename = "/media/csudxy0218/ZL/AgentToolmem/geoplan-bench/outputs/deepseek-v3.2/graph/tool_call.json"

    extract_tool_names(input_directory, output_filename)