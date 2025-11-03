import json
import requests
from Base import ToolNode
from typing import Dict
import numpy as np

class ExtractTool:
    def __init__(self):
        self.tool_path = "./gta_dataset/toolmeta.json"

    def get_embedding(self,text):
        """
        使用本地Ollama模型获取文本的向量表示
        """
        try:
            # Ollama的API端点，假设运行在本地默认端口
            url = "http://localhost:11434/api/embeddings"
            data = {
                "model": "nomic-embed-text:latest",
                "prompt": text
            }

            response = requests.post(url, json=data)
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding)

        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(768)

    def extract_tool_from_gta(self):

        """
        "Calculator": {
            "name": "Calculator",
            "description": "A calculator tool. The input must be a single Python expression and you cannot import packages. You can use functions in the `math` package without import.",
            "inputs": [
                {
                    "type": "text",
                    "name": "expression",
                    "description": null,
                    "optional": false,
                    "default": null,
                    "filetype": null
                }
            ],
            "outputs": [
                {
                    "type": "text",
                    "name": null,
                    "description": null,
                    "optional": false,
                    "default": null,
                    "filetype": null
                }
            ]
        }
        """
        tool_node: Dict[str, ToolNode] = {}
        with open(self.tool_path, "r") as f:
            tool_dict = json.load(f)

        for i, (tool_name, tool_info) in enumerate(tool_dict.items(), 1):
            inputs = tool_info["inputs"]
            outputs = tool_info["outputs"]
            vector = self.get_embedding(tool_info["description"])

            tool_node[str(i)] = ToolNode(
                name=tool_name,
                description=tool_info["description"],
                inputs=inputs,
                outputs=outputs,
                state=0,
                feedback=[],
                vector=vector
            )

        return tool_node

# if __name__ == '__main__':
#     extract_tool = ExtractTool()
#     print(extract_tool.extract_tool_from_gta())