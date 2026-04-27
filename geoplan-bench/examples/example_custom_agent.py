"""
Custom Agent Example

This example demonstrates how to create a custom agent and integrate it into GeoPlan Benchmark.
"""

import sys
import os
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoplan_bench.agents.base import BaseAgent
from geoplan_bench.tools import download_satellite_imagery, geometric_correction, classify_land_cover


class CustomAgent(BaseAgent):
    """
    Custom Agent Example
    
    This is a simple custom agent demonstrating how to implement the BaseAgent interface.
    """
    
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model=model, agent_type="Custom")
        
        self.add_tool(download_satellite_imagery)
        self.add_tool(geometric_correction)
        self.add_tool(classify_land_cover)
    
    def run(self, query: str) -> str:
        """
        Run the agent and return results
        
        This is a simplified example. Actual implementation should include complete reasoning logic.
        """
        result = f"CustomAgent processing query: {query}\n"
        result += "Executing tool call sequence:\n"
        
        tool_trajectory = self.run_and_return_tool_trajectory(query)
        for i, tool in enumerate(tool_trajectory, 1):
            result += f"{i}. {tool}\n"
        
        result += "\nTask completed!"
        return result
    
    def run_and_return_tool_trajectory(self, query: str) -> List[str]:
        """
        Run the agent and return tool calling trajectory
        """
        trajectory = []
        
        if "land" in query.lower() or "cover" in query.lower():
            trajectory.append("download_satellite_imagery")
            trajectory.append("geometric_correction")
            trajectory.append("classify_land_cover")
        else:
            trajectory.append("download_satellite_imagery")
        
        return trajectory


def main():
    print("=" * 60)
    print("GeoPlan Benchmark - Custom Agent Example")
    print("=" * 60)
    
    print("\n1. Create Custom Agent")
    print("-" * 60)
    
    agent = CustomAgent()
    print(f"Agent Type: {agent.agent_type}")
    print(f"Available Tools: {len(agent.tools)}")
    print("Available Tools:")
    for tool_name in agent.tools.keys():
        print(f"  - {tool_name}")
    
    print("\n2. Run Agent")
    print("-" * 60)
    
    query = "Analyze land cover types in a region"
    print(f"Query: {query}")
    
    result = agent.run(query)
    print("\nExecution Result:")
    print(result)
    
    print("\n3. Get Tool Trajectory")
    print("-" * 60)
    
    trajectory = agent.run_and_return_tool_trajectory(query)
    print("Tool Calling Trajectory:")
    for i, tool in enumerate(trajectory, 1):
        print(f"  {i}. {tool}")
    
    print("\n4. Integration into Evaluation Pipeline")
    print("-" * 60)
    print("To integrate custom agent into evaluation pipeline, you need:")
    print("1. Add creation function in geoplan_bench/utils/arena.py")
    print("2. Call it in geoplan_bench/pipeline/task_evaluation.py")
    print("\nExample code:")
    print("""
        # In arena.py:
        def create_custom_agent():
            return CustomAgent()

        # In task_evaluation.py:
        custom_agent = create_custom_agent()
        custom_agent_trajectory = custom_agent.run_and_return_tool_trajectory(question)
        agents_results["Custom"] = custom_agent_trajectory
        """)
    
    print("\n" + "=" * 60)
    print("Custom agent example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
