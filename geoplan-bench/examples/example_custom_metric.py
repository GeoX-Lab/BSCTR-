"""
Custom Evaluation Metric Example

This example demonstrates how to create a custom evaluation metric and integrate it into GeoPlan Benchmark.
"""

import sys
import os
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CustomMetricEvaluator:
    """
    Custom Evaluation Metric Example
    
    This example demonstrates how to create a simple evaluation metric.
    Actual implementation should include more complex evaluation logic.
    """
    
    def __init__(self):
        self.name = "CustomMetric"
    
    def compute_custom_score(self, agent_tool_flow: List[str], 
                             ground_truth_tool_flow: List[str]) -> Dict[str, float]:
        """
        Compute custom evaluation score
        
        This is a simple example computing length similarity of tool flows.
        Actual implementation can include more complex evaluation logic.
        """
        agent_length = len(agent_tool_flow)
        gt_length = len(ground_truth_tool_flow)
        
        length_ratio = agent_length / gt_length if gt_length > 0 else 0
        
        overlap = len(set(agent_tool_flow) & set(ground_truth_tool_flow))
        union = len(set(agent_tool_flow) | set(ground_truth_tool_flow))
        jaccard_similarity = overlap / union if union > 0 else 0
        
        return {
            "length_ratio": length_ratio,
            "jaccard_similarity": jaccard_similarity,
            "overlap_count": overlap
        }


def main():
    print("=" * 60)
    print("GeoPlan Benchmark - Custom Evaluation Metric Example")
    print("=" * 60)
    
    print("\n1. Create Custom Evaluator")
    print("-" * 60)
    
    evaluator = CustomMetricEvaluator()
    print(f"Evaluator Name: {evaluator.name}")
    
    print("\n2. Example Data")
    print("-" * 60)
    
    ground_truth = ["download_satellite_imagery", "geometric_correction", 
                    "atmospheric_correction", "classify_land_cover"]
    agent_flow = ["download_satellite_imagery", "geometric_correction", 
                 "classify_land_cover", "format_data"]
    
    print("Ground Truth Tool Flow:")
    for i, tool in enumerate(ground_truth, 1):
        print(f"  {i}. {tool}")
    
    print("\nAgent Tool Flow:")
    for i, tool in enumerate(agent_flow, 1):
        print(f"  {i}. {tool}")
    
    print("\n3. Compute Evaluation Score")
    print("-" * 60)
    
    scores = evaluator.compute_custom_score(agent_flow, ground_truth)
    
    print("Evaluation Results:")
    print(f"  Length Ratio: {scores['length_ratio']:.3f}")
    print(f"  Jaccard Similarity: {scores['jaccard_similarity']:.3f}")
    print(f"  Overlap Count: {scores['overlap_count']}")
    
    print("\n4. Integration into Evaluation Pipeline")
    print("-" * 60)
    print("To integrate custom evaluation metric into evaluation pipeline, you need:")
    print("1. Create evaluator class in geoplan_bench/evaluation/metrics/")
    print("2. Export it in geoplan_bench/evaluation/metrics/__init__.py")
    print("3. Use it in geoplan_bench/pipeline/task_evaluation.py")
    print("\nExample code:")
    print("""
        # In task_evaluation.py:
        from geoplan_bench.evaluation.metrics import CustomMetricEvaluator

        custom_evaluator = CustomMetricEvaluator()
        custom_score = custom_evaluator.compute_custom_score(
            agent_tool_trajectory, 
            ground_truth_tool_trajectory
        )

        eval_results[agent_name]["custom_metric"] = custom_score
        """)
    
    print("\n5. Evaluation Metric Design Recommendations")
    print("-" * 60)
    print("When designing evaluation metrics, consider:")
    print("  - Metrics should reflect agent performance")
    print("  - Metrics should be easy to understand and interpret")
    print("  - Metrics should be relevant to task objectives")
    print("  - Metrics should be able to distinguish between different agents")
    
    print("\n" + "=" * 60)
    print("Custom evaluation metric example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
