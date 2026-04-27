"""
Task Evaluation Example

This example demonstrates how to evaluate agents using GeoPlan Benchmark.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoplan_bench.pipeline.task_evaluation import RemoteSensingTaskEval


def main():
    print("=" * 60)
    print("GeoPlan Benchmark - Task Evaluation Example")
    print("=" * 60)
    
    task_dir = "data/tasks/filtered"
    
    print(f"\n1. Load Tasks")
    print("-" * 60)
    print(f"Task directory: {task_dir}")
    
    if not os.path.exists(task_dir):
        print(f"Error: Task directory {task_dir} does not exist")
        print("Please run task generation and filtering first")
        return
    
    tasks = []
    for filename in os.listdir(task_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(task_dir, filename)
            print(f"  Loading file: {filename}")
            with open(filepath, 'r', encoding='utf-8') as f:
                task = json.load(f)
                tasks.append(task)
    
    if not tasks:
        print("Error: No task files found")
        return
    
    print(f"Total tasks loaded: {len(tasks)}")
    
    print("\n2. Initialize Evaluator")
    print("-" * 60)
    
    evaluator = RemoteSensingTaskEval()
    print("Evaluator initialized")
    
    print("\n3. Evaluate Single Task Example")
    print("-" * 60)
    
    if len(tasks) > 0:
        task = tasks[0]
        print(f"Task ID: {task.get('task_id', 'unknown')}")
        print(f"Question: {task.get('question', 'N/A')}")
        print(f"Domain: {task.get('domain', 'N/A')}")
        print(f"Complexity: {task.get('complexity', 'N/A')}")
        print("\nEvaluating...")
        
        try:
            eval_results = evaluator.evaluate_task(task)
            
            print("\nEvaluation Results:")
            print("-" * 60)
            
            for agent_name, metrics in eval_results.items():
                print(f"\n{agent_name}:")
                print(f"  Tool Trajectory Length: {len(metrics.get('tool_trajectory', []))}")
                print(f"  Key Step Recall: {metrics.get('key_step_recall', 0):.3f}")
                print(f"  Key Tool Precision: {metrics.get('key_tool_precision', 0):.3f}")
                print(f"  F1 Score: {metrics.get('F1_score', 0):.3f}")
                print(f"  Tool Flow Similarity: {metrics.get('tool_flow_similarity', 0):.3f}")
                print(f"  Enhanced Edit Distance: {metrics.get('enhanced_edit_distance', 0):.3f}")
                print(f"  Completeness Score: {metrics.get('completeness_score', 0):.1f}")
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n4. Batch Evaluation Instructions")
    print("-" * 60)
    print("To evaluate multiple tasks in batch, use the following code:")
    print("""
        from geoplan_bench.pipeline.task_evaluation import execute_task_evaluation_pipeline

        execute_task_evaluation_pipeline(
            start_from_task_index=0,
            end_to_task_index=10
        )
        """)
    
    print("\n5. View Evaluation Results")
    print("-" * 60)
    print("Evaluation results are saved in data/eval_results/ directory")
    print("Each task's result file is named: eval_{task_id}.json")
    
    print("\n" + "=" * 60)
    print("Task evaluation example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
