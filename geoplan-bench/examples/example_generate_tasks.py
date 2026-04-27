"""
Task Generation Example

This example demonstrates how to generate remote sensing tasks using GeoPlan Benchmark.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoplan_bench.pipeline.task_generation import RemoteSensingTaskPipeline


def main():
    print("=" * 60)
    print("GeoPlan Benchmark - Task Generation Example")
    print("=" * 60)
    
    pipeline = RemoteSensingTaskPipeline()
    
    print("\n1. Single Task Generation Example")
    print("-" * 60)
    
    domain = "Agriculture & Forestry"
    complexity = "Simple"
    
    print(f"Domain: {domain}")
    print(f"Complexity: {complexity}")
    print("Generating task...")
    
    tasks = pipeline.generate_task_with_ground_truth(complexity, domain)
    pipeline.export_tasks(tasks, output_dir="data/tasks/raw")
    
    if tasks and len(tasks) > 0:
        task = tasks[0]
        print(f"\nGenerated Task:")
        print(f"  Task ID: {task['task_id']}")
        print(f"  Question: {task['question']}")
        print(f"  Tool Flow: {task['ground_truth_tool_flow']}")
    else:
        print("Task generation failed")
    
    print("\n2. Batch Task Generation Example")
    print("-" * 60)
    
    num_dags = 2
    print(f"Generating {num_dags} DAG templates per domain-complexity combination")
    print("Generating tasks in batch...")
    
    all_tasks = pipeline.generate_batch_tasks(num_dags=num_dags)
    
    print(f"\nTotal tasks generated: {len(all_tasks)}")
    
    print("\n3. Task Statistics")
    print("-" * 60)
    
    domain_count = {}
    complexity_count = {}
    
    for task in all_tasks:
        domain = task.get('domain', 'Unknown')
        complexity = task.get('complexity', 'Unknown')
        
        domain_count[domain] = domain_count.get(domain, 0) + 1
        complexity_count[complexity] = complexity_count.get(complexity, 0) + 1
    
    print("Domain Distribution:")
    for domain, count in domain_count.items():
        print(f"  {domain}: {count}")
    
    print("\nComplexity Distribution:")
    for complexity, count in complexity_count.items():
        print(f"  {complexity}: {count}")
    
    print("\n" + "=" * 60)
    print("Task generation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
