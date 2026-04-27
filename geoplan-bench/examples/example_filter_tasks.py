"""
Task Filtering Example

This example demonstrates how to filter tasks using GeoPlan Benchmark.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoplan_bench.pipeline.task_validation import filter_tasks, generate_report
from geoplan_bench.utils.importance import analyze_tool_importance_from_tasks


def main():
    print("=" * 60)
    print("GeoPlan Benchmark - Task Filtering Example")
    print("=" * 60)
    
    input_dir = "data/tasks/raw"
    output_dir = "data/tasks/filtered"
    
    print(f"\n1. Load Tasks")
    print("-" * 60)
    print(f"Input directory: {input_dir}")
    
    print("\n2. Execute Filtering")
    print("-" * 60)
    
    filtered_tasks, stats = filter_tasks(input_dir)
    
    print("\n3. Filtering Statistics")
    print("-" * 60)
    print(f"Original task count: {stats['original_count']}")
    print(f"After tool flow filtering: {stats['tool_flow_filtered_count']}")
    print(f"After complexity filtering: {stats['complexity_filtered_count']}")
    print(f"After domain filtering: {stats['domain_filtered_count']}")
    print(f"Final task count: {stats['final_count']}")
    print(f"\nRemoved Statistics:")
    print(f"  Removed by tool flow filtering: {stats['removed_by_tool_flow']}")
    print(f"  Removed by complexity filtering: {stats['removed_by_complexity']}")
    print(f"  Removed by domain filtering: {stats['removed_by_domain']}")
    print(f"  Removed by semantic deduplication: {stats['removed_by_semantic']}")
    
    retention_rate = stats['final_count'] / stats['original_count'] * 100 if stats['original_count'] > 0 else 0
    print(f"\nRetention rate: {retention_rate:.1f}%")
    
    print("\n4. Generate Report")
    print("-" * 60)
    
    report = generate_report(stats, filtered_tasks)
    
    print("Domain Distribution:")
    for domain, count in report['filtering_report']['domain_distribution'].items():
        print(f"  {domain}: {count}")
    
    print("\nComplexity Distribution:")
    for complexity, count in report['filtering_report']['complexity_distribution'].items():
        print(f"  {complexity}: {count}")
    
    print("\n5. Save Filtered Tasks")
    print("-" * 60)
    
    os.makedirs(output_dir, exist_ok=True)

    for task in filtered_tasks:
        domain = task['domain']
        complexity = task['complexity']
        task_id = task['task_id']
        filename = f"task_{domain}_{complexity}_{task_id}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(task, f, ensure_ascii=False, indent=2)

    print("\n6. Analyze Tool Importance")
    print("-" * 60)
    analyze_tool_importance_from_tasks(output_dir)
    print(f"Tool importance analysis saved to: {os.path.join(output_dir, 'tool_importance_analysis.json')}")

    print("\n" + "=" * 60)
    print("Task filtering completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
