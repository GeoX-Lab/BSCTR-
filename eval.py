from eval.earth_agent.eval_tool_mem import run_tool_rag_experiment, plot_pr_curve, plot_recall_precision

recall_results, precision_results = run_tool_rag_experiment(groundtruth_path="eval/earth_agent/toolRAG_ground_truth.json", tools_path="tools_graph/node.json")
plot_recall_precision(recall_results, precision_results, [4, 5, 6], out_dir="eval/earth_agent/plots/tool_mem")
plot_pr_curve(recall_results, precision_results, [4, 5, 6], out_dir="eval/earth_agent/plots/tool_mem")