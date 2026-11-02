"""Rebuild and compare the DeepSeek course-run metrics from raw trajectories."""
import argparse
import json
import os

from run_eval import build_metrics, _aggregate_multi_run_metrics


def load_mode(root, mode):
    output_path = os.path.join(root, f"{mode}.json")
    with open(output_path, encoding="utf-8") as f:
        output = json.load(f)
    if output and all(k.startswith("run_") for k in output):
        run_outputs = [output[k] for k in sorted(output, key=lambda x: int(x.split("_")[1]))]
    else:
        run_outputs = [output]
    metrics = [build_metrics(run) for run in run_outputs]
    aggregate = _aggregate_multi_run_metrics(metrics)
    metrics_path = os.path.join(root, f"{mode}_metrics.json")
    configuration = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            old_metrics = json.load(f)
        configuration = old_metrics.get("run_configuration", {})
    saved_metrics = {f"run_{i}": m for i, m in enumerate(metrics)}
    saved_metrics["mean"] = aggregate
    saved_metrics["run_configuration"] = configuration
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(saved_metrics, f, ensure_ascii=False, indent=2)
    return {
        "runs": metrics,
        "aggregate": aggregate,
        "n_runs": len(metrics),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing default.json and sparse_tool.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    modes = {name: load_mode(args.root, name) for name in ("default", "sparse_tool")}
    comparison = {}
    for metric in ("final_accuracy", "avg_tool_calls", "api_prompt_tokens", "api_completion_tokens"):
        base = modes["default"]["aggregate"].get("overall", {}).get(metric)
        changed = modes["sparse_tool"]["aggregate"].get("overall", {}).get(metric)
        comparison[metric] = {
            "default_mean": base,
            "sparse_mean": changed,
            "sparse_minus_default": round(changed - base, 6) if base is not None and changed is not None else None,
        }

    summary = {
        "experiment": "WHEN2TOOL CalculatorEnv: DeepSeek default vs sparse_tool",
        "n_tasks_per_run": modes["default"]["runs"][0]["overall"]["n"],
        "modes": modes,
        "comparison": comparison,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# DeepSeek CalculatorEnv Results",
        "",
        "| Policy | Accuracy (mean ± SD) | Avg tool calls (mean ± SD) | Prompt tokens (mean/run) | Completion tokens (mean/run) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in (("default", "Default"), ("sparse_tool", "Sparse")):
        agg = modes[mode]["aggregate"]
        overall = agg.get("overall", {})
        stdev = agg.get("overall_std", {})
        lines.append(
            f"| {label} | {overall.get('final_accuracy', 0):.4f} ± {stdev.get('final_accuracy', 0):.4f} "
            f"| {overall.get('avg_tool_calls', 0):.4f} ± {stdev.get('avg_tool_calls', 0):.4f} "
            f"| {overall.get('api_prompt_tokens', 0):,.0f} | {overall.get('api_completion_tokens', 0):,.0f} |"
        )
    lines.extend(["", "## By difficulty", "", "| Difficulty | Policy | Accuracy (mean ± SD) | Avg tool calls (mean ± SD) |", "|---|---|---:|---:|"])
    for diff in ("easy", "medium", "hard"):
        for mode, label in (("default", "Default"), ("sparse_tool", "Sparse")):
            agg = modes[mode]["aggregate"]
            vals = agg.get("by_difficulty", {}).get(diff, {})
            stds = agg.get("by_difficulty_std", {}).get(diff, {})
            lines.append(
                f"| {diff.title()} | {label} | {vals.get('final_accuracy', 0):.4f} ± {stds.get('final_accuracy', 0):.4f} "
                f"| {vals.get('avg_tool_calls', 0):.4f} ± {stds.get('avg_tool_calls', 0):.4f} |"
            )
    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {args.output} and {md_path}")


if __name__ == "__main__":
    main()
