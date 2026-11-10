# FTEC5660 Reproducibility Work

## Project and reproduction target

**Project:** WHEN2TOOL, using the authors' public repository at commit `8c00ef7b4f7576de5f00e7ed2d4bf99cdac7f30a`.

**Target behavior:** the paper reports that prompts discouraging tool calls can reduce usage indiscriminately, with hard tasks paying a larger accuracy cost. I reproduce that behavior on the full 150-task `CalculatorEnv` test subset, comparing the default `current` prompt with the authors' `sparse_tool` prompt.

The assignment permits one clearly defined benchmark slice. This experiment preserves all three difficulty levels (50 tasks each) and repeats each setting three times. Its numbers are specific to DeepSeek and CalculatorEnv; they do not reproduce the paper's Qwen/Llama table values.

## Paper result comparison

Table 2 in the paper reports the change from Default to Sparse for Prompt-only baselines. On hard tasks, Qwen3-4B-Instruct loses 20.3 accuracy points and makes 0.48 fewer tool calls per task; Qwen3-14B loses 27.3 points and makes 0.47 fewer calls. In this DeepSeek CalculatorEnv slice, hard-task accuracy falls by 58 points while calls fall by 0.7867 per task. The direction matches the paper. The magnitudes are not directly comparable because the model differs and this reproduction covers one environment (50 tasks per difficulty), whereas the paper evaluates the full benchmark. This supports the qualitative behavior, not the paper's numeric result. Source: Sun et al., arXiv:2605.09252v3, Table 2, 4 Nov 2026.

## Model substitution and scope limit

The original paper evaluates local Qwen/Llama models and uses their hidden states. The requested model is DeepSeek's hosted API, accessed with model ID `deepseek-flash` at `https://api.deepseek.com`. DeepSeek documents this model ID as DeepSeek-V4.1-Flash-1017 as of this run. The hosted API does not expose hidden states, so PROBE&PREFILL's linear probe and AUROC cannot be evaluated here. Instead, this project reproduces a measurable prompt-baseline behavior in WHEN2TOOL. The model substitution is a controlled variable and is not the experiment's selected modification.

## Setup and method

- **Repository:** `Trustworthy-ML-Lab/when2tool`, pinned local source commit above.
- **Dataset:** `cesun/When2Tool`, Hugging Face revision `4a6d05f`; test split filtered to CalculatorEnv, all difficulties, 150 examples.
- **Provider/model:** DeepSeek API, `deepseek-flash` (DeepSeek-V4.1-Flash-1017 at run time).
- **Base URL:** `https://api.deepseek.com`.
- **Decoding:** non-thinking mode; temperature 0.7; top-p omitted; max output 512 tokens; native function calls with `tool_choice=auto`; up to 10 evaluation rounds; 4 concurrent requests.
- **Prompt settings:** default `current` versus `sparse_tool`; no-reasoning mode; authors' system prompt and task tool schemas retained.
- **Repetitions:** three runs per prompt setting. Local run IDs/seeds are logged, but DeepSeek's chat API does not accept a supported sampling seed; the values do not control server-side randomness. Report mean and population standard deviation.
- **Secrets:** key read from the ignored local `.env`; it is not part of tracked files or result metadata.

Run with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-deepseek.txt
# Set DEEPSEEK_API_KEY in .env
./run_course_reproduction.sh
```

## Metrics

- **Accuracy:** fraction of tasks whose boxed final answer matches the benchmark answer under the repository evaluator.
- **Average tool calls:** executed calls divided by 150 tasks.
- **API token usage:** sum of prompt and completion tokens returned by the provider. The repository runner now records these per task and per run.
- **Variation:** mean ± population SD across the three runs.

## Results

| Policy | Accuracy, mean ± SD | Average tool calls, mean ± SD | Prompt tokens/run | Completion tokens/run |
|---|---:|---:|---:|---:|
| Default (`current`) | 1.0000 ± 0.0000 | 0.7289 ± 0.0113 | 130,736 | 6,882 |
| Sparse (`sparse_tool`) | 0.8067 ± 0.0196 | 0.0867 ± 0.0054 | 158,154 | 28,857 |

| Difficulty | Default accuracy | Default calls/task | Sparse accuracy | Sparse calls/task |
|---|---:|---:|---:|---:|
| Easy | 1.0000 ± 0.0000 | 0.1400 ± 0.0283 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Medium | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Hard | 1.0000 ± 0.0000 | 1.0467 ± 0.0094 | 0.4200 ± 0.0589 | 0.2600 ± 0.0163 |

Sparse prompting reduced tool calls by **88.1%** (0.7289 to 0.0867 per task) and reduced accuracy by **19.3 percentage points** overall. Easy and medium accuracy stayed at 100%; hard accuracy fell from 100% to 42%. This reproduces the paper's qualitative observation that a sparse prompt saves calls but can suppress useful calls on hard tasks. It does not reproduce a paper model's numeric result.

Sparse prompting also increased mean completion tokens per run from 6,882 to 28,857, because more tasks needed extra answer attempts. Across both settings the experiment used about 867,000 prompt tokens and 107,000 completion tokens. Using the provider's listed rates and assuming every prompt token was a cache miss, the estimated total is about **US$0.20–$0.39** depending on off-peak or peak rates; cache hits lower the actual charge. This is an estimate, not a billing statement.

Price source: [DeepSeek API Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/), checked 10 November 2026.

## Selected modification and impact

**Modification:** changed one prompt-policy setting from the default to the existing `sparse_tool` instruction, which tells the agent to default to zero tool calls and try solving directly.

**Impact:** calls dropped 88.1%, while accuracy dropped 19.3 points. The hard-task result (42% accuracy with 0.26 calls per task) shows the cost of suppressing needed tools; easy and medium tasks kept full accuracy with no calls. This is the one isolated, measurable modification reported for the assignment.

The implementation work adds a DeepSeek API backend, native tool-call/response handling, a pinned dataset snapshot, request concurrency control, API token accounting, and per-run checkpoints. These changes make the selected homework experiment runnable and auditable; the DeepSeek model substitution is documented separately as a controlled variable.

## Debug diary

| Date | Blocker | Resolution |
|---|---|---|
| 2026-11-02 | The official inference wrappers supported local Transformers/vLLM models only. | Added a DeepSeek API backend using the provider's OpenAI-compatible function-call format. |
| 2026-11-02 | The workspace could not write to the default Hugging Face cache path. | Set `HF_HOME` to an ignored project-local cache directory. |
| 2026-11-02 | A function call with accompanying assistant text was rejected by the benchmark's no-reasoning validator before its tool reply was appended, leaving invalid API history. | Kept structured API tool calls paired with their tool replies and skipped the free-form reasoning check for structured API calls. Added per-run checkpoint output. |

## Conclusion and limitations

The sparse-prompt trade-off reproduced qualitatively on DeepSeek and CalculatorEnv: fewer calls came with lower accuracy, concentrated on hard tasks. The default policy was already fully accurate and used almost no tools on easy tasks, while making about one call per medium/hard task. The DeepSeek API produces variable outputs and offers no supported seed parameter, so repetition quantifies observed variation rather than guaranteeing deterministic reruns.

This study uses one environment and a different hosted model. It cannot evaluate hidden-state AUROC or PROBE&PREFILL. The prompt comparison is a scoped reproducibility result, not a direct replication of the paper's full-model tables.

The homework submission still needs a public GitHub URL for the student's fork/copy and a recorded Zoom presentation link. The local branch is prepared for a course commit; do not push it to the authors' upstream repository.

## 15-minute presentation outline

1. (2 min) Paper question and reported tool-overuse behavior.
2. (3 min) WHEN2TOOL and CalculatorEnv scope.
3. (3 min) DeepSeek API adaptation, exact settings, and repeated-run method.
4. (3 min) Default versus sparse results, including the hard-task drop and token usage.
5. (2 min) The one prompt-policy modification and its measured impact.
6. (2 min) What reproduced, limitations, and practical recommendations.
