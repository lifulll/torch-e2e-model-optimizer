# torch-e2e-model-optimizer

A Codex skill for end-to-end PyTorch training and inference optimization when the workload already uses, or should be treated as, `torch.compile` first.

The workflow follows the optimization chain from model code to PyTorch runtime, Dynamo/AOTAutograd, TorchInductor, generated Triton kernels, and finally pass-inserted custom kernels only when profiling evidence justifies it. Each iteration records correctness, code/config changes, end-to-end performance, retained/rejected decisions, and stop reasons.

Required user inputs:

- model execution command,
- model-generated profile or trace directory,
- command to kill stale model processes before each run.

The skill includes scripts for environment capture, run-directory setup, cleanup-before-run execution, torch log analysis, iteration recording, log discovery, and final summary generation. See `SKILL.md` for the full agent workflow.
