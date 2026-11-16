# vendor/LESS adaptations (2026-11 bring-up)

LESS (princeton-nlp, 2023) is pinned to a `python -m less...`-style external checkout under
`vendor/LESS` (gitignored). Per the protocol, we adapt the checkout, never fork it, and never
touch our own code for these reasons. Applied changes (all re-appliable with `git diff` inside
the checkout; re-clone = re-run these):

1. **Pythia LoRA targets** — GPT-NeoX fuses QKV; LESS defaults target Llama/Mistral naming:
   - `less/scripts/train/base_training_args.sh`: `--lora_target_modules query_key_value`
     (was `q_proj k_proj v_proj o_proj`)
   - `less/data_selection/get_info.py`: default `--lora_target_modules` → `["query_key_value"]`
   (Keep `q_proj ...` when switching to Mistral/Llama models.)
2. **transformers CLI rename**: `base_training_args.sh`: `--eval_strategy no` (was
   `--evaluation_strategy no`; removed in transformers >= 4.46, we run 4.57 pinned `<5`).
3. **No W&B** for reproduction runs: `base_training_args.sh`: `--report_to none` (was `wandb`;
   avoids a wandb dependency and any network calls).
4. **Optimizer file rename**: `get_info.py` now falls back to `optimizer.pt` when
   `optimizer.bin` is absent (HF Trainer renamed the artifact; needed for `--grad_type adam`).
5. **Optimizer state re-keying**: `collect_grad_reps.prepare_optimizer_state` now maps the
   integer-keyed `optimizer.pt["state"]` (modern Trainer) back to parameter names
   (2023 `named_optimizer` format) via trainable-parameter order.
6. **Projector kwargs**: dropped `max_batch_size=` from the projector construction in
   `collect_grad_reps.py` — in trak v0.1.3 only `CudaProjector` accepts it, and the
   `BasicProjector` fallback (used when the optional `fast_jl` CUDA extension is absent,
   which is our case) rejects it. CudaProjector's default (32) is unaffected.
7. **trak pin**: `llm` extra uses `traker @ git+.../trak@v0.1.3` (dist renamed at HEAD;
   v0.1.3 = April 2023, the API generation LESS was written against; module remains `trak`).
8. **fp16 merge overflow**: `merge_and_normalize_info` now normalizes in fp32. `BasicProjector`
   (no `fast_jl`) writes fp16 chunk rows with norms ~5e5 > fp16 max (65504); `F.normalize`
   then produced `inf` denominators and silently zeroed the rows of `all_orig.pt`
   (99.7-100% of train dumps, 31-45% of mmlu val — which made LESS's max-over-subtasks score
   tie at exactly 0.0 and select identical subsets for all target tasks). Upstream LESS avoids
   this via `CudaProjector`'s fp32 outputs. Fixed dumps by re-merging from intact `grads-*.pt`
   chunks (no GPU re-run needed).
9. **eval without open-instruct**: `evaluation/eval/utils.py` top-level-imported
   `open_instruct.finetune.encode_with_prompt_completion_format`; the import now falls back to
   a verbatim local copy of that function (identical to LESS's own
   `less/data_selection/get_training_dataset.py`).
10. **eval without vllm**: `evaluation/eval/{tydiqa,bbh}/run_eval.py` top-level-imported
    `vllm`, which is only used behind `--use_vllm` — a flag none of `eval_*.sh` passes.
    The import is now `try/except ImportError: vllm = None`.
    (With 9-10 the eval stack needs only our venv plus `evaluate` + `openai`; the SQuAD
    metric is prefetched into `$HF_HOME` once for the offline compute nodes.)

## Measured settings (Pythia-410M, warmup 5%, 4 epochs, effective batch 128)

- ~423 steps/epoch → trajectory checkpoints **{423, 846, 1269, 1692}**
  (LESS's shipped defaults 105/211/317/420 belong to their setup; `scripts/llm/pipeline.phase0.mila.json`
  carries the corrected list with uniform checkpoint weights).
- Epoch-end saves come from `--save_strategy epoch` (set in `base_training_args.sh`).

## Dependency notes

- `llm` extra pins `transformers>=4.40,<5` (v5 removed `warmup_ratio`/`overwrite_output_dir`
  args LESS passes).
- No deepspeed/open-instruct needed for warmup + gradient stages (`torchrun` + `python -m
  less.data_selection.get_info` only). For the final train/eval stages, open-instruct stays
  uninstalled too: its two uses (finetune encode import, vllm generation) are covered by
  vendor patches 9-10. Eval *data* is what open-instruct's layout defines — download it once
  with `scripts/llm/fetch_eval_data.sh` (MMLU + BBH + cot-prompts + TyDiQA originals).
