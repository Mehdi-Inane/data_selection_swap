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

## Measured settings (Pythia-410M, warmup 5%, 4 epochs, effective batch 128)

- ~423 steps/epoch → trajectory checkpoints **{423, 846, 1269, 1692}**
  (LESS's shipped defaults 105/211/317/420 belong to their setup; `scripts/llm/pipeline.phase0.mila.json`
  carries the corrected list with uniform checkpoint weights).
- Epoch-end saves come from `--save_strategy epoch` (set in `base_training_args.sh`).

## Dependency notes

- `llm` extra pins `transformers>=4.40,<5` (v5 removed `warmup_ratio`/`overwrite_output_dir`
  args LESS passes).
- No deepspeed/open-instruct needed for warmup + gradient stages (`torchrun` + `python -m
  less.data_selection.get_info` only). The final train/eval stages (`lora_train.sh`,
  `evaluation/*.sh`) are the only places open-instruct enters — Phase 0 step 5.
