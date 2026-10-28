"""Reproducible orchestration of the LESS stage pipeline + our selection.

This module only *builds* the command sequence; it never imports LESS and only
shells out when :func:`run_stages` is called (not at import). That keeps it
unit-testable with a dry run and safe to import anywhere.

Pipeline (per seed) — LESS stages run from the ``vendor/LESS`` checkout, our
stages run from the repo root:

    warmup -> train grads -> validation grads -> ingest(ours) -> select(ours)
           -> train selected -> eval

``build_stages`` returns an ordered list of :class:`Stage` (argv + cwd + env +
``requires``), so a driver can dry-run, resume, or schedule on a cluster.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Stage:
    name: str
    argv: List[str]
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineConfig:
    vendor: str                      # path to vendor/LESS
    repo_root: str                   # this repo (for scripts/llm/*.py)
    data_dir: str                    # LESS processed-data root (has train/processed/...)
    eval_data_dir: str               # open-instruct eval data root
    model: str                       # target model (path or HF id)
    selection_model: Optional[str] = None   # LESS-T selector model (defaults to model)
    python: str = "python"

    grads_dir: str = "GRADS"
    store_dir: str = "llm_store"
    runs_dir: str = "runs"
    less_out: str = "out"            # where LESS warmup/final checkpoints land

    train_file_names: List[str] = field(default_factory=lambda: ["flan_v2", "cot", "dolly", "oasst1"])
    tasks: List[str] = field(default_factory=lambda: ["mmlu", "tydiqa", "bbh"])
    methods: List[str] = field(default_factory=lambda: ["swap", "less", "random"])
    budgets: List[float] = field(default_factory=lambda: [0.05])
    seeds: List[int] = field(default_factory=lambda: [0])

    warmup_pct: float = 0.05
    dim: int = 8192
    grad_type: str = "adam"
    ckpts: List[int] = field(default_factory=lambda: [105, 211, 317, 420])
    checkpoint_weights: List[float] = field(default_factory=lambda: [1.0])

    # templates (LESS-relative; overridable for other setups)
    train_file_template: str = "{data_dir}/train/processed/{name}/{name}_data.jsonl"
    rule: str = "ranked"

    @property
    def selector_model(self) -> str:
        return self.selection_model or self.model

    def warmup_job(self, seed: int) -> str:
        tag = os.path.basename(self.selector_model.rstrip("/"))
        return f"{tag}-warmup-p{self.warmup_pct}-seed{seed}"

    def ckpt_path(self, seed: int, ckpt: int) -> str:
        return os.path.join(self.vendor, self.less_out, self.warmup_job(seed), f"checkpoint-{ckpt}")

    def _grad_out(self, name: str, ckpt: int, suffix: str) -> str:
        return os.path.join(self.grads_dir, f"{name}-ckpt{ckpt}-{suffix}")


def _selection_name(cfg: PipelineConfig, method: str, task: str, budget: float) -> str:
    if method == "swap":
        return f"swap-{cfg.rule}-{task}-p{budget}"
    return f"{method}-{task}-p{budget}" if method != "random" else f"random-p{budget}"


def build_stages(cfg: PipelineConfig) -> List[Stage]:  # noqa: C901 - explicit linear plan
    """Build the ordered stage list for the whole configuration (no execution)."""
    stages: List[Stage] = []
    dims = str(cfg.dim)

    for seed in cfg.seeds:
        warm = cfg.warmup_job(seed)
        warm_stage = f"warmup:seed{seed}"
        stages.append(Stage(
            warm_stage,
            ["bash", os.path.join(cfg.vendor, "less/scripts/train/warmup_lora_train.sh"),
             cfg.data_dir, cfg.selector_model, str(cfg.warmup_pct), str(seed), warm],
            cfg.vendor))

        # train gradients (per checkpoint, per train file)
        grads_done: List[str] = []
        for ckpt in cfg.ckpts:
            for tfn in cfg.train_file_names:
                tf = cfg.train_file_template.format(data_dir=cfg.data_dir, name=tfn)
                s = f"train_grads:seed{seed}:ckpt{ckpt}:{tfn}"
                stages.append(Stage(s, [
                    "bash", os.path.join(cfg.vendor, "less/scripts/get_info/grad/get_train_lora_grads.sh"),
                    tf, cfg.ckpt_path(seed, ckpt), cfg._grad_out(tfn, ckpt, cfg.grad_type),
                    dims, cfg.grad_type], cfg.vendor, requires=[warm_stage]))
                grads_done.append(s)

        # validation gradients (per checkpoint, per task)
        val_done: List[str] = []
        for ckpt in cfg.ckpts:
            for task in cfg.tasks:
                s = f"val_grads:seed{seed}:ckpt{ckpt}:{task}"
                stages.append(Stage(s, [
                    "bash", os.path.join(cfg.vendor, "less/scripts/get_info/grad/get_eval_lora_grads.sh"),
                    task, cfg.data_dir, cfg.ckpt_path(seed, ckpt),
                    cfg._grad_out(task, ckpt, "sgd"), dims], cfg.vendor, requires=[warm_stage]))
                val_done.append(s)

        # ingest LESS dumps into our FeatureStore
        store = f"store:seed{seed}"
        pool_args: List[str] = []
        for tfn in cfg.train_file_names:
            pool_args.append(f"{tfn}={cfg.train_file_template.format(data_dir=cfg.data_dir, name=tfn)}")
        stages.append(Stage(store, [
            cfg.python, os.path.join(cfg.repo_root, "scripts/llm/build_datastore.py"),
            "--out", cfg.store_dir, "--from-less", "--dim", str(cfg.dim),
            "--ckpts", *[str(c) for c in cfg.ckpts],
            "--checkpoint-weights", *[str(w) for w in cfg.checkpoint_weights],
            "--train-file-names", *cfg.train_file_names,
            "--train-grad-template", os.path.join(cfg.grads_dir, "{name}-ckpt{ckpt}-%s" % cfg.grad_type, "dim{dim}"),
            "--val-grad-template", os.path.join(cfg.grads_dir, "{task}-ckpt{ckpt}-sgd", "dim{dim}"),
            "--tasks", *cfg.tasks, "--pool-files", *pool_args],
            cfg.repo_root, requires=grads_done + val_done))

        # selection (swap + baselines) then training then eval
        for task in cfg.tasks:
            for budget in cfg.budgets:
                for method in cfg.methods:
                    sel = f"select:{method}:{task}:p{budget}:seed{seed}"
                    if method in ("less", "random"):
                        argv = [cfg.python, os.path.join(cfg.repo_root, "scripts/llm/select_baselines.py"),
                                "--store", cfg.store_dir, "--out", os.path.join(cfg.runs_dir, sel),
                                "--method", method, "--fraction", str(budget)]
                        if method == "less":
                            argv += ["--task", task]
                    else:  # swap
                        argv = [cfg.python, os.path.join(cfg.repo_root, "scripts/llm/select_swap.py"),
                                "--store", cfg.store_dir, "--out", os.path.join(cfg.runs_dir, sel),
                                "--setting", "targeted", "--task", task, "--fraction", str(budget),
                                "--rule", cfg.rule]
                    stages.append(Stage(sel, argv, cfg.repo_root, requires=[store]))

                    job = f"{os.path.basename(cfg.model.rstrip('/'))}-{_selection_name(cfg, method, task, budget)}-seed{seed}"
                    train = f"train:{job}"
                    selected = os.path.join(cfg.runs_dir, sel, "selected.jsonl")
                    stages.append(Stage(train, [
                        "bash", os.path.join(cfg.vendor, "less/scripts/train/lora_train.sh"),
                        selected, cfg.model, job], cfg.vendor, requires=[sel]))

                    for eval_task in cfg.tasks:
                        s = f"eval:{eval_task}:{job}"
                        script = f"eval_{eval_task}.sh"
                        inner = (f"cd {os.path.join(cfg.vendor, 'evaluation')} && "
                                 f"source {script} && eval_{eval_task} {cfg.vendor}/{cfg.less_out}/{job} && "
                                 f"extract_{eval_task} {cfg.vendor}/{cfg.less_out}/{job}")
                        stages.append(Stage(s, ["bash", "-lc", inner], cfg.vendor,
                                            env={"DATA_DIR": cfg.eval_data_dir}, requires=[train]))
    return stages


def run_stages(stages: List[Stage], *, dry_run: bool = False, only: Optional[List[str]] = None,
               echo=print) -> List[Stage]:
    """Execute (or, with ``dry_run``, print) stages. Returns the stages it acted on."""
    acted = []
    prefix = "" if not only else "|".join(only)
    for st in stages:
        if only and not any(pat in st.name for pat in only):
            continue
        acted.append(st)
        env_note = f"  env={st.env}" if st.env else ""
        if dry_run:
            echo(f"[dry] {st.name} (cwd={st.cwd}){env_note}\n      {' '.join(st.argv)}")
        else:
            echo(f"[run] {st.name}")
            env = {**os.environ, **st.env}
            subprocess.run(st.argv, cwd=st.cwd, env=env, check=True)
    if prefix == "":
        pass
    return acted


def config_from_json(path: str) -> PipelineConfig:
    with open(path) as fh:
        raw = json.load(fh)
    return PipelineConfig(**raw)
