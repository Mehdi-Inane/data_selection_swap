#!/usr/bin/env bash
# Download the open-instruct-style evaluation data LESS's eval scripts expect
# (run once per machine, on a node with internet access; the compute nodes are
# offline and read this tree through $SCRATCH).
#
# Layout produced under $1 (default $SCRATCH/less/DATA/eval):
#   mmlu/              Hendrycks MMLU (test/ dev/ val/ *.csv)   <- eval_mmlu.sh
#   tydiqa/            tydiqa-goldp-v1.1-{dev,train}.json       <- eval_tydiqa.sh
#   bbh/bbh/           BIG-Bench-Hard 27 task *.json            <- eval_bbh.sh
#   bbh/cot-prompts/   few-shot CoT prompts (BIG-Bench-Hard repo) <- eval_bbh.sh
#
# The TyDiQA GCS bucket 403s as of 2026-11; files come from the Internet Archive
# snapshot of the original bucket URLs (byte-identical originals matter: LESS
# samples per-language subsets under a fixed seed).
set -euo pipefail
dest="${1:-/network/scratch/a/ahmedm/less/DATA/eval}"
mkdir -p "$dest"

# --- MMLU (Hendrycks data.tar: test/ dev/ val/ *.csv) ---
if [[ ! -f "$dest/mmlu/test/high_school_biology_test.csv" ]]; then
    tmp=$(mktemp -d)
    curl -sSL --retry 3 -o "$tmp/data.tar" https://people.eecs.berkeley.edu/~hendrycks/data.tar
    tar -xf "$tmp/data.tar" -C "$tmp"          # extracts to data/
    mkdir -p "$dest/mmlu"
    mv "$tmp"/data/* "$dest/mmlu/"
    rm -rf "$tmp"
fi
echo "mmlu: $(ls "$dest/mmlu/test" | wc -l) test files"

# --- BBH (task jsons + cot-prompts, both from the BIG-Bench-Hard repo) ---
if [[ ! -f "$dest/bbh/bbh/temporal_sequences.json" || ! -f "$dest/bbh/cot-prompts/navigate.txt" ]]; then
    tmp=$(mktemp -d)
    curl -sSL --retry 3 -o "$tmp/bbh.tar.gz" \
        https://codeload.github.com/suzgunmirac/BIG-Bench-Hard/tar.gz/refs/heads/main
    mkdir -p "$dest/bbh"
    tar -xzf "$tmp/bbh.tar.gz" -C "$tmp" --wildcards '*/bbh/*.json' '*/cot-prompts/*'
    [[ -d "$dest/bbh/bbh" ]] || mv "$tmp"/BIG-Bench-Hard-main/bbh "$dest/bbh/bbh"
    [[ -d "$dest/bbh/cot-prompts" ]] || mv "$tmp"/BIG-Bench-Hard-main/cot-prompts "$dest/bbh/cot-prompts"
    rm -rf "$tmp"
fi
echo "bbh: $(ls "$dest/bbh/bbh" | wc -l) task files, cot-prompts: $(ls "$dest/bbh/cot-prompts" | wc -l) files"

# --- TyDiQA goldp (original v1.1 dev+train json via Internet Archive) ---
mkdir -p "$dest/tydiqa"
for f in tydiqa-goldp-v1.1-dev.json tydiqa-goldp-v1.1-train.json; do
    if [[ ! -f "$dest/tydiqa/$f" ]]; then
        curl -sSL --retry 3 -o "$dest/tydiqa/$f" \
            "http://web.archive.org/web/20250501223403id_/https://storage.googleapis.com/tydiqa/v1.1/$f"
    fi
done
python3 - "$dest/tydiqa" <<'EOF'
import json, sys, os
for f in ("tydiqa-goldp-v1.1-dev.json", "tydiqa-goldp-v1.1-train.json"):
    d = json.load(open(os.path.join(sys.argv[1], f)))   # fails loudly if not the original nesting
    n = sum(len(qa.get("qas", [])) for a in d["data"] for qa in a["paragraphs"])
    print(f, "questions:", n)
EOF
