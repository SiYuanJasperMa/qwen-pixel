#!/bin/bash
# Prune old LoRA checkpoints to protect the disk quota.
# Keeps: latest ckpt per run, ckpts with an eval summary, and PROTECTed ckpts.
# Deletes: other ckpts older than 3 hours (safe: training never re-reads old ckpts).
set -u
cd /mnt/bn/embodied-lf3/masiyuan/qwen_pixel
LOG=outputs/prune_ckpts.log
# ckpts that eval queues still need (run:steps)
PROTECT="sft_all_v13:600,1200,1800,2400
sft_v13_warmstart_v2:500,1000,1500
sft_all_v13_r32_v2:800,1600,2400
sft_all_v12_cont:300,500
sft_mixed_v3_cont:200,400
sft_ti_only_v1:300,600
sft_all_v14:600,1200,1800,2400
sft_all_v14_cont:600,1200,1800,2300
sft_v13_long_cont:2400,4200
sft_all_v12_fresh:1200,2400
sft_magicbrush_27b_oneshot_v3_cont:1200
sft_magicbrush_27b_v4:900
sft_magicbrush_rft_v1_cont2:200"

echo "[$(date)] prune start" >> "$LOG"
for train in outputs/*/train; do
  [ -d "$train" ] || continue
  run=$(basename "$(dirname "$train")")
  latest=$(ls -d "$train"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  protect_steps=$(echo "$PROTECT" | awk -F: -v r="$run" '$1==r{print $2}')
  for ck in "$train"/checkpoint-*; do
    [ -d "$ck" ] || continue
    step=$(basename "$ck" | sed 's/checkpoint-//')
    [ "$step" = "$latest" ] && continue
    case ",$protect_steps," in *",$step,"*) continue;; esac
    [ -f "$(dirname "$train")/development_${step}/summary.json" ] && continue
    # skip anything written in the last 3 hours
    if [ -n "$(find "$ck" -maxdepth 0 -mmin -90 2>/dev/null)" ]; then continue; fi
    echo "[$(date)] rm $ck" >> "$LOG"
    rm -rf "$ck"
  done
done
echo "[$(date)] prune done" >> "$LOG"
