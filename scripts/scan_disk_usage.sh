#!/bin/bash
# Full disk usage scan for /mnt/bn/embodied-lf3/masiyuan, written to a report file.
cd /mnt/bn/embodied-lf3/masiyuan
OUT=/mnt/bn/embodied-lf3/masiyuan/qwen_pixel/outputs/disk_scan.txt
{
  echo "scan started: $(date)"
  du -sh --max-depth=1 . 2>/dev/null | sort -rh | head -80
  echo "scan finished: $(date)"
} > "$OUT" 2>&1
