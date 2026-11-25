#!/bin/bash
# Kill local CPU RFT scorer and relaunch on a free GPU on the worker.
pkill -9 -f "score_rft_candidates.py" 2>/dev/null
sleep 1
echo killed_local
