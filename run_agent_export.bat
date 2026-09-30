@echo off
conda run --no-capture-output -n xnewenv python tools\record_agent.py --model D:\models\Qwen3-1.7B %*
