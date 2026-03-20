#!/usr/bin/env bash

python thinkomni/eval.py \
    --output_dir output \
    --datasets mmau-testmini-bboxed omnibench-test-bboxed daily_omni-test-bboxed mathvista-testmini-vlrethinker mathverse-testmini-vlrethinker mathvision-test-vlrethinker \
    --gpt_api_url YOUR_API_URL \
    --gpt_api_key YOUR_API_KEY \
    --judge_model gemini-2.0-flash
