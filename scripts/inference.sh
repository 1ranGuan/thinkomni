#!/bin/bash
THREAD_NUM=4

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)
TOTAL_GPUS=${#AVAILABLE_GPUS[@]}

GPUS_PER_PROCESS=$((TOTAL_GPUS / THREAD_NUM))
REMAINING_GPUS=$((TOTAL_GPUS % THREAD_NUM))

echo "Total GPUs: ${TOTAL_GPUS}"
echo "Processes: ${THREAD_NUM}"
echo "GPUs per process: ${GPUS_PER_PROCESS}"
echo "Remaining GPUs: ${REMAINING_GPUS}"
echo "----------------------------------------"

for rank in $(seq 0 $((THREAD_NUM-1))); do
  start_idx=$((rank * GPUS_PER_PROCESS))
  
  if [ $rank -lt $REMAINING_GPUS ]; then
    start_idx=$((start_idx + rank))
    current_gpus_count=$((GPUS_PER_PROCESS + 1))
  else
    start_idx=$((start_idx + REMAINING_GPUS))
    current_gpus_count=$GPUS_PER_PROCESS
  fi
  
  gpu_list=""
  for i in $(seq 0 $((current_gpus_count-1))); do
    gpu_idx=$((start_idx + i))
    if [ $i -eq 0 ]; then
      gpu_list="${AVAILABLE_GPUS[$gpu_idx]}"
    else
      gpu_list="${gpu_list},${AVAILABLE_GPUS[$gpu_idx]}"
    fi
  done
  
  echo "Process ${rank}: GPUs ${gpu_list}"
  CUDA_VISIBLE_DEVICES=${gpu_list} python thinkomni/run_thinkomni.py \
    --base_model_path Qwen/Qwen2.5-Omni-7B \
    --positive_model_path Qwen/Qwen3-8B \
    --negative_model_path Qwen/Qwen2.5-Omni-7B \
    --datasets mmau-testmini-bboxed omnibench-test-bboxed  daily_omni-test-bboxed mathvista-testmini-vlrethinker mathverse-testmini-vlrethinker mathvision-test-vlrethinker \
    --temperature 0.6 \
    --top_p 0.95 \
    --repetition_penalty 1.0 \
    --batch_size 4 \
    --output_dir output/ \
    --use_parallel \
    --parallel_size ${THREAD_NUM} \
    --parallel_rank ${rank} &
done

echo "----------------------------------------"
echo "All processes started!"
wait
echo "All processes completed!"