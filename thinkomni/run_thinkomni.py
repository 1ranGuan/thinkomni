import fcntl  # For file locking on Unix systems
from contextlib import contextmanager
import os
import json
import torch
import argparse
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
)
from math_common_utils import (
    Timer, 
    DatasetType, 
    get_dataset_config, 
    process_dataset_to_message,
    load_image_dataset,
)

from inference_utils import (
    ProxyThinkerWrapper, generate_completions
)
import warnings
import logging as py_logging

warnings.filterwarnings('ignore')
py_logging.disable(py_logging.WARNING)
from logger import setup_logger

logger = None

def parse_arguments():
    global logger 
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default=None)
    parser.add_argument('--positive_model_path', type=str, default=None)
    parser.add_argument('--negative_model_path', type=str, default=None)
    parser.add_argument('--cd_decoding_alpha', type=float, default=None)
    parser.add_argument('--cd_put_on_diff_gpus', action='store_true', default=False)
    parser.add_argument('--output_dir', default="cd_native_results/", type=str)
    parser.add_argument('--log_dir', default="logs/", type=str)
    parser.add_argument('--dataset', default="mathvista", type=str)
    parser.add_argument('--split', type=str, choices=["testmini", "test"], default="testmini")
    parser.add_argument('--datasets', nargs='+', help="List of dataset configs in format: dataset-split-prompt_type")
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, choices=["answer", "bboxed", "none"], default="bboxed")
    parser.add_argument('--num_samples', type=int, default=None) 
    parser.add_argument('--skip_multi_choice', action='store_true', default=False)
    
    parser.add_argument('--max_model_len', default=65536, type=int)  # Max model length
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_p', type=float, default=0.001)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--min_pixels', type=int, default=None)
    parser.add_argument('--max_pixels', type=int, default=None)
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_auto', action='store_true', default=False)
    parser.add_argument('--n', type=int, default=1)

    # parallel args
    parser.add_argument('--use_parallel', action='store_true', default=False)
    parser.add_argument('--parallel_size', type=int, default=1)
    parser.add_argument('--parallel_rank', type=int, default=0)
    
    args = parser.parse_args()
    logger = setup_logger(
        log_dir=args.log_dir,
        log_file=f"native_{args.base_model_path.split('/')[-1]}-{args.dataset}-{args.split}.log",
    )
    logger.info("Arguments: %s", args)
    return args

def get_device_map(model_path):
    if '72b' in model_path.lower() or '32b' in model_path.lower():
        # For large models, use all GPUs
        return "auto"
    else:
        # For smaller models, use the first GPU
        return {"": 0}

@contextmanager
def file_lock(file_path):
    """Context manager for file locking"""
    lock_file = file_path + '.lock'
    with open(lock_file, 'w') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def safe_append_to_jsonl(file_path, data_list, parallel_rank=0):
    """Safely append data to JSONL file with file locking"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with file_lock(file_path):
        with open(file_path, 'a', encoding='utf-8') as f:
            for item in data_list:
                # Add process rank info for debugging if needed
                if parallel_rank is not None:
                    item['_process_rank'] = parallel_rank
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush()  # Ensure data is written immediately
    
def process_single_dataset(args, cd_model, processor, dataset_type, split, prompt_type, 
                           stop_id_seqs, banned_begin_ids, model_suffix=None,
                           use_parallel=False, parallel_size=4, parallel_rank=0, positive_processor=None):
    dataset_config = get_dataset_config(dataset_type, split=split)
    
    print(f"Loading dataset {dataset_config}...")
    dataset = load_image_dataset(dataset_config)
    if use_parallel:
        # Split the dataset into chunks for parallel processing
        chunk_size = len(dataset) // parallel_size
        start_idx = parallel_rank * chunk_size
        end_idx = (parallel_rank + 1) * chunk_size if parallel_rank < parallel_size - 1 else len(dataset)
        dataset = dataset[start_idx:end_idx]
    
    # Save the current prompt_type to process the dataset correctly
    original_prompt_type = args.prompt_type
    args.prompt_type = prompt_type
    args.dataset_type = dataset_type
    
    resp_messages = process_dataset_to_message(dataset, args)
    batch_size =  1 if dataset_type==DatasetType.DAILY_OMNI else args.batch_size

    with Timer(f"Huggingface Generation for {dataset_type.value}-{split}-{prompt_type}") as t:
        generations = generate_completions(
            cd_model,
            processor,
            resp_messages,
            postive_processor=positive_processor,
            positive_messages=None, 
            negative_messages=None,
            batch_size=batch_size,
            stop_id_seqs=stop_id_seqs,
            banned_id_seqs=None,
            banned_begin_ids=None, 
            disable_tqdm=False,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k, 
            repetition_penalty=args.repetition_penalty, 
            max_new_tokens=args.max_new_tokens,
            do_sample=True, 
            use_cache=True,
            is_instruct=True,
            prompt_type=args.prompt_type,
        )
        
    ret_list = []
    for i, generation in enumerate(generations):
        ret = {
            'prompt': resp_messages[i][0]['content'][-1]['text'],
            'generated_text': generation,
        }
        ret_list.append(ret)
        
     # Define result path
    result_path = os.path.join(args.output_dir, 
                             f"{dataset_type.value}-{split}", 
                             f"{model_suffix}_{prompt_type}{'_' + args.tags if args.tags is not None else ''}.jsonl")
    
    # Use safe append method for multi-process writing
    if use_parallel:
        safe_append_to_jsonl(result_path, ret_list, parallel_rank)
        print(f"Process {parallel_rank}: Results appended to {result_path}")
    else:
        # Single process - write normally
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            for item in ret_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Results saved to {result_path}")
    
    args.precomputed_json = result_path
    
    # Restore the original prompt_type
    args.prompt_type = original_prompt_type
            
    return t.elapsed_time
    
    
if __name__ == "__main__":
    args = parse_arguments()
    real_model_name = args.base_model_path.split("/")[-1]
    assert real_model_name, f"Expect a non-empty model name, got {real_model_name}"
    
    if args.positive_model_path:
        positive_model_name = args.positive_model_path.split("/")[-1]
        assert positive_model_name, f"Expect a non-empty positive model name, got {positive_model_name}"
        model_suffix = f"{real_model_name}-{positive_model_name}"
    else:
        model_suffix = real_model_name

    print("Loading models...")
    
    base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=get_device_map(args.base_model_path)
        ).eval()
    
    positive_model, negative_model = None, None
    positive_tokenizer = None
    positive_processor = None
    if args.positive_model_path is not None:
        positive_model = AutoModelForCausalLM.from_pretrained(
            args.positive_model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map={"": 1},
        ).eval()
        positive_tokenizer = AutoTokenizer.from_pretrained(args.positive_model_path)
        positive_processor = AutoProcessor.from_pretrained(args.positive_model_path)

    
    if args.negative_model_path is not None:
        negative_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.negative_model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map={"": 1},
        ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model_path)
    
    cd_model = ProxyThinkerWrapper(
        base_model=base_model,
        preprocessor=processor,
        positive_model=positive_model,
        negative_model=negative_model,
        positive_tokenizer=positive_tokenizer,
    )
    stop_id_seqs = [[151645]]
    
    banned_begin_ids = base_model.generation_config.eos_token_id
    if isinstance(banned_begin_ids, int):
        banned_begin_ids = [banned_begin_ids]
    total_time = 0
    
    for dataset_config in args.datasets:
        parts = dataset_config.split('-')
        if len(parts) == 3:
            dataset_name, split, prompt_type = parts
        elif len(parts) == 2:
            dataset_name, split = parts
            prompt_type = args.prompt_type
        else:
            raise ValueError(f"Invalid dataset config format: {dataset_config}")
        
        dataset_type = DatasetType(dataset_name)
        elapsed_time = process_single_dataset(
            args, cd_model, processor, dataset_type, split, prompt_type,
            stop_id_seqs=stop_id_seqs, banned_begin_ids=banned_begin_ids, model_suffix=model_suffix,
            use_parallel=args.use_parallel, parallel_size=args.parallel_size, parallel_rank=args.parallel_rank,
            positive_processor=positive_processor
        )
        total_time += elapsed_time
        
    print(f"\nTotal processing time: {total_time:.2f} seconds")