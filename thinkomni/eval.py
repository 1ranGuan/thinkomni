import os
import json
import argparse
import requests
from logger import setup_logger
from math_common_utils import (
    DatasetType, 
    run_evaluate_math,
    run_evaluate_mmmu,
    run_evaluate_mme,
    get_dataset_config, 
    load_image_dataset,
)
from eval_omni import MMAUEvaluator, OmniBenchEvaluator
from math_judge_utils import run_evaluate_math_with_judge
logger = None

DEFAULT_GPT_API_URL = None
DEFAULT_GPT_API_KEY = None


class GenAI:
    def __init__(self, headers=None, model="gpt-4o", url=None, system_prompt=None):
        self.headers = headers
        self.model = model
        self.url = url
        self.system_prompt = system_prompt

    def generate(self, prompt, temperature=0.0):
        payload = {
            "model": self.model,
            "key_id": 0,
            "system_prompt": self.system_prompt,
            "contents": [
                prompt,
            ]
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        return response.json()

def parse_arguments():
    global logger 
    # native version keeps most of the params the same with vLLM version
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default="output1", type=str)
    parser.add_argument('--log_dir', default="logs/", type=str)
    parser.add_argument('--dataset', default="mathvista", type=str)
    parser.add_argument('--split', type=str, choices=["testmini", "test"], default="testmini")
    parser.add_argument('--datasets', nargs='+', default=["mathvista-testmini-bboxed"], help="List of dataset configs in format: dataset-split-prompt_type")
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, choices=["answer", "bboxed", "none"], default="bboxed")  # ignore this field
    parser.add_argument('--skip_multi_choice', action='store_true', default=False)
    parser.add_argument('--run_gpt_eval_for_omni', action='store_true', default=True,
                        help='Run GPT-based evaluator for MMAU/OmniBench after parsed evaluation.')
    parser.add_argument('--gpt_eval_workers', type=int, default=16,
                        help='Max workers used by GPT evaluator for MMAU/OmniBench.')
    parser.add_argument('--run_gpt_eval_for_math', action='store_true', default=True,
                        help='Run GPT math judge for MathVision/MathVerse in the same eval loop.')
    parser.add_argument('--gpt_api_url', type=str, default=DEFAULT_GPT_API_URL,
                        help='Unified API URL used by math judge and omni judge evaluators.')
    parser.add_argument('--gpt_api_key', type=str, default=DEFAULT_GPT_API_KEY,
                        help='Unified API key used by math judge and omni judge evaluators.')
    parser.add_argument('--judge_model', type=str, default='gemini-2.0-flash',
                        help='Unified model name for all judge evaluators.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    gpt_headers = {
        "Content-Type": "application/json",
        "X-API-Key": args.gpt_api_key,
    }

    for dataset_config in args.datasets:
        print(dataset_config)
        # try:
        parts = dataset_config.split('-')
        if len(parts) == 3:
            dataset_name, split, prompt_type = parts
        elif len(parts) == 2:
            dataset_name, split = parts
            prompt_type = args.prompt_type
        else:
            raise ValueError(f"Invalid dataset config format: {dataset_config}")
        
        dataset_type = DatasetType(dataset_name)
        dataset_config = get_dataset_config(dataset_type, split=split)
        dataset = load_image_dataset(dataset_config, eval = True)

        data_path = os.path.join(args.output_dir, f"{dataset_name}-{split}")
        jsonl_files = [f for f in os.listdir(data_path) if f.endswith('.jsonl') and not (f.startswith('acc') or f.startswith('parsed'))]
        if not jsonl_files:
            logger.error(f"No jsonl files found in {data_path}")
            continue
        jsonl_file = jsonl_files[0]
        ori_msg = {f"{i}": [] for i in range(10)}  # Assuming 8 processes, adjust as needed

        with open(os.path.join(data_path, jsonl_file), 'r') as f:
            for line in f.readlines():
                item = json.loads(line)
                ori_msg[str(item['_process_rank'])].append(item)
        
        ret_list = []
        for key in ori_msg:
            ret_list = ret_list + ori_msg[key]

        args.precomputed_json = os.path.join(data_path, jsonl_file)
        args.dataset_type = dataset_type
        args.prompt_type = prompt_type

        if args.run_gpt_eval_for_math and dataset_name in {"mathvision", "mathverse"}:
            judge_model = GenAI(
                headers=gpt_headers,
                model=args.judge_model,
                url=args.gpt_api_url,
                system_prompt="You are a professional math judge, please evaluate the math question and answer."
            )
            print(f"Running GPT math judge for {dataset_name} on: {args.precomputed_json}")
            run_evaluate_math_with_judge(dataset, ret_list, args, judge_model=judge_model)
        elif "mmmu" in dataset_name:
            run_evaluate_mmmu(dataset, ret_list, args, run_time=None, tokenizer=None)
        elif "mme" in dataset_name:
            run_evaluate_mme(dataset, ret_list, args, run_time=None, tokenizer=None)
        else:
            run_evaluate_math(dataset, ret_list, args, run_time=None, tokenizer=None)

        if args.run_gpt_eval_for_omni and dataset_name in {"mmau", "omnibench"}:
            parsed_json = args.precomputed_json.replace('.jsonl', '_parsed.jsonl')
            if not os.path.exists(parsed_json):
                print(f"Skip GPT eval: parsed file not found: {parsed_json}")
                continue

            evaluator_cls = MMAUEvaluator if dataset_name == "mmau" else OmniBenchEvaluator
            print(f"Running GPT eval for {dataset_name} on: {parsed_json}")
            evaluator = evaluator_cls(
                data_dir=parsed_json,
                max_workers=args.gpt_eval_workers,
                api_url=args.gpt_api_url,
                api_headers=gpt_headers,
                model=args.judge_model,
            )
            evaluator.evaluation()