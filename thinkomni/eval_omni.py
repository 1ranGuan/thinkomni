import json
import traceback

from tqdm import tqdm

import requests
from concurrent.futures import ThreadPoolExecutor

from utils.model_parser import (
    retry_with_backoff
)

from utils.omni_utils import (
    get_audio_extract_ICE,
    get_audio_score_ICE,
    get_multimodal_extract_ICE,
    get_multimodal_score_ICE
)


class GenAI:
    def __init__(self, headers=None, model="gpt-4o",url=None, system_prompt=None):
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
        response = requests.post(self.url, headers=self.headers, json=payload)
        try:
            body = response.json()
        except ValueError:
            print(
                f"Judge API returned non-JSON response. status={response.status_code}, "
                f"text={response.text[:500]}"
            )
            return None

        if response.status_code != 200:
            print(f"Judge API error status={response.status_code}, body={body}")

        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list) or not results:
            print(f"Judge API unexpected body format: {body}")
            return None

        return results[0]
###########

def load_data_from_json(file_path):
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

class MMAUEvaluator:
    def __init__(self, data_dir, max_workers=5, api_url=None, api_headers=None, model=None):
        self.data_dir = data_dir
        self.data = load_data_from_json(data_dir)

        self.extract_demo_examples = get_audio_extract_ICE()
        self.score_demo_examples = get_audio_score_ICE()

        selected_url = api_url 
        selected_headers = api_headers 
        selected_model = model

        self.model = GenAI(headers=selected_headers, model=selected_model, url=selected_url, system_prompt=None)

        self.max_workers = max_workers

    def build_extract_prompt(self, prediction):
        task_description = (
            "I am providing you a response from a model to an audio understanding problem, termed 'Model Response'. "
            "You should extract the answer from the response as 'Extracted Answer'. "
            "Directly output the extracted answer with no explanation.\n\n"
        )
        demo_prompt = task_description + "Below are some examples of how to extract the answer:\n"
        for example in self.extract_demo_examples:
            demo_prompt += example + '\n\n'
        target_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
        full_prompt = demo_prompt + target_prompt

        return full_prompt
    
    def build_score_prompt(self, question, extract, answer):
        task_description = (
            "Below are two answers to an audio understanding problem. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.\n"
            "Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent.\n"
            "If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n"
        )
        demo_prompt = task_description + "Below are some examples of how to judge the consistency of two answers:\n"
        for example in self.score_demo_examples:
            demo_prompt += example + '\n\n'
        target_prompt = (
            "Please output the judgement score directly with no explanation.\n"
            f"[Question]: {question}\n"
            f"[Standard Answer]: {answer}\n"
            f"[Model_answer]: {extract}\n"
            "Judgement: "
        )
        full_prompt = demo_prompt + target_prompt

        return full_prompt

    def evaluate_answer(self, prediction):
        extract_prompt = self.build_extract_prompt(prediction)
        extracted_answer = retry_with_backoff(
            self.model.generate,
            max_retries=5,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0
        )
        
        return extracted_answer

    def evaluate_prediction(self, prediction, answer, question):
        extract_prompt = self.build_extract_prompt(prediction)
        extracted_answer = retry_with_backoff(
            self.model.generate,
            max_retries=5,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0
        )
        
        score_prompt = self.build_score_prompt(question, extracted_answer, answer)

        raw_response = retry_with_backoff(
            self.model.generate,
            max_retries=5,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0
        )

        if not isinstance(raw_response, str):
            raise TypeError(
                "Judge scoring response is not a string. "
                f"type={type(raw_response).__name__}, value={repr(raw_response)[:200]}, "
                f"question={repr(question)[:200]}"
            )

        response_text = raw_response.strip()

        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0

    def process_outputs(self):
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, output in enumerate(self.data):
                abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                prediction = output['reasoning']
                question = output['instruction']
                parsed_correct = output.get('correct', None)

                # Keep the parsed result when it is already correct.
                if parsed_correct == 1:
                    eval_result = {
                        "question": output['instruction'],
                        "gt_answer": output['response'],
                        "prediction": prediction,
                        "accuracy": 1,
                        "correct": True,
                    }
                    results.append(eval_result)
                    continue

                if output['response'] != "":
                    gt_answer = output['response']
                    future = executor.submit(
                        self.evaluate_prediction, 
                        prediction, 
                        gt_answer, 
                        question
                    )
                    futures.append((future, i, prediction, output))
                else:
                    future = executor.submit(
                        self.evaluate_answer,
                        prediction,
                    )
                    futures.append((future, i, prediction, output))

            for future, i, prediction, output in tqdm(futures, desc="Evaluating predictions"):
                try:
                    accuracy = future.result()
                    if accuracy == 0 or accuracy == 1:    
                        eval_result = {
                            "question": output['instruction'],
                            "gt_answer": output['response'],
                            "prediction": prediction,
                            "accuracy": accuracy,
                            "correct": accuracy > 0,
                        }
                        results.append(eval_result)
                    else:
                        eval_result = {
                            "id": output.get("id", i),
                            "question": output['instruction'],
                            "gt_answer": output['response'],
                            "prediction": prediction,
                            "choices": output.get('choices', []),
                            "extracted_answer": accuracy,
                        }
                        results.append(eval_result)
                except Exception as e:
                    print(f"Error evaluating prediction {i}: {e!r}")
                    print(f"Question: {repr(output.get('instruction', ''))[:300]}")
                    print(f"GT Answer: {repr(output.get('response', ''))[:120]}")
                    print(f"Prediction: {repr(prediction)[:300]}")
                    print(traceback.format_exc())
        return results
    
    def evaluation(self):
        results = self.process_outputs()
        if not results:
            accuracy = 0.0
        else:
            accuracy = sum(1 for r in results if r.get("correct", 0)) / len(results)
        metrics = {"accuracy": accuracy}
        print(metrics)
        # save results
        save_dir = self.data_dir.replace(".jsonl", "_gpt_accuracy.json")
        result_save_dir = self.data_dir.replace(".jsonl", "_gpt_result.jsonl")
        with open(result_save_dir, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        with open(save_dir, 'w') as f:
            json.dump(metrics, f)


class OmniBenchEvaluator(MMAUEvaluator):
    def __init__(self, data_dir, max_workers=5, api_url=None, api_headers=None, model=None):
        super().__init__(
            data_dir,
            max_workers,
            api_url=api_url,
            api_headers=api_headers,
            model=model,
        )

        self.extract_demo_examples = get_multimodal_extract_ICE()
        self.score_demo_examples = get_multimodal_score_ICE()

    def build_extract_prompt(self, prediction):
        task_description = (
            "I am providing you a response from a model to a multimodal reasoning problem, termed 'Model Response'. "
            "You should extract the answer from the response as 'Extracted Answer'. "
            "Directly output the extracted answer with no explanation.\n\n"
        )
        demo_prompt = task_description + "Below are some examples of how to extract the answer:\n"
        for example in self.extract_demo_examples:
            demo_prompt += example + '\n\n'
        target_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
        full_prompt = demo_prompt + target_prompt

        return full_prompt
    
    def build_score_prompt(self, question, extract, answer):
        task_description = (
            "Below are two answers to a multimodal reasoning problem. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.\n"
            "Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent.\n"
            "If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n"
        )
        demo_prompt = task_description + "Below are some examples of how to judge the consistency of two answers:\n"
        for example in self.score_demo_examples:
            demo_prompt += example + '\n\n'
        target_prompt = (
            "Please output the judgement score directly with no explanation.\n"
            f"[Question]: {question}\n"
            f"[Standard Answer]: {answer}\n"
            f"[Model_answer]: {extract}\n"
            "Judgement: "
        )
        full_prompt = demo_prompt + target_prompt

        return full_prompt
    
    def evaluate_prediction(self, prediction, answer, question):
        extract_prompt = self.build_extract_prompt(prediction)
        extracted_answer = retry_with_backoff(
            self.model.generate,
            max_retries=5,
            initial_delay=2,
            prompt=extract_prompt,
            temperature=0.0
        )
        
        score_prompt = self.build_score_prompt(question, extracted_answer, answer)

        raw_response = retry_with_backoff(
            self.model.generate,
            max_retries=5,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0
        )

        if not isinstance(raw_response, str):
            raise TypeError(
                "Judge scoring response is not a string. "
                f"type={type(raw_response).__name__}, value={repr(raw_response)[:200]}, "
                f"question={repr(question)[:200]}"
            )

        response_text = raw_response.strip()

        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0

    def process_outputs(self):
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, output in enumerate(self.data):
                abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                rep2index = output['choices'].index(output['response'])
                parsed_correct = output['correct']
                prediction = output['reasoning']
                question = output['instruction']
                gt_answer = abc[rep2index] if rep2index < len(abc) else output['response']
                if parsed_correct == 1:
                    eval_result = {
                        "question": output['instruction'],
                        "gt_answer": output['response'],
                        "prediction": prediction,
                        "accuracy": 1,
                        "correct": True,
                    }
                    results.append(eval_result)
                    continue

                future = executor.submit(
                    self.evaluate_prediction, 
                    prediction, 
                    gt_answer, 
                    question
                )
                futures.append((future, i, prediction, output))

            for future, i, prediction, output in tqdm(futures, desc="Evaluating predictions"):
                try:
                    accuracy = future.result()
                    eval_result = {
                        "question": output['instruction'],
                        "gt_answer": output['response'],
                        "prediction": prediction,
                        "accuracy": accuracy,
                        "correct": accuracy > 0,
                    }
                    results.append(eval_result)
                except Exception as e:
                    print(f"Error evaluating prediction {i}: {e!r}")
                    print(f"Question: {repr(output.get('instruction', ''))[:300]}")
                    print(f"GT Answer: {repr(output.get('response', ''))[:120]}")
                    print(f"Prediction: {repr(prediction)[:300]}")
                    print(traceback.format_exc())
        return results
    
    def evaluation(self):
        results = self.process_outputs()
        if not results:
            accuracy = 0.0
        else:
            accuracy = sum(1 for r in results if r["correct"]) / len(results)
        metrics = {"accuracy": accuracy}
        print(metrics)
        # save results
        save_dir = self.data_dir.replace(".jsonl", "_gpt_accuracy.json")
        result_save_dir = self.data_dir.replace(".jsonl", "_gpt_result.jsonl")
        with open(result_save_dir, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        with open(save_dir, 'w') as f:
            json.dump(metrics, f)

