import csv
import pickle
import string
import copy as cp
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from math_common_utils import (
    DatasetType, 
    MATHVISION_TESTMINI_IDS, 
)
from typing import Iterable
import time
import os.path as osp
import json
from latex2sympy2 import latex2sympy   # mandatory for mathvision evaluation
import logging

logger = logging.getLogger(__name__)
from mathruler.grader import grade_answer
FAIL_MSG = "Failed to obtain answer via API."
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import deque
from threading import Lock
from typing import Iterable

class TokenBucket:
    def __init__(self, capacity=5, refill_rate=5):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = Lock()
    
    def consume(self, tokens=1):
        with self.lock:
            now = time.time()
            # 补充令牌
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, 
                            self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class RateLimitedExecutor:
    def __init__(self, max_workers=4, rate_limit=5, warmup_duration=30):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.rate_limiter = TokenBucket(capacity=rate_limit*2, refill_rate=rate_limit)
        self.startup_time = time.time()
        self.warmup_duration = warmup_duration
        self.pending_tasks = deque()
        self.active_futures = set()
        
    def get_current_rate_limit(self):
        """根据启动时间调整速率限制"""
        elapsed = time.time() - self.startup_time
        if elapsed < self.warmup_duration:
            # 渐进式增加速率限制
            progress = elapsed / self.warmup_duration
            return max(1, int(5 * progress))
        return 5
    
    def submit_with_rate_limit(self, func, *args, **kwargs):
        """提交任务，带速率限制"""
        # 冷启动期间更宽松的处理
        current_limit = self.get_current_rate_limit()
        
        if self.rate_limiter.consume():
            future = self.executor.submit(func, *args, **kwargs)
            self.active_futures.add(future)
            return future
        else:
            # 如果令牌不足，稍等片刻再尝试
            time.sleep(0.05)  # 50ms
            if self.rate_limiter.consume():
                future = self.executor.submit(func, *args, **kwargs)
                self.active_futures.add(future)
                return future
            else:
                # 仍然没有令牌，加入等待队列
                return None
    
    def shutdown(self):
        self.executor.shutdown(wait=True)

def track_progress_rich_with_rate_limit(
        func,
        tasks=tuple(),
        nproc: int = 4,
        rate_limit: int = 5,  # 新增：每秒最大请求数
        warmup_duration: int = 30,  # 新增：预热时间
        save=None,
        keys=None,
        **kwargs) -> list:
    
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError("func must be a callable object")
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f"tasks must be an iterable object, but got {type(tasks)}")
    assert nproc > 0, "nproc must be a positive number"
    
    res = {}
    results = [None for _ in range(len(tasks))]
    
    # 使用自定义的速率限制执行器
    rate_limited_executor = RateLimitedExecutor(
        max_workers=nproc, 
        rate_limit=rate_limit,
        warmup_duration=warmup_duration
    )
    
    try:
        futures = [None for _ in range(len(tasks))]
        submitted = [False for _ in range(len(tasks))]
        pending_queue = deque(range(len(tasks)))
        
        # 初始提交一批任务
        initial_batch = min(rate_limit, len(tasks))
        for i in range(initial_batch):
            idx = pending_queue.popleft()
            inputs = tasks[idx]
            
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            
            if isinstance(inputs, dict):
                future = rate_limited_executor.submit_with_rate_limit(func, **inputs)
            else:
                future = rate_limited_executor.submit_with_rate_limit(func, *inputs)
            
            if future is not None:
                futures[idx] = future
                submitted[idx] = True
            else:
                pending_queue.appendleft(idx)  # 重新加入队列
        
        unfinished = set(i for i in range(len(tasks)) if submitted[i])
        pbar = tqdm(total=len(tasks))
        
        last_submit_time = time.time()
        
        while len(unfinished) > 0 or len(pending_queue) > 0:
            # 检查已完成的任务
            new_finished = set()
            for idx in list(unfinished):
                if futures[idx] is not None and futures[idx].done():
                    try:
                        results[idx] = futures[idx].result()
                        new_finished.add(idx)
                        if keys is not None:
                            res[keys[idx]] = results[idx]
                        rate_limited_executor.active_futures.discard(futures[idx])
                    except Exception as e:
                        print(f"Task {idx} failed with error: {e}")
                        results[idx] = None
                        new_finished.add(idx)
            
            # 更新进度
            if len(new_finished):
                if save is not None:
                    dump(res, save)
                pbar.update(len(new_finished))
                for k in new_finished:
                    unfinished.remove(k)
            
            # 尝试提交新任务（控制提交频率）
            current_time = time.time()
            if pending_queue and (current_time - last_submit_time) >= (1.0 / rate_limit):
                idx = pending_queue.popleft()
                inputs = tasks[idx]
                
                if not isinstance(inputs, (tuple, list, dict)):
                    inputs = (inputs, )
                
                if isinstance(inputs, dict):
                    future = rate_limited_executor.submit_with_rate_limit(func, **inputs)
                else:
                    future = rate_limited_executor.submit_with_rate_limit(func, *inputs)
                
                if future is not None:
                    futures[idx] = future
                    submitted[idx] = True
                    unfinished.add(idx)
                    last_submit_time = current_time
                else:
                    pending_queue.appendleft(idx)  # 重新加入队列
            
            time.sleep(0.05)  # 减少CPU占用
        
        pbar.close()
    
    finally:
        rate_limited_executor.shutdown()
    
    if save is not None:
        dump(res, save)
    
    return results

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)
# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, "wb"))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, "w", encoding="utf-8"), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, "w", encoding="utf8") as fout:
            fout.write("\n".join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine="xlsxwriter")

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding="utf-8", quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep="\t", index=False, encoding="utf-8", quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split(".")[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, "rb"))

    def load_json(pth):
        return json.load(open(pth, "r", encoding="utf-8"))

    def load_jsonl(f):
        lines = open(f, encoding="utf-8").readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == "":
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep="\t")

    # import validators
    # if validators.url(f):
    #     tgt = osp.join(LMUDataRoot(), "files", osp.basename(f))
    #     if not osp.exists(tgt):
    #         download_file(f, tgt)
    #     f = tgt

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split(".")[-1]
    return handlers[suffix](f)

def track_progress_rich(
        func,
        tasks = tuple(),
        nproc: int = 4,
        save=None,
        keys=None,  # indices of tasks
        **kwargs) -> list:

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError("func must be a callable object")
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f"tasks must be an iterable object, but got {type(tasks)}")
    assert nproc > 0, "nproc must be a positive number"
    # res = load(save) if save is not None else {}
    res = {}
    results = [None for _ in range(len(tasks))]

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = []

        for inputs in tasks:
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)

        unfinished = set(range(len(tasks)))
        pbar = tqdm(total=len(unfinished))
        while len(unfinished):
            new_finished = set()
            for idx in unfinished:
                if futures[idx].done():
                    results[idx] = futures[idx].result()
                    new_finished.add(idx)
                    if keys is not None:
                        res[keys[idx]] = results[idx]
            if len(new_finished):
                if save is not None:
                    dump(res, save)
                pbar.update(len(new_finished))
                for k in new_finished:
                    unfinished.remove(k)
            time.sleep(0.1)
        pbar.close()

    if save is not None:
        dump(res, save)
    
    return results   # after auxeval -- containing `log` and `res`


def can_infer_option(answer, choices):
    verbose = os.environ.get("VERBOSE", 0)
    # Choices is a dictionary
    if "Failed to obtain answer via API" in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer"
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = ".()[],:;!*#{}"
    for c in chars:
        answer_mod = answer_mod.replace(c, " ")

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3 and verbose:
                print(f"A might be a quantifier in the string: {answer}.")
                # logger = get_logger("Evaluation")
                # logger.info(f"A might be a quantifier in the string: {answer}.")
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

# ------- general matching utils -------


# ------- math_vista -------

def get_gpt4_mathvista():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_mathvista_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n Directly output the extracted answer with no explanation.\n\n
"""
    question = line["question"]
    prediction = str(line["prediction"]).replace("</think>", " ").replace("<think>", " ")
    prompt = task_description
    examples = get_gpt4_mathvista()
    for example in examples:
        prompt += example + "\n"
    prompt += question + "\n"
    prompt += "Model respone: " + prediction
    prompt += "\n\nExtracted answer:"
    return prompt


def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}

# each line need:
# {
    # "index": 1,
    # "question_type": "multi_choice" / "free_form"
    # "answer_type": "integer" / "float" / "text"
    # "question": "question text"
    # "prediction": "model response"
# }
def post_check(line, prefetch=False):
    res = None
    ans = line["answer"]
    response = line["prediction"] if prefetch else line["res"]  # "prediction" from original model
    # "res" from gpt-4o judge
    try:
        if line["question_type"] == "multi_choice":
            ans = line["answer_option"]
            # choices = list_to_dict(eval(line["choices"]))
            choices = list_to_dict(line["choices"])
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            if line["answer_type"] == "integer":
                res = int(response)
                ans = int(line["answer"])
            elif line["answer_type"] == "float":
                res = float(response)
                ans = float(line["answer"])
            else:
                res = str(res)
                ans = str(ans)
    except ValueError:
        pass

    if res == ans:
        return res if prefetch else True
    else:
        return False
    
def mathvista_auxeval(model, line):  
    # need a threadpool to run this in parallel and get new_results of {"res": "xxx", "log": "xxx"}
    prompt = build_mathvista_gpt4_prompt(line)
    log = ""
    retry = 10
    if post_check(line, prefetch=True):  # directly get response from GPT-4o
        res = post_check(line, prefetch=True)
        return dict(log="Prefetch succeed", res=res, prompt=prompt)
    for i in range(retry):
        prediction = line["prediction"]
        res = model.generate(prompt,temperature=i*0.5)  # gradually increase temperature

        if "code" not in res.keys() or res["code"] != 200:
            time.sleep(6 ** i)  # wait for a while
            log += f"Try {i}: output is {res}, failed to parse.\n"
        else:
            log += "Succeed"
            res = res["results"][0]
            return dict(log=log, res=res, prompt=prompt)
    log += "All 10 retries failed.\n"
    return dict(log=log, res="", prompt=prompt)


# ------- mathverse -------

def get_gpt4_mathverse_extract():
    example_1 = """
1.
Model response: "Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)"
Extracted Answer: (-2, 1)
""" # noqa

    example_2 = """
2.
Model response: "at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.","
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: " at 1 (there"s a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)"
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""" # noqa

    example_4 = """
4.
Model response: "As it stands, I cannot provide the correct option letter because there isn"t enough information to solve for "y"."
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: "Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters."
Extracted answer: 22.3
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)""
Extracted answer: f(x) = -x^2 - 2x + 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_gpt4_mathverse_score():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""" # noqa

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0
""" # noqa

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4]


def build_mathverse_gpt4_extract_prompt(line):
    task_description = """
I am providing you a response from a model to a math problem, termed "Model Response". You should extract the answer from the response as "Extracted Answer". Directly output the extracted answer with no explanation.\n\n
""" # noqa
    prediction = str(line["prediction"])
    demo_prompt = task_description
    examples = get_gpt4_mathverse_extract()
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f"{demo_prompt}7.\n{test_prompt}"

    return full_prompt

def build_mathverse_gpt4_score_prompt(line):
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model"s output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\nDirectly output the result with no explanation!!! e.g. 0 or 1 \n\n
""" # noqa
    question_for_eval = line["question_for_eval"]
    extract = line["extract"]
    answer = line["answer"]
    demo_prompt = task_description
    examples = get_gpt4_mathverse_score()
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
    [Question]: {question_for_eval}
    [Standard Answer]: {answer}
    [Model_answer] : {extract}
    Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt

def post_check_score(line, prefetch=False):
    ans = str(line["answer"]).strip()
    response = str(line["extract"]).strip()

    if response == ans:
        return response if prefetch else True
    else:
        return False
    
def MathVerse_auxeval_extract(model, line):
    prompt = build_mathverse_gpt4_extract_prompt(line)
    log = ""
    retry = 10
    for i in range(retry):
        prediction = line["prediction"]
        res = model.generate(prompt,temperature=i*0.5)  # gradually increase temperature
        if "code" not in res.keys() or res["code"] != 200:
            time.sleep(6 *(i+1))
            log += f"Try {i}: output is {res}, failed to parse.\n"
        else:
            log += "Succeed"
            res = res["results"][0]
            return dict(log_extract=log, extract=res)
    log += "All 10 retries failed.\n"
    return dict(log_extract=log, extract="")


def MathVerse_auxeval_score(model, line):
    prompt = build_mathverse_gpt4_score_prompt(line)
    log = ""
    retry = 10
    if post_check_score(line, prefetch=True):
        res = post_check_score(line, prefetch=True)
        return dict(log_score="Prefetch succeed", score=True)
    for i in range(retry):
        prediction = line["prediction"]
        res = model.generate(prompt, temperature=i*0.5)
        if "code" not in res.keys() or res["code"] != 200 or res["results"][0].strip() not in ["0", "1"]:
            time.sleep(6 *(i+1))
            log += f"Try {i}: output is {prediction}, res is {res}, failed to parse.\n"
        else:
            log += "Succeed"
            res = res["results"][0]
            return dict(log_score=log, score=int(res) == 1)
    log += "All 10 retries failed.\n"
    return dict(log_score=log, score=False)

# ------- math_vision -------
# @timeout_decorator.timeout(30)
def is_equal(asw: str, gt_asw: str) -> bool:
    if not isinstance(asw, str) != str or not isinstance(gt_asw, str):
        print("Warning: input is not string")
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    asw = asw.replace("π", "\pi")
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    return False


def get_gpt4_mathvision():
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]

def build_mathv_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n Directly output the extracted answer with no explanation.\n\n
"""
    question = line["question"]
    prediction = str(line["prediction"])
    prompt = task_description
    examples = get_gpt4_mathvision()
    for example in examples:
        prompt += example + "\n"
    prompt += question + "\n"
    prompt += "Model respone: " + prediction
    prompt += "Extracted answer:"
    return prompt

def post_check_mathvision(line, prefetch=False):
    res = None
    ans = line["answer"]
    response = line["prediction"] if prefetch else line["res"]
    try:
        # if len(eval(line["choices"])) > 0:
        if len(line["choices"]) > 0:
            ans = line["answer"]
            # choices = list_to_dict(eval(line["choices"]))
            choices = list_to_dict(line["choices"])
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            res = str(response)
            ans = str(ans)
    except ValueError:
        pass

    try:
        if is_equal(res, ans):
        # if grade_answer(res, ans):
            return res if prefetch else True
        else:
            return False
    except Exception as err:
        # logging.warning(f"{type(err)}: {err}")
        logger.warning(f"Error in is_equal: {err}")
        return False
    
def MATH_V_auxeval(model, line):
    prompt = build_mathv_gpt4_prompt(line)
    log = ""
    retry = 10
    if post_check_mathvision(line, prefetch=True):
        res = post_check_mathvision(line, prefetch=True)
        if res == line["answer"]:
            log = "Prefetch succeed"
            return dict(log="Prefetch succeed", res=res)
    for i in range(retry):
        prediction = line["prediction"]
        res = model.generate(prompt)

        if "code" not in res.keys() or res["code"] != 200:
            log += f"Try {i}:{res}\n"
            time.sleep(2 * i)  # wait for a while before retry
        else:
            log += "Succeed"
            res = res["results"][0]
            return dict(log=log, res=res)
    log += "All retries failed.\n"
    return dict(log=log, res="")


def run_evaluate_math_with_judge(dataset, final_output, args, run_time=None, 
                                 judge_model=None):
    # parsed_final_output = []
    # dataset specific counter...
    correct_counter = 0
    per_type_total_counter = defaultdict(int)
    per_type_correct_counter = defaultdict(int)
    # error_counter = defaultdict(int)
    
    assert len(dataset) == len(final_output), \
        f"Expect dataset and final_output to have the same length, got {len(dataset)} and {len(final_output)}"
        
    evaluation_lines = []
        
    for i, (input_example, model_output) in enumerate(zip(dataset, final_output)):
        model_response = model_output.get("reasoning", None)
        if model_response is None:
            model_response = model_output.get("generated_text", None)
        # correct_flag = 0
        
        input_example_skills = []
        
        # different dataset need different line-format
        if args.dataset_type == DatasetType.MATHVISTA:
            input_example_skills = input_example["skills"]
            for skill in input_example_skills:
                per_type_total_counter[skill] += 1
            
            # organize the `line` dict
            line = {
                "index": i + 1, 
                "question_type": input_example["question_type"],
                "answer_type": input_example["answer_type"],
                # "answer_option": chr(65 + answer_index),
                "question": input_example["instruction"],
                "answer": input_example["response"],
                "prediction": model_response,
                "skills": input_example_skills,
            }
            if input_example["choices"] is not None:
                answer_index = input_example["choices"].index(
                    input_example["response"]
                )
                line["answer_option"] = chr(65 + answer_index)
                line["choices"] = input_example["choices"]
            
        elif args.dataset_type == DatasetType.MATHVERSE:
            line = {
                "index": i + 1,
                "question_for_eval": input_example["question_for_eval"],
                "prediction": model_response,
                "answer": input_example["response"]
                # "extract" fields will be added in extract step
            }
        
        elif args.dataset_type == DatasetType.MATHVISION:
            if input_example["id"] in MATHVISION_TESTMINI_IDS:
                input_example_skills = ["mathvision_testmini"]
                per_type_total_counter["mathvision_testmini"] += 1
                
            line = {
                "index": i + 1,
                "question": input_example["instruction"],
                "prediction": model_response,
                "answer": input_example["response"],
                "choices": input_example["options"],  # mathvision 
                "skills": input_example_skills,
            }
        
        else:
            raise NotImplementedError(f"Dataset {args.dataset_type} not supported.")
        
        evaluation_lines.append(line)


    # ans = {}
    # if osp.exists(tmp_file):
    #     ans = load(tmp_file)
    # submit the lines to the auxeval judge 
    model_line_pairs = [(judge_model, line) for line in evaluation_lines]
    indices = [line["index"] for line in evaluation_lines]
    tmp_file =  args.precomputed_json.replace(".jsonl", "_auxeval.pkl")
    
    if args.dataset_type == DatasetType.MATHVISTA:
        new_results = track_progress_rich_with_rate_limit(
            mathvista_auxeval,
            tasks=model_line_pairs,
            nproc=args.gpt_eval_workers,
            rate_limit=10,  # 每秒最多5个请求
            warmup_duration=10,  
            keys=indices, 
            save=tmp_file,
        )
        ans = load(tmp_file)   # "res" and "log" from gpt-4o judge
        # insert them back to the original lines
        for j, line in enumerate(evaluation_lines):
            line["res"] = ans[line["index"]]["res"]
            line["log"] = ans[line["index"]]["log"]
            # TODO: only for debug usage -- remove in production
            line["prompt"] = ans[line["index"]]["prompt"]
            
            # if line["log"] == "Prefetch succeed" or post_check(line, prefetch=False):
            if post_check(line, prefetch=False):    
                line["correct_flag"] = 1
                correct_counter += 1
                for skill in line["skills"]:
                    per_type_correct_counter[skill] += 1
            else:
                line["correct_flag"] = 0
        
    elif args.dataset_type == DatasetType.MATHVERSE:
        # mathverse needs dedicated extract and score
        tmp_file_extract = tmp_file.replace(".pkl", "_extract.pkl")
        tmp_file_score = tmp_file.replace(".pkl", "_score.pkl")
        
        # step 1: extract
        results = track_progress_rich_with_rate_limit(
            MathVerse_auxeval_extract,
            tasks=model_line_pairs,
            nproc=args.gpt_eval_workers,
            rate_limit=20,  # 每秒最多5个请求
            warmup_duration=30,  # 30秒预热期
            keys=indices,
            save=tmp_file_extract,
        )
      
        ans = load(tmp_file_extract)
        # insert them back to the original lines
        for j, line in enumerate(evaluation_lines):
            line["extract"] = ans[line["index"]]["extract"]
            line["log_extract"] = ans[line["index"]]["log_extract"]
        
        new_model_line_pairs = [(judge_model, line) for line in evaluation_lines]
        # indices remains the same
        # step 2: score
        results = track_progress_rich_with_rate_limit(
            MathVerse_auxeval_score,
            tasks=new_model_line_pairs,
            nproc=args.gpt_eval_workers,
            rate_limit=20,  # 每秒最多5个请求
            warmup_duration=30,  # 30秒预热期
            keys=indices,
            save=tmp_file_score,
        )

        ans = load(tmp_file_score)
        for k, line in enumerate(evaluation_lines):
            line["score"] = ans[line["index"]]["score"]
            line["log_score"] = ans[line["index"]]["log_score"]
        
            # judgement = 1 --> correct; judgement = 0 --> incorrect
            if line["score"]:
                line["correct_flag"] = 1
                correct_counter += 1
            
    elif args.dataset_type == DatasetType.MATHVISION:
        results = track_progress_rich_with_rate_limit(
            MATH_V_auxeval,
            tasks=model_line_pairs,
            nproc=args.gpt_eval_workers,
            rate_limit=5,  # 每秒最多5个请求
            warmup_duration=30,  # 30秒预热期
            keys=indices,
            save=tmp_file,
        )

        ans = load(tmp_file)
        # insert them back to the original lines
        for j, line in enumerate(evaluation_lines):
            line["res"] = ans[line["index"]]["res"]
            line["log"] = ans[line["index"]]["log"]
            
            if post_check_mathvision(line, prefetch=False):
                line["correct_flag"] = 1
                correct_counter += 1
                for skill in line["skills"]:
                    per_type_correct_counter[skill] += 1
    
    else:
        raise NotImplementedError(f"Dataset {args.dataset_type} not supported.")
        
    acc = {"Total Accuracy": correct_counter / len(evaluation_lines)}
    
    if per_type_correct_counter:
        for skill, correct_count in per_type_correct_counter.items():
            acc[f"Accuracy on {skill}"] = correct_count / per_type_total_counter[skill]
            
    print(acc)
            
    # should be done -- save it to _gpt_parsed.jsonl and a final acc file
    output_parsed_path = args.precomputed_json.replace(".jsonl", "_gpt_parsed.jsonl")
    output_acc_path = args.precomputed_json.replace(".jsonl", "_gpt_acc.json")
    
    with open(output_parsed_path, "w", encoding="utf-8") as f:
        for line in evaluation_lines:
            f.write(f"{json.dumps(line)}\n")
            
    with open(output_acc_path, "w", encoding="utf-8") as f:
        json.dump(acc, f, indent=4)