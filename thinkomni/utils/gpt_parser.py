import requests
import json
import re
from time import sleep

from utils.model_parser import (
    extract_boxed_answer, 
    build_mathverse_extract_prompt, 
    build_score_prompt, 
    retry_with_backoff,
    build_extract_prompt,
    build_wemath_extract_prompt
)

GPT_URL = "http://10.221.105.108:48099/generate"
GPT_HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "35198cc7-22f1-4fdd-a9c4-326c2efe5f9e"  # 替换为您的API密钥
}

def chat_by_gpt(prompt, system_prompt="You are a helpful assistant.", temperature=0.0):
    payload = {
        "model": "gpt-4o",
        "key_id": 1,
        "prompt": prompt,
        "extra": {"temperature": temperature}
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt
    
    response = requests.post(GPT_URL, headers=GPT_HEADERS, data=json.dumps(payload))
    
    if response.status_code == 200:
        sleep(0.1)  # 确保不会过快地发送请求
        return response.json()['results'][0]  # return plain text
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def gpt_eval_score_retry(question, prediction, answer, dataset):
    if dataset.lower() == "mathverse":
        extracted_answer, boxed_flag = extract_boxed_answer(prediction)
        if not boxed_flag:
            extract_prompt = build_mathverse_extract_prompt(prediction)
            
            # 使用重试机制调用模型
            extracted_answer = retry_with_backoff(
                chat_by_gpt,
                max_retries=50,
                initial_delay=2,
                prompt=extract_prompt,
                temperature=0.0
            )

        score_prompt = build_score_prompt(question, extracted_answer, answer)
        
        # 直接调用生成内容，依靠外部重试逻辑
        response_text = retry_with_backoff(
            chat_by_gpt,
            max_retries=50,
            initial_delay=2,
            prompt=score_prompt,
            temperature=0.0
        ).strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0
    
    elif dataset.lower() in ["mathvista", "mathvision"]:
        extract_prompt = build_extract_prompt(prediction, question)
        
        # 使用重试机制调用模型
        extracted_answer = retry_with_backoff(
            chat_by_gpt,
            max_retries=50,
            initial_delay=2,
            contents=extract_prompt,
            generation_config={"temperature": 0.0}
        ).text

        if extracted_answer.strip() == answer:
            return 1.0
        else:
            return 0.0
        
    elif dataset.lower() == "wemath":
        extract_prompt = build_wemath_extract_prompt(prediction, question)
        
        # 使用重试机制调用模型
        response = retry_with_backoff(
            chat_by_gpt,
            max_retries=50,
            initial_delay=2,
            contents=extract_prompt,
            generation_config={"temperature": 0.0}
        )
        
        extracted_answer = response.text.strip().upper()
        
        if re.match(r'^[A-G]$', extracted_answer):
            accuracy = 1.0 if extracted_answer == answer else 0.0
            return accuracy
        else:
            return 0.0