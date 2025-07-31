import os
import json
import random
import hashlib
import base64
import asyncio


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [json.loads(i) for i in data]
    return data

def dump_json(obj, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    return

def dump_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as jsonl_file:
        if isinstance(data, list):
            for item in data:
                jsonl_file.write(json.dumps(item, ensure_ascii=False, indent=None) + '\n')
        else:
            jsonl_file.write(json.dumps(data, ensure_ascii=False, indent=None) + '\n')
    return


def sample_list(lst: list, k: int, mode="uniform"):
    # 从list中采样k个元素并保序返回，uniform模式保证每次返回结果一致
    if not lst:
        return []
    
    if k < 0:
        raise ValueError("采样数量k不能为负数")
    if k > len(lst):
        raise ValueError("采样数量k不能大于列表长度")
    if k == 0:
        return []
    
    if mode == "uniform":
        if k == 1:
            return [lst[0]]
        indices = [i * len(lst) // k for i in range(k)]
    elif mode == "random":
        indices = sorted(random.sample(range(len(lst)), k))
    else:
        raise ValueError("模式必须是 'random' 或 'uniform'")
    
    return [lst[i] for i in indices]


def get_sha256(input_str: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(input_str.encode('utf-8'))
    return hasher.hexdigest()
