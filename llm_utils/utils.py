import os
import json
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
