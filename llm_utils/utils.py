import os
import json
import random
import hashlib
import base64
import asyncio
from datetime import datetime, timedelta
try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
except:
    pass


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


class DateHandle:

    @staticmethod
    def get_random_date(start: str, end: str) -> str:
        """ 获得指定范围内的随机日期 YYYYMMDD 格式，包含start和end
        """
        start_date = datetime.strptime(start, "%Y%m%d")
        end_date = datetime.strptime(end, "%Y%m%d")
        if start_date > end_date:
            raise ValueError("Start date must be earlier than or equal to end date")
        delta = (end_date - start_date).days
        random_days = random.randint(0, delta)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%Y%m%d")

    @staticmethod
    def shift_date(base_date: str, days: int) -> str:
        """ 根据基准日期计算偏移后的日期 YYYYMMDD
        """
        base_datetime = datetime.strptime(base_date, "%Y%m%d")
        offset_date = base_datetime + timedelta(days=days)
        return offset_date.strftime("%Y%m%d")


class Tokenizer_Handle:

    def __init__(self, tokenizer):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
    
    def count_tokens(self, text: str|list[str], add_special_tokens = False) -> int|list[int]:
        """ str -> int | list -> list
        """
        encoded = self.tokenizer(text, 
                                 add_special_tokens=add_special_tokens, 
                                 return_length=True)
        token_lengths = encoded["length"]

        if isinstance(text, str):
            assert len(token_lengths) == 1
            return token_lengths[0]
        elif isinstance(text, list):
            return token_lengths
        else:
            raise ValueError(f"type(text) = {type(text)}")
    
    def cut_by_token_len(self, text: str|list[str], max_tokens: int) -> str|list[str]:
        """ str -> str | list -> list
        """
        text_lst = [text] if isinstance(text, str) else text
        encoded: list[list[int]] = self.tokenizer(text=text_lst, add_special_tokens=False, padding=False, truncation=True, max_length=max_tokens)["input_ids"]
        text_lst_truncated = self.tokenizer.batch_decode(encoded, skip_special_tokens=False)
        if isinstance(text, str):
            assert len(text_lst_truncated) == 1
            return text_lst_truncated[0]
        else:
            return text_lst_truncated

